"""
IFRS 9 ECL Engine — Phase 3
============================
Extends Phase 2 with:
  - Stage classification engine (Stage 1 / 2 / 3)
  - SICR (Significant Increase in Credit Risk) detection
  - Qualitative and quantitative triggers
  - 30-day and 90-day past-due backstops
  - Portfolio-level ECL aggregation by stage

Regulatory references:
  - IFRS 9 §5.5.3–5.5.5:  Stage definitions and ECL measurement
  - IFRS 9 §5.5.9:        SICR — lifetime ECL trigger
  - IFRS 9 §5.5.11:       30-day rebuttable presumption
  - IFRS 9 §B5.5.17–18:   Relative vs absolute PD change
  - IFRS 9 Appendix A:    Definition of credit-impaired
  - CRR Art. 178:         Definition of default (90 DPD backstop)
  - EBA GL/2017/06:       Guidelines on ECL practices
  - ACPR IFRS 9 notice:   French supervisory expectations

Run directly:  python ecl_engine_phase3.py
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

# Import Phase 2 functions
from ecl_engine_phase2 import (
    LoanPhase2, lifetime_ecl, build_pd_term_structure,
    CUMULATIVE_PD_TABLE, CUMULATIVE_PD_EUROPE,
)
from ecl_engine_phase1 import SP_1Y_PD, SP_1Y_PD_EUROPE


# ---------------------------------------------------------------------------
# S&P Rating Scale — notch mapping for migration distance
# ---------------------------------------------------------------------------

RATING_SCALE = [
    "AAA", "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC/C", "D",
]

RATING_TO_NOTCH = {r: i for i, r in enumerate(RATING_SCALE)}


def notch_distance(rating_from: str, rating_to: str) -> int:
    """
    Compute the signed notch distance between two ratings.
    Positive = downgrade (worse), negative = upgrade (better).

    Example: BBB → BB+ = +1 notch downgrade
             BBB → A-  = -1 notch upgrade
    """
    if rating_from not in RATING_TO_NOTCH:
        raise ValueError(f"Unknown rating: {rating_from}")
    if rating_to not in RATING_TO_NOTCH:
        raise ValueError(f"Unknown rating: {rating_to}")
    return RATING_TO_NOTCH[rating_to] - RATING_TO_NOTCH[rating_from]


def is_investment_grade(rating: str) -> bool:
    """BBB- and above are investment grade."""
    return RATING_TO_NOTCH.get(rating, 99) <= RATING_TO_NOTCH["BBB-"]


# ---------------------------------------------------------------------------
# SICR configuration — calibrated to European bank practice
# ---------------------------------------------------------------------------

@dataclass
class SICRConfig:
    """
    Configurable SICR thresholds.

    These defaults reflect common practice at large French banks
    (SG, BNP, Natixis, CA-CIB) as described in EBA GL/2017/06
    and ACPR peer reviews.

    Teaching note: in an interview, you should be able to explain
    WHY each threshold exists, not just WHAT it is.
    """
    # ── Quantitative: rating migration ──
    # A downgrade of this many notches from origination triggers SICR.
    # 2 notches is standard (e.g., BBB → BBB- → BB+ = 2 notches).
    notch_downgrade_threshold: int = 2

    # ── Quantitative: PD-based ──
    # SICR if BOTH conditions are met:
    #   (1) current PD / origination PD >= pd_relative_threshold  (PD has doubled)
    #   (2) current PD - origination PD >= pd_absolute_threshold  (meaningful in absolute terms)
    # This dual test comes from IFRS 9 B5.5.17: a relative change alone is
    # insufficient if the absolute change is trivial.
    pd_relative_threshold: float = 2.0     # PD must at least double
    pd_absolute_threshold: float = 0.005   # AND increase by ≥50 bps

    # ── Backstops: days past due ──
    # 30 DPD: rebuttable presumption of SICR (IFRS 9 §5.5.11)
    # 90 DPD: default / Stage 3 (CRR Art. 178)
    dpd_stage2_threshold: int = 30
    dpd_stage3_threshold: int = 90

    # ── Qualitative flags that force Stage 2 ──
    # In practice these come from the bank's watchlist/early warning system
    qualitative_stage2_triggers: Tuple[str, ...] = (
        "watchlist",
        "forbearance",
        "sector_stress",
        "covenant_breach",
    )

    # ── Qualitative flags that force Stage 3 ──
    qualitative_stage3_triggers: Tuple[str, ...] = (
        "bankruptcy",
        "distressed_restructuring",
        "fraud",
        "unlikely_to_pay",
    )

    # ── Low credit risk exemption (IFRS 9 §5.5.10) ──
    # If current rating is investment grade, some banks apply a
    # "low credit risk" exemption and keep the loan in Stage 1
    # regardless of migration. The ACPR discourages over-reliance
    # on this exemption, but it exists in the standard.
    allow_low_credit_risk_exemption: bool = False


# ---------------------------------------------------------------------------
# Stage classification engine
# ---------------------------------------------------------------------------

class Stage(Enum):
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3


@dataclass
class StagingResult:
    """Full audit trail of a staging decision."""
    stage: Stage
    triggers_fired: List[str]
    detail: dict


@dataclass
class StagedLoan:
    """A loan with all attributes needed for staging and ECL."""
    name: str
    principal: float
    origination_rating: str   # rating at initial recognition
    current_rating: str       # rating at reporting date
    lgd: float
    eir: float
    maturity: int             # remaining maturity (years)
    amort_type: str = "bullet"
    region: str = "europe"
    days_past_due: int = 0
    qualitative_flags: List[str] = field(default_factory=list)

    # Stage 3 override: a different (downturn) LGD is often used
    lgd_downturn: Optional[float] = None


def classify_stage(loan: StagedLoan,
                   config: SICRConfig = SICRConfig()) -> StagingResult:
    """
    Determine the IFRS 9 stage for a loan.

    The logic follows a waterfall — we check Stage 3 triggers first
    (most severe), then Stage 2, then default to Stage 1. This mirrors
    how banks implement it operationally.

    Returns a StagingResult with full audit trail of which triggers
    fired, so you can explain each decision to an auditor or interviewer.
    """
    triggers = []
    detail = {
        "origination_rating": loan.origination_rating,
        "current_rating": loan.current_rating,
        "notch_change": notch_distance(loan.origination_rating, loan.current_rating),
        "days_past_due": loan.days_past_due,
        "qualitative_flags": loan.qualitative_flags,
    }

    # ── STAGE 3 CHECKS (credit-impaired) ──

    # 3a. 90+ days past due (CRR Art. 178 default definition)
    if loan.days_past_due >= config.dpd_stage3_threshold:
        triggers.append(f"DPD >= {config.dpd_stage3_threshold} (CRR Art. 178)")

    # 3b. Current rating is D (already in default)
    if loan.current_rating == "D":
        triggers.append("Rating = D (default)")

    # 3c. Qualitative Stage 3 flags
    for flag in loan.qualitative_flags:
        if flag in config.qualitative_stage3_triggers:
            triggers.append(f"Qualitative flag: {flag}")

    if triggers:
        return StagingResult(
            stage=Stage.STAGE_3,
            triggers_fired=triggers,
            detail=detail,
        )

    # ── STAGE 2 CHECKS (significant increase in credit risk) ──

    # 2a. Rating migration: downgrade by N+ notches
    notches = notch_distance(loan.origination_rating, loan.current_rating)
    detail["notch_change"] = notches
    if notches >= config.notch_downgrade_threshold:
        triggers.append(
            f"Downgrade of {notches} notches "
            f"({loan.origination_rating} → {loan.current_rating}) "
            f">= threshold of {config.notch_downgrade_threshold}"
        )

    # 2b. PD-based test (relative AND absolute)
    pd_orig = _get_1y_pd(loan.origination_rating, loan.region)
    pd_curr = _get_1y_pd(loan.current_rating, loan.region)
    detail["pd_origination"] = pd_orig
    detail["pd_current"] = pd_curr

    if pd_orig > 0:
        pd_ratio = pd_curr / pd_orig
        pd_abs_change = pd_curr - pd_orig
        detail["pd_ratio"] = round(pd_ratio, 2)
        detail["pd_abs_change"] = round(pd_abs_change, 6)

        if (pd_ratio >= config.pd_relative_threshold and
                pd_abs_change >= config.pd_absolute_threshold):
            triggers.append(
                f"PD ratio {pd_ratio:.1f}x >= {config.pd_relative_threshold:.1f}x "
                f"AND absolute Δ {pd_abs_change:.2%} >= {config.pd_absolute_threshold:.2%}"
            )

    # 2c. 30+ days past due (rebuttable presumption)
    if loan.days_past_due >= config.dpd_stage2_threshold:
        triggers.append(
            f"DPD = {loan.days_past_due} >= {config.dpd_stage2_threshold} "
            f"(IFRS 9 §5.5.11 rebuttable presumption)"
        )

    # 2d. Qualitative Stage 2 flags
    for flag in loan.qualitative_flags:
        if flag in config.qualitative_stage2_triggers:
            triggers.append(f"Qualitative flag: {flag}")

    # 2e. Low credit risk exemption check
    if triggers and config.allow_low_credit_risk_exemption:
        if is_investment_grade(loan.current_rating):
            detail["low_credit_risk_exemption"] = True
            triggers_note = triggers.copy()
            triggers = [
                f"SICR triggers overridden by low-credit-risk exemption "
                f"(current rating {loan.current_rating} is IG) — "
                f"IFRS 9 §5.5.10. Original triggers: {triggers_note}"
            ]
            return StagingResult(
                stage=Stage.STAGE_1,
                triggers_fired=triggers,
                detail=detail,
            )

    if triggers:
        return StagingResult(
            stage=Stage.STAGE_2,
            triggers_fired=triggers,
            detail=detail,
        )

    # ── STAGE 1 (default) ──
    return StagingResult(
        stage=Stage.STAGE_1,
        triggers_fired=["No SICR triggers fired — loan remains performing"],
        detail=detail,
    )


def _get_1y_pd(rating: str, region: str) -> float:
    """Look up 1-year PD, falling back to global if not in European table."""
    if rating == "D":
        return 1.0
    if region == "europe" and rating in SP_1Y_PD_EUROPE:
        return SP_1Y_PD_EUROPE[rating]
    if rating in SP_1Y_PD:
        return SP_1Y_PD[rating]
    # For ratings not in the level table, map to category
    for cat_rating in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]:
        if rating.startswith(cat_rating.replace("/C", "")):
            if region == "europe" and cat_rating in SP_1Y_PD_EUROPE:
                return SP_1Y_PD_EUROPE[cat_rating]
            return SP_1Y_PD.get(cat_rating, 0.0)
    return 0.0


def _rating_to_category(rating: str) -> str:
    """Map a fine rating (e.g. BBB+) to a broad category (e.g. BBB)
    for cumulative PD lookup, since Table 24/25 uses categories."""
    categories = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"]
    if rating in categories:
        return rating
    if rating == "D":
        return "CCC/C"
    for cat in categories:
        base = cat.replace("/C", "")
        if rating.startswith(base):
            return cat
    return "B"  # fallback


# ---------------------------------------------------------------------------
# Portfolio ECL with staging
# ---------------------------------------------------------------------------

def compute_portfolio_ecl(loans: List[StagedLoan],
                          config: SICRConfig = SICRConfig()) -> dict:
    """
    Classify each loan, compute its ECL (12m or lifetime depending
    on stage), and aggregate.

    Returns portfolio-level summary and loan-level detail.
    """
    results = []
    totals = {
        "total_ead": 0.0,
        "stage1_ead": 0.0, "stage1_ecl": 0.0, "stage1_count": 0,
        "stage2_ead": 0.0, "stage2_ecl": 0.0, "stage2_count": 0,
        "stage3_ead": 0.0, "stage3_ecl": 0.0, "stage3_count": 0,
        "total_ecl": 0.0,
    }

    for loan in loans:
        staging = classify_stage(loan, config)

        # Map current rating to PD table category for lifetime ECL
        cat_rating = _rating_to_category(loan.current_rating)

        # Build the LoanPhase2 object for ECL calculation
        lgd_used = loan.lgd
        if staging.stage == Stage.STAGE_3 and loan.lgd_downturn:
            lgd_used = loan.lgd_downturn

        loan_p2 = LoanPhase2(
            name=loan.name,
            principal=loan.principal,
            rating=cat_rating,
            lgd=lgd_used,
            eir=loan.eir,
            maturity=loan.maturity,
            amort_type=loan.amort_type,
            region=loan.region,
        )

        ecl_result = lifetime_ecl(loan_p2)

        if staging.stage == Stage.STAGE_1:
            ecl_value = ecl_result["ecl_12m"]
        else:
            ecl_value = ecl_result["ecl_lifetime"]

        stage_key = f"stage{staging.stage.value}"
        totals[f"{stage_key}_ead"] += loan.principal
        totals[f"{stage_key}_ecl"] += ecl_value
        totals[f"{stage_key}_count"] += 1
        totals["total_ead"] += loan.principal
        totals["total_ecl"] += ecl_value

        results.append({
            "loan_name":    loan.name,
            "principal":    loan.principal,
            "orig_rating":  loan.origination_rating,
            "curr_rating":  loan.current_rating,
            "dpd":          loan.days_past_due,
            "flags":        loan.qualitative_flags,
            "stage":        staging.stage,
            "triggers":     staging.triggers_fired,
            "ecl_12m":      ecl_result["ecl_12m"],
            "ecl_lifetime": ecl_result["ecl_lifetime"],
            "ecl_applied":  round(ecl_value, 2),
            "lgd_used":     lgd_used,
        })

    # Coverage ratios
    for s in [1, 2, 3]:
        ead = totals[f"stage{s}_ead"]
        ecl = totals[f"stage{s}_ecl"]
        totals[f"stage{s}_coverage"] = round(ecl / ead * 100, 4) if ead > 0 else 0.0

    totals["total_coverage"] = round(
        totals["total_ecl"] / totals["total_ead"] * 100, 4
    ) if totals["total_ead"] > 0 else 0.0

    return {"loans": results, "totals": totals}


def print_portfolio(portfolio: dict) -> None:
    """Pretty-print the portfolio ECL breakdown."""
    loans = portfolio["loans"]
    totals = portfolio["totals"]

    print(f"\n{'='*110}")
    print(f"  IFRS 9 — Portfolio ECL with Stage Classification")
    print(f"{'='*110}")

    # Loan-level detail
    print(f"\n  {'Loan':<35} {'Orig':>5} {'Curr':>5} {'DPD':>4} "
          f"{'Stage':>7} {'EAD':>14} {'12m ECL':>12} {'Life ECL':>12} {'Applied':>12}")
    print(f"  {'─'*35} {'─'*5} {'─'*5} {'─'*4} "
          f"{'─'*7} {'─'*14} {'─'*12} {'─'*12} {'─'*12}")

    for r in loans:
        stage_str = f"S{r['stage'].value}"
        print(f"  {r['loan_name']:<35} {r['orig_rating']:>5} {r['curr_rating']:>5} "
              f"{r['dpd']:>4} {stage_str:>7} "
              f"€{r['principal']:>12,.0f} €{r['ecl_12m']:>10,.2f} "
              f"€{r['ecl_lifetime']:>10,.2f} €{r['ecl_applied']:>10,.2f}")

    # Triggers detail
    print(f"\n  Staging audit trail:")
    print(f"  {'─'*100}")
    for r in loans:
        stage_str = f"Stage {r['stage'].value}"
        print(f"  {r['loan_name']}: {stage_str}")
        for t in r['triggers']:
            print(f"    → {t}")

    # Portfolio summary
    print(f"\n{'─'*110}")
    print(f"  PORTFOLIO SUMMARY BY STAGE")
    print(f"{'─'*110}")
    print(f"  {'Stage':<10} {'Count':>6} {'EAD':>16} {'ECL':>14} {'Coverage':>10}")
    print(f"  {'─'*10} {'─'*6} {'─'*16} {'─'*14} {'─'*10}")

    for s in [1, 2, 3]:
        count = totals[f"stage{s}_count"]
        ead = totals[f"stage{s}_ead"]
        ecl = totals[f"stage{s}_ecl"]
        cov = totals[f"stage{s}_coverage"]
        print(f"  Stage {s:<4} {count:>6} €{ead:>14,.0f} €{ecl:>12,.2f} {cov:>8.3f}%")

    print(f"  {'─'*10} {'─'*6} {'─'*16} {'─'*14} {'─'*10}")
    print(f"  {'Total':<10} {sum(totals[f'stage{s}_count'] for s in [1,2,3]):>6} "
          f"€{totals['total_ead']:>14,.0f} €{totals['total_ecl']:>12,.2f} "
          f"{totals['total_coverage']:>8.3f}%")
    print(f"{'='*110}\n")


# ---------------------------------------------------------------------------
# Example: a small illustrative portfolio
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    portfolio_loans = [
        # ── Stage 1: performing, no deterioration ──
        StagedLoan(
            name="AlphaIndustries SA",
            principal=2_000_000,
            origination_rating="BBB",
            current_rating="BBB",
            lgd=0.45, eir=0.035, maturity=5,
            amort_type="bullet", region="europe",
        ),
        StagedLoan(
            name="BetaLogistics GmbH",
            principal=3_000_000,
            origination_rating="A",
            current_rating="A-",
            lgd=0.45, eir=0.030, maturity=7,
            amort_type="amortising", region="europe",
        ),

        # ── Stage 2: SICR triggered ──
        # 2a. Rating downgrade (BBB → BB-: 3 notch downgrade)
        StagedLoan(
            name="GammaRetail SAS",
            principal=5_000_000,
            origination_rating="BBB",
            current_rating="BB-",
            lgd=0.45, eir=0.045, maturity=5,
            amort_type="bullet", region="europe",
        ),
        # 2b. 35 days past due (30-day backstop fires)
        StagedLoan(
            name="DeltaConstruction SpA",
            principal=1_500_000,
            origination_rating="BB",
            current_rating="BB",
            lgd=0.45, eir=0.050, maturity=4,
            days_past_due=35,
            amort_type="bullet", region="europe",
        ),
        # 2c. Watchlist flag
        StagedLoan(
            name="EpsilonEnergy BV",
            principal=4_000_000,
            origination_rating="BBB+",
            current_rating="BBB-",
            lgd=0.45, eir=0.040, maturity=6,
            qualitative_flags=["watchlist"],
            amort_type="bullet", region="europe",
        ),

        # ── Stage 3: credit-impaired ──
        # 3a. 95 days past due
        StagedLoan(
            name="ZetaServices SARL",
            principal=800_000,
            origination_rating="BB",
            current_rating="CCC/C",
            lgd=0.45, lgd_downturn=0.65,
            eir=0.060, maturity=3,
            days_past_due=95,
            amort_type="bullet", region="europe",
        ),
        # 3b. Distressed restructuring flag
        StagedLoan(
            name="EtaHoldings AG",
            principal=2_500_000,
            origination_rating="BB+",
            current_rating="B-",
            lgd=0.45, lgd_downturn=0.65,
            eir=0.055, maturity=5,
            qualitative_flags=["distressed_restructuring"],
            amort_type="bullet", region="europe",
        ),
    ]

    # Run the portfolio
    result = compute_portfolio_ecl(portfolio_loans)
    print_portfolio(result)

    # ── Demonstrate the cliff effect: re-run GammaRetail as if it were still Stage 1 ──
    print("\n" + "="*80)
    print("  CLIFF EFFECT DEMONSTRATION: GammaRetail SAS")
    print("="*80)
    gamma = portfolio_loans[2]
    staging = classify_stage(gamma)
    gamma_ecl = [r for r in result["loans"] if r["loan_name"] == gamma.name][0]
    print(f"\n  Origination rating:  {gamma.origination_rating}")
    print(f"  Current rating:      {gamma.current_rating}")
    print(f"  Notch downgrade:     {notch_distance(gamma.origination_rating, gamma.current_rating)}")
    print(f"  Stage assigned:      {staging.stage.value}")
    print(f"  Triggers:            {staging.triggers_fired[0]}")
    print(f"\n  If Stage 1 (12m ECL):    €{gamma_ecl['ecl_12m']:>10,.2f}")
    print(f"  As Stage 2 (Lifetime):   €{gamma_ecl['ecl_lifetime']:>10,.2f}")
    print(f"  P&L impact of staging:   €{gamma_ecl['ecl_lifetime'] - gamma_ecl['ecl_12m']:>10,.2f}")
    print(f"  Cliff multiplier:        {gamma_ecl['ecl_lifetime']/gamma_ecl['ecl_12m']:.1f}x")
    print(f"{'='*80}\n")
