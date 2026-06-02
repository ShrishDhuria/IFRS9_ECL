"""
IFRS 9 ECL Engine — Phase 4
============================
Extends Phase 3 with:
  - Macroeconomic scenario definitions (baseline, upside, adverse)
  - Rating transition-based PD stress (Option B: shift ratings by
    N notches under each scenario, then recompute staging + ECL)
  - Downturn LGD under stress (Option A: fixed regulatory uplift)
  - Probability-weighted ECL across scenarios (Option A: expert weights)
  - Full scenario decomposition + sensitivity table (Option C)

Data sources:
  - EBA/ESRB "Macro-financial scenario for the 2025 EU-wide stress test",
    January 2025 (adverse scenario macro paths)
  - ECB December 2024 projections (baseline)
  - S&P 2024 Annual Default Study, Table 20/22 (transition matrices)

Regulatory references:
  - IFRS 9 §5.5.17(c):     Forward-looking information requirement
  - IFRS 9 §B5.5.42-44:    Probability-weighted ECL, multiple scenarios
  - EBA GL/2017/06 §30-32:  Scenario design and probability weighting
  - EBA 2025 Stress Test Methodology Note §73-75: Staging under stress
  - ACPR IFRS 9 notice:     French supervisory expectations on scenarios

Run directly:  python ecl_engine_phase4.py
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import math

from ecl_engine_phase3 import (
    StagedLoan, classify_stage, SICRConfig, Stage,
    compute_portfolio_ecl, _rating_to_category,
    RATING_SCALE, RATING_TO_NOTCH, notch_distance,
)
from ecl_engine_phase2 import (
    LoanPhase2, lifetime_ecl, build_pd_term_structure,
)
from ecl_engine_phase1 import SP_1Y_PD, SP_1Y_PD_EUROPE


# ---------------------------------------------------------------------------
# Macroeconomic scenarios
# ---------------------------------------------------------------------------

@dataclass
class MacroScenario:
    """
    A forward-looking macroeconomic scenario.

    The key mechanism is the notch_shift: under each scenario, every
    loan's current rating is shifted by this many notches before staging
    and ECL are recomputed. This is the transition-based approach
    (Decision 1, Option B), which mirrors the EBA stress test methodology.

    Calibration rationale for notch shifts:
    ─────────────────────────────────────────
    The EBA 2025 adverse scenario projects cumulative EU real GDP
    deviation of -10.4% from baseline and unemployment +6.1pp.

    S&P data (Table 7) shows that in severe stress years:
      - 2009: downgrade rate was 19.1%, avg 1.7 notches per downgrade
      - 2020: downgrade rate was 18.5%
      - 2008: downgrade rate was 16.1%
    In those years, roughly 20% of the rated universe was downgraded,
    with the average downgrade being ~2 notches.

    A 2-notch shift on the entire portfolio under adverse is therefore
    a plausible central estimate — it's severe but not as extreme as
    applying the worst-year transition matrix, which is appropriate
    because the EBA scenario is designed to be severe but not
    catastrophic.

    For the upside, a 1-notch upgrade is conservative — upgrade rates
    in good years (2005-2007) averaged ~12% with ~1.1 notches per
    upgrade.
    """
    name: str
    probability: float         # scenario weight
    description: str
    notch_shift: int           # + = downgrade, - = upgrade
    lgd_override: Optional[float] = None  # if set, replaces base LGD

    # Macro path data (for documentation / sensitivity; not used in
    # the core calculation since we use notch_shift instead)
    gdp_growth: List[float] = field(default_factory=list)
    unemployment: List[float] = field(default_factory=list)


def build_scenarios() -> List[MacroScenario]:
    """
    Construct the three EBA-anchored scenarios.

    Weights: 50% baseline, 30% adverse, 20% upside.
    The ACPR expects adverse weight of 25-35%; 30% is the midpoint.
    """
    baseline = MacroScenario(
        name="Baseline",
        probability=0.50,
        description="ECB Dec-2024 projections: moderate growth continues",
        notch_shift=0,
        lgd_override=None,        # use loan's base LGD (45%)
        gdp_growth=[1.3, 1.6, 1.6],
        unemployment=[6.4, 6.2, 6.0],
    )

    adverse = MacroScenario(
        name="Adverse",
        probability=0.30,
        description=(
            "EBA/ESRB 2025 adverse: cum. GDP -6.3%, "
            "unemployment +6.1pp, equities -50%"
        ),
        notch_shift=2,            # 2-notch downgrade across the board
        lgd_override=0.55,        # downturn LGD (CRR-style uplift from 45%)
        gdp_growth=[-1.8, -4.3, -0.2],
        unemployment=[8.5, 10.5, 12.1],
    )

    upside = MacroScenario(
        name="Upside",
        probability=0.20,
        description="Symmetric upside: stronger growth, improving conditions",
        notch_shift=-1,           # 1-notch upgrade
        lgd_override=0.40,        # recovery rates improve in benign environment
        gdp_growth=[2.5, 3.0, 2.8],
        unemployment=[5.8, 5.3, 5.0],
    )

    return [baseline, adverse, upside]


# ---------------------------------------------------------------------------
# Rating shift mechanics
# ---------------------------------------------------------------------------

def shift_rating(rating: str, notches: int) -> str:
    """
    Shift a rating by N notches on the S&P scale.

    Positive notches = downgrade (worse).
    Negative notches = upgrade (better).

    Floors at CCC/C (can't go below without defaulting).
    Caps at AAA (can't go above).
    D is never shifted — a defaulted loan stays defaulted.
    """
    if rating == "D":
        return "D"
    if rating not in RATING_TO_NOTCH:
        return rating

    current_idx = RATING_TO_NOTCH[rating]
    new_idx = current_idx + notches

    # Floor: CCC/C (index 16); ceiling: AAA (index 0)
    # Don't shift into D (index 17) — that's a default event,
    # not a rating migration for ECL purposes
    new_idx = max(0, min(16, new_idx))

    return RATING_SCALE[new_idx]


# ---------------------------------------------------------------------------
# Scenario-level ECL computation
# ---------------------------------------------------------------------------

def compute_scenario_ecl(loans: List[StagedLoan],
                         scenario: MacroScenario,
                         config: SICRConfig = SICRConfig()) -> dict:
    """
    Compute portfolio ECL under a single macro scenario.

    Mechanism:
    1. For each loan, shift the current rating by scenario.notch_shift
    2. If scenario.lgd_override is set, apply the downturn LGD
    3. Re-run staging (the shifted rating may trigger new SICR events)
    4. Compute 12m or lifetime ECL based on new stage assignment
    5. Aggregate

    This naturally produces the key IFRS 9 dynamic: under adverse,
    more loans migrate into Stage 2 (because the shifted rating
    triggers the notch-downgrade SICR test), which increases ECL
    non-linearly via the cliff effect.
    """
    stressed_loans = []

    for loan in loans:
        stressed = deepcopy(loan)
        stressed.current_rating = shift_rating(
            loan.current_rating, scenario.notch_shift
        )

        # Apply downturn LGD if specified
        if scenario.lgd_override is not None:
            stressed.lgd = scenario.lgd_override
            if stressed.lgd_downturn is not None:
                # Stage 3 downturn LGD should be at least as high
                stressed.lgd_downturn = max(
                    stressed.lgd_downturn,
                    scenario.lgd_override + 0.10
                )

        stressed_loans.append(stressed)

    result = compute_portfolio_ecl(stressed_loans, config)

    # Enrich with scenario metadata
    result["scenario"] = scenario.name
    result["notch_shift"] = scenario.notch_shift
    result["lgd_override"] = scenario.lgd_override
    result["probability"] = scenario.probability

    # Add per-loan rating shift detail
    for i, loan_result in enumerate(result["loans"]):
        loan_result["orig_current_rating"] = loans[i].current_rating
        loan_result["stressed_rating"] = stressed_loans[i].current_rating
        loan_result["rating_shifted_by"] = scenario.notch_shift

    return result


# ---------------------------------------------------------------------------
# Probability-weighted ECL
# ---------------------------------------------------------------------------

def compute_weighted_ecl(loans: List[StagedLoan],
                         scenarios: Optional[List[MacroScenario]] = None,
                         config: SICRConfig = SICRConfig()) -> dict:
    """
    Compute probability-weighted ECL across multiple scenarios.

    This is the IFRS 9 §B5.5.42 calculation:
      ECL_weighted = Σ (probability_i × ECL_i)

    Returns scenario-level detail and the weighted aggregate.
    """
    if scenarios is None:
        scenarios = build_scenarios()

    # Validate weights sum to 1.0
    total_prob = sum(s.probability for s in scenarios)
    if abs(total_prob - 1.0) > 0.001:
        raise ValueError(
            f"Scenario probabilities sum to {total_prob:.3f}, not 1.0"
        )

    scenario_results = []
    weighted_ecl = 0.0

    for scenario in scenarios:
        result = compute_scenario_ecl(loans, scenario, config)
        scenario_results.append(result)
        weighted_ecl += scenario.probability * result["totals"]["total_ecl"]

    # Baseline ECL for comparison (the "IAS 39 equivalent" — single scenario)
    baseline_result = scenario_results[0]  # assumes baseline is first
    baseline_ecl = baseline_result["totals"]["total_ecl"]

    # The "IFRS 9 overlay" = weighted ECL - baseline ECL
    # This quantifies the impact of forward-looking scenario weighting
    overlay = weighted_ecl - baseline_ecl

    return {
        "scenarios": scenario_results,
        "weighted_ecl": round(weighted_ecl, 2),
        "baseline_ecl": round(baseline_ecl, 2),
        "ifrs9_overlay": round(overlay, 2),
        "overlay_pct": round(overlay / baseline_ecl * 100, 1) if baseline_ecl > 0 else 0,
        "total_ead": scenario_results[0]["totals"]["total_ead"],
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis: vary adverse probability
# ---------------------------------------------------------------------------

def adverse_probability_sensitivity(
        loans: List[StagedLoan],
        prob_range: List[float] = None,
        config: SICRConfig = SICRConfig()) -> List[dict]:
    """
    Vary the adverse scenario probability from 10% to 50% and show
    how weighted ECL changes.

    For each adverse probability p_adv:
      - Upside probability stays at 20% (capped so total <= 100%)
      - Baseline = 1 - p_adv - p_upside

    This table demonstrates that scenario weights are the most
    judgmental input in the IFRS 9 framework — and the one European
    supervisors scrutinise most intensely.
    """
    if prob_range is None:
        prob_range = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = []

    for p_adv in prob_range:
        p_upside = min(0.20, 1.0 - p_adv - 0.01)  # cap upside
        p_baseline = 1.0 - p_adv - p_upside

        scenarios = build_scenarios()
        scenarios[0].probability = round(p_baseline, 2)
        scenarios[1].probability = round(p_adv, 2)
        scenarios[2].probability = round(p_upside, 2)

        weighted = compute_weighted_ecl(loans, scenarios, config)

        results.append({
            "p_adverse": p_adv,
            "p_baseline": p_baseline,
            "p_upside": p_upside,
            "weighted_ecl": weighted["weighted_ecl"],
            "overlay": weighted["ifrs9_overlay"],
            "overlay_pct": weighted["overlay_pct"],
        })

    return results


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_weighted_ecl(result: dict) -> None:
    """Pretty-print the probability-weighted ECL."""
    print(f"\n{'='*100}")
    print(f"  IFRS 9 — Probability-Weighted ECL Across Macroeconomic Scenarios")
    print(f"{'='*100}")

    # Scenario-level summary
    print(f"\n  {'Scenario':<12} {'Weight':>7} {'Shift':>6} {'LGD':>6} "
          f"{'S1 ECL':>12} {'S2 ECL':>12} {'S3 ECL':>12} {'Total ECL':>14}")
    print(f"  {'─'*12} {'─'*7} {'─'*6} {'─'*6} "
          f"{'─'*12} {'─'*12} {'─'*12} {'─'*14}")

    for sr in result["scenarios"]:
        sc = sr["scenario"]
        prob = sr["probability"]
        shift = sr["notch_shift"]
        lgd = sr["lgd_override"] or 0.45
        t = sr["totals"]
        print(f"  {sc:<12} {prob:>6.0%} {shift:>+5}  {lgd:>5.0%} "
              f"€{t['stage1_ecl']:>10,.0f} €{t['stage2_ecl']:>10,.0f} "
              f"€{t['stage3_ecl']:>10,.0f} €{t['total_ecl']:>12,.0f}")

    print(f"{'─'*100}")
    print(f"  {'Weighted ECL':.<60} €{result['weighted_ecl']:>12,.0f}")
    print(f"  {'Baseline-only ECL':.<60} €{result['baseline_ecl']:>12,.0f}")
    print(f"  {'IFRS 9 overlay (weighted − baseline)':.<60} €{result['ifrs9_overlay']:>12,.0f}"
          f"  ({result['overlay_pct']:+.1f}%)")
    print(f"{'='*100}")

    # Stage migration detail
    print(f"\n  Stage migration under each scenario:")
    print(f"  {'─'*90}")
    for sr in result["scenarios"]:
        sc = sr["scenario"]
        t = sr["totals"]
        s1_pct = t["stage1_ead"] / t["total_ead"] * 100 if t["total_ead"] > 0 else 0
        s2_pct = t["stage2_ead"] / t["total_ead"] * 100 if t["total_ead"] > 0 else 0
        s3_pct = t["stage3_ead"] / t["total_ead"] * 100 if t["total_ead"] > 0 else 0
        print(f"  {sc:<12}  S1: {t['stage1_count']} loans ({s1_pct:.0f}% EAD)  "
              f"S2: {t['stage2_count']} loans ({s2_pct:.0f}% EAD)  "
              f"S3: {t['stage3_count']} loans ({s3_pct:.0f}% EAD)")

    # Loan-level rating shifts under adverse
    print(f"\n  Rating shifts under adverse scenario:")
    print(f"  {'─'*90}")
    adverse_loans = result["scenarios"][1]["loans"]
    print(f"  {'Loan':<30} {'Current':>8} {'Stressed':>8} {'Shift':>6} "
          f"{'Base Stage':>10} {'Stress Stage':>12}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*6} {'─'*10} {'─'*12}")

    baseline_loans = result["scenarios"][0]["loans"]
    for bl, al in zip(baseline_loans, adverse_loans):
        print(f"  {al['loan_name']:<30} {al['orig_current_rating']:>8} "
              f"{al['curr_rating']:>8} {al['rating_shifted_by']:>+5} "
              f"{'S' + str(bl['stage'].value):>10} "
              f"{'S' + str(al['stage'].value):>12}")


def print_sensitivity(sensitivity: List[dict]) -> None:
    """Pretty-print the adverse probability sensitivity table."""
    print(f"\n{'='*80}")
    print(f"  Sensitivity: Weighted ECL vs Adverse Scenario Probability")
    print(f"{'='*80}")
    print(f"\n  {'P(Adverse)':>10} {'P(Base)':>8} {'P(Up)':>7} "
          f"{'Weighted ECL':>14} {'Overlay':>12} {'Overlay %':>10}")
    print(f"  {'─'*10} {'─'*8} {'─'*7} {'─'*14} {'─'*12} {'─'*10}")

    for r in sensitivity:
        print(f"  {r['p_adverse']:>9.0%} {r['p_baseline']:>7.0%} "
              f"{r['p_upside']:>6.0%} €{r['weighted_ecl']:>12,.0f} "
              f"€{r['overlay']:>10,.0f} {r['overlay_pct']:>+8.1f}%")

    # Highlight the range
    ecl_min = min(r['weighted_ecl'] for r in sensitivity)
    ecl_max = max(r['weighted_ecl'] for r in sensitivity)
    print(f"\n  ECL range:  €{ecl_min:,.0f}  →  €{ecl_max:,.0f}  "
          f"(spread: €{ecl_max - ecl_min:,.0f}, "
          f"{(ecl_max/ecl_min - 1)*100:.0f}% variation)")
    print(f"\n  This spread quantifies the model risk from scenario weights —")
    print(f"  the single most judgmental input in IFRS 9 provisioning.")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Main: run the full Phase 4 analysis
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Same portfolio as Phase 3
    portfolio = [
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
        StagedLoan(
            name="GammaRetail SAS",
            principal=5_000_000,
            origination_rating="BBB",
            current_rating="BB-",
            lgd=0.45, eir=0.045, maturity=5,
            amort_type="bullet", region="europe",
        ),
        StagedLoan(
            name="DeltaConstruction SpA",
            principal=1_500_000,
            origination_rating="BB",
            current_rating="BB",
            lgd=0.45, eir=0.050, maturity=4,
            days_past_due=35,
            amort_type="bullet", region="europe",
        ),
        StagedLoan(
            name="EpsilonEnergy BV",
            principal=4_000_000,
            origination_rating="BBB+",
            current_rating="BBB-",
            lgd=0.45, eir=0.040, maturity=6,
            qualitative_flags=["watchlist"],
            amort_type="bullet", region="europe",
        ),
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

    # ── 1. Probability-weighted ECL ──
    weighted = compute_weighted_ecl(portfolio)
    print_weighted_ecl(weighted)

    # ── 2. Sensitivity analysis ──
    sensitivity = adverse_probability_sensitivity(portfolio)
    print_sensitivity(sensitivity)
