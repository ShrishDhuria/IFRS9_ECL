"""
IFRS 9 ECL Engine — Phase 2
============================
Extends Phase 1 with:
  - PD term structure derived from S&P cumulative default rates
  - Amortisation schedules (bullet and annuity)
  - Lifetime ECL for Stage 2 / Stage 3 exposures

Data source: S&P Global Ratings, "2024 Annual Global Corporate Default
And Rating Transition Study", March 27, 2025 — Table 24 (global average
cumulative default rates by rating category, 1981-2024).

Run directly:  python ecl_engine_phase2.py
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math


# ---------------------------------------------------------------------------
# S&P Table 24: global average cumulative default rates (%) by rating
# category, years 1-15.  Source: page 55-56 of the 2024 study.
# ---------------------------------------------------------------------------
CUMULATIVE_PD_TABLE = {
    "AAA":   [0.00, 0.03, 0.13, 0.23, 0.34, 0.44, 0.49, 0.57, 0.62, 0.67,
              0.70, 0.73, 0.75, 0.81, 0.86],
    "AA":    [0.02, 0.05, 0.11, 0.19, 0.28, 0.37, 0.45, 0.52, 0.59, 0.65,
              0.71, 0.76, 0.81, 0.86, 0.90],
    "A":     [0.05, 0.11, 0.19, 0.29, 0.39, 0.51, 0.65, 0.78, 0.90, 1.03,
              1.14, 1.25, 1.35, 1.45, 1.56],
    "BBB":   [0.14, 0.38, 0.67, 1.01, 1.36, 1.71, 2.00, 2.30, 2.58, 2.86,
              3.13, 3.35, 3.56, 3.78, 4.01],
    "BB":    [0.56, 1.76, 3.12, 4.48, 5.75, 6.93, 7.94, 8.86, 9.68, 10.44,
              11.06, 11.65, 12.17, 12.60, 13.05],
    "B":     [2.93, 6.93, 10.46, 13.31, 15.60, 17.45, 18.90, 20.06, 21.08,
              22.02, 22.82, 23.43, 24.02, 24.57, 25.11],
    "CCC/C": [26.12, 35.92, 41.32, 44.35, 46.53, 47.57, 48.61, 49.29,
              49.89, 50.43, 50.85, 51.32, 51.86, 52.26, 52.30],
}

# European cumulative PDs from Table 25 (category level, Y1-Y10)
CUMULATIVE_PD_EUROPE = {
    "AAA":   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "AA":    [0.00, 0.02, 0.05, 0.09, 0.14, 0.19, 0.22, 0.24, 0.27, 0.27],
    "A":     [0.03, 0.06, 0.08, 0.13, 0.19, 0.24, 0.31, 0.34, 0.35, 0.36],
    "BBB":   [0.05, 0.14, 0.28, 0.41, 0.56, 0.76, 0.93, 1.09, 1.27, 1.43],
    "BB":    [0.36, 1.15, 1.90, 2.61, 3.47, 4.29, 4.94, 5.35, 5.72, 6.13],
    "B":     [1.75, 4.72, 7.47, 9.82, 11.88, 13.46, 14.67, 15.53, 16.28, 16.81],
    "CCC/C": [26.26, 36.23, 41.57, 45.22, 47.27, 47.82, 48.12, 48.45, 48.45, 48.93],
}


# ---------------------------------------------------------------------------
# PD term structure derivation
# ---------------------------------------------------------------------------

def build_pd_term_structure(rating: str, max_years: int = 15,
                            region: str = "global") -> List[dict]:
    """
    Derive marginal and forward PDs from cumulative default rates.

    For each year t (1 to max_years):
      - Cumulative PD:   C(t)  — probability of default by end of year t
      - Survival prob:   S(t)  = 1 - C(t)
      - Forward PD:      fPD(t) = S(t-1) - S(t)  [unconditional]
      - Marginal PD:     mPD(t) = fPD(t) / S(t-1) [conditional on survival]

    Parameters
    ----------
    rating : str     — S&P rating category (e.g. "BBB", "BB")
    max_years : int  — number of years (capped by available data)
    region : str     — "global" uses Table 24, "europe" uses Table 25

    Returns
    -------
    List of dicts, one per year, with all derived quantities.
    """
    table = CUMULATIVE_PD_EUROPE if region == "europe" else CUMULATIVE_PD_TABLE

    if rating not in table:
        raise ValueError(
            f"Rating '{rating}' not in {region} table. "
            f"Available: {list(table.keys())}"
        )

    cum_pds_pct = table[rating]
    n = min(max_years, len(cum_pds_pct))
    results = []

    for t in range(1, n + 1):
        cum_pd = cum_pds_pct[t - 1] / 100.0  # convert % to decimal
        cum_pd_prev = cum_pds_pct[t - 2] / 100.0 if t > 1 else 0.0

        surv = 1.0 - cum_pd
        surv_prev = 1.0 - cum_pd_prev

        forward_pd = surv_prev - surv          # unconditional
        marginal_pd = forward_pd / surv_prev if surv_prev > 0 else 0.0  # conditional

        results.append({
            "year":        t,
            "cum_pd":      cum_pd,
            "survival":    surv,
            "forward_pd":  forward_pd,
            "marginal_pd": marginal_pd,
        })

    return results


# ---------------------------------------------------------------------------
# Amortisation schedules
# ---------------------------------------------------------------------------

def bullet_schedule(principal: float, eir: float, maturity: int) -> List[dict]:
    """
    Bullet (interest-only) loan: full principal repaid at maturity.
    EAD is constant until the final year.
    """
    schedule = []
    for t in range(1, maturity + 1):
        interest = principal * eir
        princ_repay = principal if t == maturity else 0.0
        ead_bop = principal  # balance at beginning of period
        schedule.append({
            "year":          t,
            "ead_bop":       ead_bop,
            "interest":      round(interest, 2),
            "principal":     round(princ_repay, 2),
            "instalment":    round(interest + princ_repay, 2),
            "ead_eop":       round(ead_bop - princ_repay, 2),
        })
        principal -= princ_repay
    return schedule


def amortising_schedule(principal: float, eir: float,
                        maturity: int) -> List[dict]:
    """
    Fully amortising loan with equal annual instalments (French amortisation).
    This is the standard structure for many European corporate term loans.
    """
    if eir == 0:
        instalment = principal / maturity
    else:
        instalment = principal * eir / (1 - (1 + eir) ** (-maturity))

    schedule = []
    balance = principal
    for t in range(1, maturity + 1):
        interest = balance * eir
        princ_repay = instalment - interest
        ead_bop = balance
        balance -= princ_repay
        schedule.append({
            "year":          t,
            "ead_bop":       round(ead_bop, 2),
            "interest":      round(interest, 2),
            "principal":     round(princ_repay, 2),
            "instalment":    round(instalment, 2),
            "ead_eop":       round(max(balance, 0), 2),
        })
    return schedule


# ---------------------------------------------------------------------------
# Lifetime ECL calculation
# ---------------------------------------------------------------------------

@dataclass
class LoanPhase2:
    """Loan with term structure for lifetime ECL."""
    name: str
    principal: float      # original notional (EUR)
    rating: str           # S&P category (must be in cumulative PD table)
    lgd: float            # loss given default (decimal)
    eir: float            # effective interest rate (decimal)
    maturity: int         # remaining maturity in years
    amort_type: str = "bullet"   # "bullet" or "amortising"
    region: str = "europe"       # "global" or "europe"


def lifetime_ecl(loan: LoanPhase2) -> dict:
    """
    Compute lifetime ECL for a Stage 2 or Stage 3 exposure.

    ECL = Σ [ fPD(t) × LGD × EAD(t) × DF(t) ]

    where:
      fPD(t)  = unconditional forward PD for year t
      EAD(t)  = outstanding balance at the start of year t
      DF(t)   = 1 / (1 + EIR)^t

    This is mathematically equivalent to:
      Σ [ mPD(t) × S(t-1) × LGD × EAD(t) × DF(t) ]

    Both forms are computed for transparency.

    Returns a dict with the period-by-period breakdown and totals.
    """
    # Build PD term structure
    pd_curve = build_pd_term_structure(
        loan.rating, loan.maturity, loan.region
    )

    # Build amortisation schedule
    if loan.amort_type == "amortising":
        amort = amortising_schedule(loan.principal, loan.eir, loan.maturity)
    else:
        amort = bullet_schedule(loan.principal, loan.eir, loan.maturity)

    # Compute ECL for each period
    periods = []
    total_ecl = 0.0
    total_el_nominal = 0.0

    for t in range(1, loan.maturity + 1):
        if t > len(pd_curve):
            # Beyond available PD data: extrapolate flat marginal PD
            last = pd_curve[-1]
            pd_info = {
                "year": t,
                "cum_pd": None,
                "survival": last["survival"],
                "forward_pd": last["forward_pd"],
                "marginal_pd": last["marginal_pd"],
            }
        else:
            pd_info = pd_curve[t - 1]

        ead_t = amort[t - 1]["ead_bop"]
        forward_pd = pd_info["forward_pd"]
        marginal_pd = pd_info["marginal_pd"]
        survival_prev = pd_curve[t - 2]["survival"] if t > 1 else 1.0

        discount_factor = 1.0 / (1.0 + loan.eir) ** t

        # ECL contribution for this period (using forward PD approach)
        el_nominal = forward_pd * loan.lgd * ead_t
        ecl_period = el_nominal * discount_factor

        # Verification: mPD × S(t-1) should equal forward_pd
        check_fpd = marginal_pd * survival_prev

        total_ecl += ecl_period
        total_el_nominal += el_nominal

        periods.append({
            "year":            t,
            "ead":             round(ead_t, 2),
            "cum_pd":          pd_info["cum_pd"],
            "survival":        round(pd_info["survival"], 6) if pd_info["survival"] else None,
            "forward_pd":      round(forward_pd, 6),
            "marginal_pd":     round(marginal_pd, 6),
            "lgd":             loan.lgd,
            "el_nominal":      round(el_nominal, 2),
            "discount_factor": round(discount_factor, 6),
            "ecl_period":      round(ecl_period, 2),
            "ecl_cumulative":  round(total_ecl, 2),
        })

    # Also compute 12-month ECL for comparison (Stage 1 vs Stage 2)
    ecl_12m = periods[0]["ecl_period"] if periods else 0.0

    return {
        "loan_name":       loan.name,
        "rating":          loan.rating,
        "region":          loan.region,
        "principal":       loan.principal,
        "maturity":        loan.maturity,
        "amort_type":      loan.amort_type,
        "lgd":             loan.lgd,
        "eir":             loan.eir,
        "ecl_12m":         round(ecl_12m, 2),
        "ecl_lifetime":    round(total_ecl, 2),
        "el_nominal_total": round(total_el_nominal, 2),
        "ratio_lifetime_to_12m": round(total_ecl / ecl_12m, 2) if ecl_12m > 0 else None,
        "periods":         periods,
    }


def print_lifetime_ecl(result: dict) -> None:
    """Pretty-print a lifetime ECL result."""
    print(f"\n{'='*90}")
    print(f"  IFRS 9 — Lifetime ECL (Stage 2)")
    print(f"{'='*90}")
    print(f"  Loan:         {result['loan_name']}")
    print(f"  Rating:       {result['rating']} ({result['region']})")
    print(f"  Principal:    €{result['principal']:,.2f}")
    print(f"  Maturity:     {result['maturity']} years ({result['amort_type']})")
    print(f"  LGD:          {result['lgd']:.2%}")
    print(f"  EIR:          {result['eir']:.2%}")
    print(f"{'─'*90}")

    # Period-by-period table
    print(f"  {'Year':>4}  {'EAD':>14}  {'Fwd PD':>8}  {'Marg PD':>8}  "
          f"{'Nominal EL':>12}  {'DF':>8}  {'Period ECL':>12}  {'Cumul ECL':>12}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*8}  {'─'*8}  "
          f"{'─'*12}  {'─'*8}  {'─'*12}  {'─'*12}")

    for p in result["periods"]:
        print(f"  {p['year']:>4}  €{p['ead']:>12,.2f}  "
              f"{p['forward_pd']:>7.4%}  {p['marginal_pd']:>7.4%}  "
              f"€{p['el_nominal']:>10,.2f}  {p['discount_factor']:>7.4f}  "
              f"€{p['ecl_period']:>10,.2f}  €{p['ecl_cumulative']:>10,.2f}")

    print(f"{'─'*90}")
    print(f"  12-Month ECL (Stage 1):   €{result['ecl_12m']:>12,.2f}")
    print(f"  Lifetime ECL (Stage 2):   €{result['ecl_lifetime']:>12,.2f}")
    if result['ratio_lifetime_to_12m']:
        print(f"  Ratio (Lifetime / 12m):   {result['ratio_lifetime_to_12m']:>12.1f}x")
    print(f"{'='*90}\n")


# ---------------------------------------------------------------------------
# Example runs
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ── Example 1: BBB bullet loan, 5 years, European PDs ──
    loan_bbb_bullet = LoanPhase2(
        name="EuroIndustrial AG — Senior Unsecured Term Loan",
        principal=1_000_000,
        rating="BBB",
        lgd=0.45,
        eir=0.035,
        maturity=5,
        amort_type="bullet",
        region="europe",
    )
    result1 = lifetime_ecl(loan_bbb_bullet)
    print_lifetime_ecl(result1)

    # ── Example 2: Same loan but amortising ──
    loan_bbb_amort = LoanPhase2(
        name="EuroIndustrial AG — Amortising Term Loan",
        principal=1_000_000,
        rating="BBB",
        lgd=0.45,
        eir=0.035,
        maturity=5,
        amort_type="amortising",
        region="europe",
    )
    result2 = lifetime_ecl(loan_bbb_amort)
    print_lifetime_ecl(result2)

    # ── Example 3: BB loan, 7 years, to show the impact of lower ratings ──
    loan_bb = LoanPhase2(
        name="MidCap Services SAS — Leveraged Term Loan B",
        principal=5_000_000,
        rating="BB",
        lgd=0.45,
        eir=0.055,
        maturity=7,
        amort_type="bullet",
        region="europe",
    )
    result3 = lifetime_ecl(loan_bb)
    print_lifetime_ecl(result3)

    # ── Summary comparison ──
    print("\n" + "="*60)
    print("  Stage 1 vs Stage 2 comparison")
    print("="*60)
    for r in [result1, result2, result3]:
        print(f"\n  {r['loan_name']}")
        print(f"    Stage 1 (12m ECL):   €{r['ecl_12m']:>10,.2f}")
        print(f"    Stage 2 (Lifetime):  €{r['ecl_lifetime']:>10,.2f}")
        if r['ratio_lifetime_to_12m']:
            print(f"    Cliff effect:        {r['ratio_lifetime_to_12m']:.1f}x")
