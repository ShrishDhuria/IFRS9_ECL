"""
IFRS 9 ECL Engine — Phase 1
============================
Single-loan, 12-month Expected Credit Loss for a Stage 1 exposure.

Data source: S&P Global Ratings, "2024 Annual Global Corporate Default
And Rating Transition Study", March 27, 2025 — Table 26 (global average
cumulative default rates by rating level, 1981-2024).

Run directly:  python ecl_engine_phase1.py
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# S&P Table 26: weighted-average 1-year PDs by rating level (1981-2024)
# These are through-the-cycle averages.  For IFRS 9 we'd ideally use
# point-in-time PDs, but the TTC averages from S&P are the standard
# academic/interview anchor.  We'll adjust to PIT in Phase 4 when we
# overlay macro scenarios.
# ---------------------------------------------------------------------------
SP_1Y_PD = {
    "AAA":   0.0000,
    "AA+":   0.0000,
    "AA":    0.0002,
    "AA-":   0.0002,
    "A+":    0.0004,
    "A":     0.0005,
    "A-":    0.0005,
    "BBB+":  0.0009,
    "BBB":   0.0013,
    "BBB-":  0.0021,
    "BB+":   0.0027,
    "BB":    0.0044,
    "BB-":   0.0087,
    "B+":    0.0183,
    "B":     0.0269,
    "B-":    0.0516,
    "CCC/C": 0.2612,
}

# European-specific 1Y PDs from Table 25 (category level, 1981-2024)
# Use these when you want a Europe-calibrated model.
SP_1Y_PD_EUROPE = {
    "AAA":   0.0000,
    "AA":    0.0000,
    "A":     0.0003,
    "BBB":   0.0005,
    "BB":    0.0036,
    "B":     0.0175,
    "CCC/C": 0.2626,
}


@dataclass
class Loan:
    """A single loan exposure for Phase 1 (Stage 1, 12-month horizon)."""
    name: str        # borrower or facility identifier
    ead: float       # exposure at default (EUR), current outstanding
    rating: str      # S&P-equivalent rating (key into SP_1Y_PD)
    lgd: float       # loss given default, decimal (0.45 = 45%)
    eir: float       # effective interest rate at origination, decimal

    @property
    def pd_12m(self) -> float:
        """Look up the 12-month PD from the S&P table."""
        if self.rating not in SP_1Y_PD:
            raise ValueError(
                f"Rating '{self.rating}' not found. "
                f"Valid ratings: {list(SP_1Y_PD.keys())}"
            )
        return SP_1Y_PD[self.rating]


def twelve_month_ecl(loan: Loan) -> dict:
    """
    Compute the 12-month Expected Credit Loss for a Stage 1 loan.

    Formula
    -------
        ECL = (PD × LGD × EAD) / (1 + EIR)

    The numerator is the nominal expected loss (Basel-style EL). The
    denominator discounts it back one period at the loan's effective
    interest rate, because IFRS 9 requires ECL to be a present value.

    Why EIR and not the risk-free rate?
    Because IFRS 9 impairment adjusts the loan's amortised cost, which
    is itself built using the EIR. Using the same discount rate keeps
    the impairment consistent with the carrying amount.

    Returns a dict with the breakdown so you can inspect each component.
    """
    pd = loan.pd_12m
    el_nominal = pd * loan.lgd * loan.ead
    discount_factor = 1 / (1 + loan.eir)
    ecl = el_nominal * discount_factor

    return {
        "loan_name":       loan.name,
        "rating":          loan.rating,
        "ead":             loan.ead,
        "pd_12m":          pd,
        "lgd":             loan.lgd,
        "eir":             loan.eir,
        "el_nominal":      round(el_nominal, 2),
        "discount_factor": round(discount_factor, 6),
        "ecl_12m":         round(ecl, 2),
    }


def print_result(result: dict) -> None:
    """Pretty-print an ECL result dict."""
    print(f"\n{'='*50}")
    print(f"  IFRS 9 — 12-Month ECL (Stage 1)")
    print(f"{'='*50}")
    print(f"  Loan:             {result['loan_name']}")
    print(f"  Rating:           {result['rating']}")
    print(f"  EAD:              €{result['ead']:,.2f}")
    print(f"  PD (12m):         {result['pd_12m']:.4%}")
    print(f"  LGD:              {result['lgd']:.2%}")
    print(f"  EIR:              {result['eir']:.2%}")
    print(f"  ─────────────────────────────────")
    print(f"  Nominal EL:       €{result['el_nominal']:,.2f}")
    print(f"  Discount factor:  {result['discount_factor']:.6f}")
    print(f"  12-month ECL:     €{result['ecl_12m']:,.2f}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Example: a €1m senior unsecured corporate loan, BBB rated, 3.5% EIR
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Base case
    loan_bbb = Loan(
        name="Acme Corp Senior Unsecured",
        ead=1_000_000,
        rating="BBB",
        lgd=0.45,   # Foundation IRB: 45% senior unsecured
        eir=0.035,
    )
    result = twelve_month_ecl(loan_bbb)
    print_result(result)

    # Compare across rating grades to build intuition
    print("\nSensitivity across ratings (€1m, LGD=45%, EIR=3.5%):\n")
    print(f"  {'Rating':<8} {'PD':>8} {'Nominal EL':>12} {'12m ECL':>12}")
    print(f"  {'─'*8} {'─'*8} {'─'*12} {'─'*12}")

    for rating in ["AAA", "AA", "A", "BBB", "BBB-", "BB", "BB-",
                    "B", "B-", "CCC/C"]:
        loan = Loan("Test", 1_000_000, rating, 0.45, 0.035)
        r = twelve_month_ecl(loan)
        print(f"  {rating:<8} {r['pd_12m']:>7.2%} "
              f"  €{r['el_nominal']:>10,.2f} "
              f"  €{r['ecl_12m']:>10,.2f}")
