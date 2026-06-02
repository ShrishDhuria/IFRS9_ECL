"""Phase 1 & 2 — single-loan ECL and lifetime PD term-structure invariants."""
import pytest

from ecl_engine_phase1 import Loan, twelve_month_ecl, SP_1Y_PD
from ecl_engine_phase2 import (
    build_pd_term_structure, lifetime_ecl, LoanPhase2,
)


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
def test_phase1_ecl_formula_exact():
    """ECL = PD * LGD * EAD / (1 + EIR), to the cent."""
    loan = Loan("t", ead=1_000_000, rating="BBB", lgd=0.45, eir=0.035)
    r = twelve_month_ecl(loan)
    pd = SP_1Y_PD["BBB"]
    expected = pd * 0.45 * 1_000_000 / 1.035
    assert r["ecl_12m"] == pytest.approx(round(expected, 2), abs=0.01)


def test_phase1_discounting_reduces_el():
    """Discounted ECL must be strictly below the nominal expected loss."""
    loan = Loan("t", 1_000_000, "BB", 0.45, 0.05)
    r = twelve_month_ecl(loan)
    assert r["ecl_12m"] < r["el_nominal"]


def test_phase1_monotonic_in_rating_risk():
    """Worse ratings must produce higher ECL (PDs are monotone in the table)."""
    order = ["AAA", "A", "BBB", "BB", "B", "CCC/C"]
    ecls = [twelve_month_ecl(Loan("t", 1e6, r, 0.45, 0.035))["ecl_12m"] for r in order]
    assert ecls == sorted(ecls)


def test_phase1_unknown_rating_raises():
    with pytest.raises(ValueError):
        twelve_month_ecl(Loan("t", 1e6, "ZZZ", 0.45, 0.035))


# ---------------------------------------------------------------------------
# Phase 2 — PD term structure
# ---------------------------------------------------------------------------
def test_survival_monotonically_decreasing():
    curve = build_pd_term_structure("BB", max_years=15, region="global")
    survivals = [row["survival"] for row in curve]
    for earlier, later in zip(survivals, survivals[1:]):
        assert later <= earlier


def test_cumulative_pd_monotonically_increasing():
    curve = build_pd_term_structure("BB", max_years=15, region="global")
    cum = [row["cum_pd"] for row in curve]
    for earlier, later in zip(cum, cum[1:]):
        assert later >= earlier


def test_forward_pd_equals_marginal_times_survival():
    """The identity fPD(t) = mPD(t) * S(t-1) must hold at every node."""
    curve = build_pd_term_structure("B", max_years=15, region="global")
    surv_prev = 1.0
    for row in curve:
        assert row["forward_pd"] == pytest.approx(row["marginal_pd"] * surv_prev, abs=1e-12)
        surv_prev = row["survival"]


def test_marginal_pd_is_conditional_hazard():
    """Conditional marginal PD must be >= unconditional forward PD (since S<=1)."""
    curve = build_pd_term_structure("BB", max_years=10, region="global")
    for row in curve[1:]:
        assert row["marginal_pd"] >= row["forward_pd"] - 1e-12


# ---------------------------------------------------------------------------
# Phase 2 — lifetime ECL
# ---------------------------------------------------------------------------
def test_lifetime_ecl_at_least_12m():
    """Lifetime ECL must be >= the 12-month ECL for a multi-year exposure."""
    loan = LoanPhase2("t", 1_000_000, "BB", 0.45, 0.04, maturity=5,
                      amort_type="bullet", region="global")
    r = lifetime_ecl(loan)
    assert r["ecl_lifetime"] >= r["ecl_12m"]


def test_one_year_lifetime_equals_12m():
    """A 1-year loan's lifetime ECL collapses to its 12-month ECL."""
    loan = LoanPhase2("t", 1_000_000, "BB", 0.45, 0.04, maturity=1,
                      amort_type="bullet", region="global")
    r = lifetime_ecl(loan)
    assert r["ecl_lifetime"] == pytest.approx(r["ecl_12m"], abs=0.01)


def test_lifetime_ecl_positive_and_finite():
    loan = LoanPhase2("t", 5_000_000, "B", 0.45, 0.06, maturity=7,
                      amort_type="amortising", region="europe")
    r = lifetime_ecl(loan)
    assert r["ecl_lifetime"] > 0
    assert r["ecl_lifetime"] < loan.principal  # can't lose more than principal


def test_higher_lgd_increases_ecl():
    base = lifetime_ecl(LoanPhase2("t", 1e6, "BB", 0.30, 0.04, 5, region="global"))
    high = lifetime_ecl(LoanPhase2("t", 1e6, "BB", 0.60, 0.04, 5, region="global"))
    assert high["ecl_lifetime"] > base["ecl_lifetime"]
