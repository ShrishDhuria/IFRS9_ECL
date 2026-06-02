"""macro_pd — single-factor Vasicek PIT-PD model invariants."""
import math

import pytest

from macro_pd import (
    norm_cdf, norm_ppf, basel_corporate_correlation,
    vasicek_conditional_pd, implied_systematic_factor,
    systematic_factor_from_macro, implied_notch_shift, stressed_1y_pd,
)


# ---------------------------------------------------------------------------
# Normal CDF / inverse CDF
# ---------------------------------------------------------------------------
def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)
    assert norm_cdf(1.959963985) == pytest.approx(0.975, abs=1e-6)
    assert norm_cdf(-1.959963985) == pytest.approx(0.025, abs=1e-6)


def test_norm_ppf_known_values():
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.975) == pytest.approx(1.959963985, abs=1e-6)
    assert norm_ppf(0.025) == pytest.approx(-1.959963985, abs=1e-6)


def test_cdf_ppf_round_trip():
    for p in (0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, abs=1e-8)


# ---------------------------------------------------------------------------
# Basel asset correlation
# ---------------------------------------------------------------------------
def test_basel_correlation_bounds():
    for pd in (0.0001, 0.001, 0.01, 0.05, 0.1, 0.3):
        rho = basel_corporate_correlation(pd)
        assert 0.12 - 1e-9 <= rho <= 0.24 + 1e-9


def test_basel_correlation_decreasing_in_pd():
    pds = [0.0005, 0.005, 0.02, 0.1, 0.3]
    rhos = [basel_corporate_correlation(pd) for pd in pds]
    for hi, lo in zip(rhos, rhos[1:]):
        assert lo <= hi  # correlation falls as PD rises


# ---------------------------------------------------------------------------
# Vasicek conditional PD
# ---------------------------------------------------------------------------
def test_z_zero_returns_ttc_pd():
    """The anchor property: at z = 0 the conditional PD equals the TTC PD."""
    for pd in (0.0002, 0.0013, 0.0044, 0.0269, 0.1):
        assert vasicek_conditional_pd(pd, 0.0) == pytest.approx(pd, abs=1e-9)


def test_conditional_pd_decreasing_in_z():
    """Stronger economy (higher z) -> lower PD; recession (z<0) -> higher PD."""
    pd = 0.0044
    zs = [-3, -2, -1, 0, 1, 2, 3]
    pds = [vasicek_conditional_pd(pd, z) for z in zs]
    for hi, lo in zip(pds, pds[1:]):
        assert lo <= hi


def test_adverse_z_raises_pd_above_ttc():
    pd = 0.0013
    assert vasicek_conditional_pd(pd, -2.0) > pd


def test_conditional_pd_in_unit_interval():
    for z in (-5, -2, 0, 2, 5):
        for pd in (0.0001, 0.01, 0.3):
            v = vasicek_conditional_pd(pd, z)
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Inverse Vasicek (back out z)
# ---------------------------------------------------------------------------
def test_implied_factor_inverts_conditional_pd():
    """implied_systematic_factor must invert vasicek_conditional_pd."""
    pd = 0.038
    for z_true in (-2.0, -1.0, 0.0, 1.0):
        dr = vasicek_conditional_pd(pd, z_true)
        z_back = implied_systematic_factor(dr, pd)
        assert z_back == pytest.approx(z_true, abs=1e-6)


def test_implied_factor_bad_year_is_negative():
    """A realised default rate above the TTC PD implies a negative factor."""
    z = implied_systematic_factor(observed_dr=0.099, pd_ttc=0.038)
    assert z < 0


# ---------------------------------------------------------------------------
# Macro bridge + notch shift
# ---------------------------------------------------------------------------
def test_eba_adverse_maps_to_negative_factor():
    z = systematic_factor_from_macro(gdp_dev_pct=-10.4, unemp_dev_pp=6.1)
    assert z < -1.5  # severe; lands near -2.0 by calibration


def test_baseline_macro_maps_to_zero_factor():
    assert systematic_factor_from_macro(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_notch_shift_zero_at_z_zero():
    """At z = 0 the PIT PD equals the TTC PD, so the implied shift is zero."""
    for rating in ("A", "BBB", "BB", "B"):
        assert implied_notch_shift(rating, 0.0, region="global") == 0


def test_notch_shift_positive_under_adverse():
    z = systematic_factor_from_macro(-10.4, 6.1)
    for rating in ("A", "BBB", "BB", "B"):
        assert implied_notch_shift(rating, z, region="global") > 0


def test_notch_shift_negative_under_upside():
    for rating in ("BBB", "BB", "B"):
        assert implied_notch_shift(rating, 1.5, region="global") <= 0


def test_stressed_pd_exceeds_ttc_under_adverse():
    from ecl_engine_phase1 import SP_1Y_PD
    z = systematic_factor_from_macro(-10.4, 6.1)
    for rating in ("A", "BBB", "BB", "B"):
        assert stressed_1y_pd(rating, z, "global") > SP_1Y_PD[rating]
