"""Tests of cytovanni.utils.evl: paired / single CVs (fractions) and the symmetric correlation."""
import warnings

import numpy as np
import pytest

from cytovanni.exceptions import CytovanniWarning
from cytovanni.utils import (get_asinh_paired_CV, get_asinh_paired_CV_alongdiag, get_asinh_single_CV,
                             get_positive_paired_CV, get_positive_paired_CV_alongdiag, get_positive_single_CV,
                             symmetric_correlation)
from cytovanni.utils.evl import get_asinh_paired_CV_varcorr, get_asinh_single_CV_varcorr


def lognormal_sd(cv):
    return np.sqrt(np.log(1 + cv**2))


@pytest.fixture
def positive_pairs(rng):
    """Docstring example: true log shares in [-5, -1], measurement CV 0.6 per value."""
    ls = np.linspace(-5, -1, 20000)
    sd = lognormal_sd(.6)
    return np.exp(ls), np.exp(ls + rng.normal(scale=sd, size=ls.size)), np.exp(ls + rng.normal(scale=sd, size=ls.size))


def single_warning(fn, *args):
    """Call fn, return (value, list of CytovanniWarnings)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = fn(*args)
    return v, [x for x in w if issubclass(x.category, CytovanniWarning)]


# ------------------------------------------------------------------ symmetric_correlation
def test_symmetric_correlation_penalises_shift_and_scale():
    x = np.arange(10.)
    assert np.isclose(symmetric_correlation(x, x), 1.)
    assert symmetric_correlation(x, x + 3) < 1 and symmetric_correlation(x, 2 * x) < 1
    assert np.isclose(np.corrcoef(x, 2 * x)[0, 1], 1.)  # plain Pearson does not


def test_symmetric_correlation_drops_nonfinite():
    x, y = np.array([1., 2, 3, np.nan, 4]), np.array([1., 3, 2, 7, np.inf])
    assert np.isclose(symmetric_correlation(x, y), symmetric_correlation([1, 2, 3], [1, 3, 2]))


# ------------------------------------------------------------------ positive CVs (recovery)
def test_positive_paired_CV_recovers_measurement_CV(positive_pairs):
    _, x, y = positive_pairs
    assert get_positive_paired_CV(x, y) == pytest.approx(.6, rel=.03)
    assert get_positive_paired_CV(x, y) == pytest.approx(get_positive_paired_CV(y, x))  # symmetric


def test_positive_paired_CV_alongdiag_recovers_spread_of_values(positive_pairs):
    true, x, y = positive_pairs
    expected = get_positive_single_CV(true)
    assert expected == pytest.approx(1.67, rel=.02)
    assert get_positive_paired_CV_alongdiag(x, y) == pytest.approx(expected, rel=.03)


def test_positive_single_CV_lognormal(rng):
    x = np.exp(rng.normal(scale=lognormal_sd(.6), size=20000))
    assert get_positive_single_CV(x) == pytest.approx(.6, rel=.03)


def test_positive_paired_CV_formula():
    x, y = np.array([1., 2, 4, 8]), np.array([1.5, 1.8, 5, 7])
    lx, ly = np.log(np.r_[x, y]), np.log(np.r_[y, x])
    u = np.array([-1, 1]) / np.sqrt(2)
    assert get_positive_paired_CV(x, y) == pytest.approx(np.sqrt(np.exp(u @ np.cov(np.vstack([lx, ly])) @ u) - 1))


def test_positive_CVs_drop_values_le_zero(positive_pairs):
    """R2c-H1: a value <= 0 drops its pair (was: the whole CV became NaN)."""
    _, x, y = positive_pairs
    x0 = x.copy(); x0[3] = 0.; x0[7] = -1.
    keep = np.ones(x.size, bool); keep[[3, 7]] = False
    assert get_positive_paired_CV(x0, y) == pytest.approx(get_positive_paired_CV(x[keep], y[keep]))
    assert get_positive_paired_CV_alongdiag(x0, y) == pytest.approx(get_positive_paired_CV_alongdiag(x[keep], y[keep]))
    assert get_positive_single_CV(x0) == pytest.approx(get_positive_single_CV(x[keep]))
    assert get_positive_single_CV(np.r_[x, np.nan, np.inf]) == pytest.approx(get_positive_single_CV(x))


def test_positive_single_CV_uses_sample_variance(rng):
    """R2c-H2: ddof=1 (was the population variance)."""
    x = np.exp(rng.normal(size=10))
    assert get_positive_single_CV(x) == pytest.approx(np.sqrt(np.exp(np.log(x).var(ddof=1)) - 1))


# ------------------------------------------------------------------ asinh CVs
def test_asinh_paired_CV_docstring_example(rng):
    ls = np.linspace(1, 3, 20000); sd = lognormal_sd(.5)
    v = get_asinh_paired_CV(ls + rng.normal(scale=sd, size=ls.size), ls + rng.normal(scale=sd, size=ls.size), 10)
    assert .4 < v < .6


@pytest.mark.parametrize("z", [3e4, 1e5, 1e6])
def test_asinh_paired_CV_is_relative_CV_above_cofactor(rng, z):
    f, sd = 1500., .1
    zx = z * np.exp(rng.normal(scale=sd, size=20000)); zy = z * np.exp(rng.normal(scale=sd, size=20000))
    v = get_asinh_paired_CV(np.arcsinh(zx / f), np.arcsinh(zy / f), f)
    assert v == pytest.approx(np.sqrt(np.exp(sd**2) - 1), rel=.05)


@pytest.mark.parametrize("z", [-300., 0., 200.])
def test_asinh_paired_CV_is_sd_over_f_below_cofactor(rng, z):
    f, sd = 1500., 30.
    zx = z + rng.normal(scale=sd, size=20000); zy = z + rng.normal(scale=sd, size=20000)
    v = get_asinh_paired_CV(np.arcsinh(zx / f), np.arcsinh(zy / f), f)
    assert v == pytest.approx(sd / f, rel=.06)


def test_asinh_paired_CV_alongdiag_docstring_example(rng):
    ls = np.linspace(1, 2, 20000); sd = lognormal_sd(.2)
    v = get_asinh_paired_CV_alongdiag(ls + rng.normal(scale=sd, size=ls.size), ls + rng.normal(scale=sd, size=ls.size), 10)
    assert v == pytest.approx(get_asinh_single_CV(ls, 10), rel=.05)


def test_asinh_single_CV_docstring_example(rng):
    v = get_asinh_single_CV(2 + rng.normal(scale=lognormal_sd(.3), size=20000), 10)
    assert v == pytest.approx(.3, rel=.1)


def test_asinh_single_CV_formula():
    x, f = np.array([1., 1.5, 2.5, 3.]), 7.
    z0 = f * np.sinh(x.mean())
    assert get_asinh_single_CV(x, f) == pytest.approx(np.sqrt(z0**2 + f**2) * x.std(ddof=1) / max(abs(z0), f))
    assert get_asinh_single_CV(np.r_[x, np.nan], f) == pytest.approx(get_asinh_single_CV(x, f))


def test_varcorr_with_zero_errors_equals_uncorrected():
    """R2d-D5: errors of 0 are used (subtract nothing); was NaN / dropped."""
    x, y = np.array([.1, .4, 1.1, 2.]), np.array([.2, .1, 1.2, 1.3])
    z = np.zeros(4)
    assert get_asinh_single_CV_varcorr(x, z, 1500) == pytest.approx(get_asinh_single_CV(x, 1500))
    assert get_asinh_paired_CV_varcorr(x, y, z, z, 1500) == pytest.approx(get_asinh_paired_CV(x, y, 1500))


def test_varcorr_drops_negative_and_nonfinite_errors():
    x, y = np.array([.1, .4, 1.1, 2., 5.]), np.array([.2, .1, 1.2, 1.3, 9.])
    e = np.array([0., 0., 0., 0., -1.])
    assert get_asinh_single_CV_varcorr(x, e, 1500) == pytest.approx(get_asinh_single_CV(x[:4], 1500))
    e2 = np.array([0., 0., 0., 0., np.nan])
    assert get_asinh_paired_CV_varcorr(x, y, e2, np.zeros(5), 1500) == pytest.approx(get_asinh_paired_CV(x[:4], y[:4], 1500))


def test_varcorr_subtracts_error_variance(rng):
    m = np.linspace(4, 9, 4000)
    x, y = m + rng.normal(scale=.1, size=m.size), m + rng.normal(scale=.1, size=m.size)
    err = np.full(m.size, .05)
    plain = get_asinh_paired_CV(x, y, 1500.)
    corr = get_asinh_paired_CV_varcorr(x, y, err, err, 1500.)
    s1, s2 = np.r_[x, y], np.r_[y, x]; u = np.array([-1, 1]) / np.sqrt(2)
    var = u @ np.cov(np.vstack([s1, s2])) @ u
    assert corr == pytest.approx(plain * np.sqrt((var - .05**2) / var))
    single = get_asinh_single_CV_varcorr(x, err, 1500.)
    assert single == pytest.approx(get_asinh_single_CV(x, 1500.) * np.sqrt((x.var(ddof=1) - .05**2) / x.var(ddof=1)))


def test_single_varcorr_clamps_negative_variance_to_zero():
    """R2-E16: error variance above the observed one gives 0 with a CytovanniWarning (was NaN)."""
    v, w = single_warning(get_asinh_single_CV_varcorr, np.array([1.0, 1.01, 0.99, 1.0]), np.full(4, .5), 1.)
    assert v == 0 and len(w) == 1


def test_paired_varcorr_clamps_negative_variance_to_zero_with_warning():
    """R2c-H3: the paired clamp warns too."""
    x = np.array([1., 1.01, 2., 2.02]); y = x + .001
    v, w = single_warning(get_asinh_paired_CV_varcorr, x, y, np.ones(4), np.ones(4), 1500.)
    assert v == 0 and len(w) == 1


# ------------------------------------------------------------------ NaN + warning cases (R2c-H3)
@pytest.mark.parametrize("fn, args", [
    (get_positive_paired_CV, ([1.], [1.])),
    (get_positive_paired_CV_alongdiag, ([0., 1.], [1., 1.])),
    (get_positive_single_CV, ([0., 2.],)),
    (get_asinh_paired_CV, ([1.], [np.nan], 1500.)),
    (get_asinh_single_CV, ([1.], 1500.)),
    (get_asinh_paired_CV_alongdiag, ([], [], 1500.)),
    (get_asinh_paired_CV_varcorr, ([1.], [1.], [.1], [.1], 1500.)),
    (get_asinh_single_CV_varcorr, ([1.], [.1], 1500.)),
])
def test_too_few_values_give_nan_with_warning(fn, args):
    """R2c-H3: fewer than 2 usable values -> NaN with a CytovanniWarning."""
    v, w = single_warning(fn, *args)
    assert np.isnan(v) and len(w) == 1


def test_alongdiag_negative_variance_gives_nan_with_warning(rng):
    """R2c-H3: anti-correlated pairs (variance along < orthogonal) -> NaN with a warning."""
    x = 2. + rng.normal(scale=.3, size=200); y = 4. - x
    v, w = single_warning(get_asinh_paired_CV_alongdiag, x, y, 1500.)
    assert np.isnan(v) and len(w) == 1
    v, w = single_warning(get_positive_paired_CV_alongdiag, np.exp(x), np.exp(y))
    assert np.isnan(v) and len(w) == 1


def test_clean_inputs_do_not_warn(positive_pairs, no_warnings):
    _, x, y = positive_pairs
    get_positive_paired_CV(x, y); get_positive_paired_CV_alongdiag(x, y); get_positive_single_CV(x)
    a, b = np.arcsinh(x * 1e6 / 1500), np.arcsinh(y * 1e6 / 1500)
    get_asinh_paired_CV(a, b, 1500); get_asinh_paired_CV_alongdiag(a, b, 1500); get_asinh_single_CV(a, 1500)

