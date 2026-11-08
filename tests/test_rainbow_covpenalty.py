"""Covariance penalty of the rainbow fluorescence fit (`fit_fluorescence(cov_penalty=..., cov_penalty_broaden_only=...)`).

Without the penalty, the free per-measurement peak precisions can broaden a peak over its neighbours, after which the
measurement's factors drift at little cost in likelihood (found for calibration beads with one bright peak per laser).
The tests check the penalty itself (zero at the reference, the KL formula, broadening-only), that the
option reaches the trainer, that 0 leaves the fit unchanged, and that the penalty keeps the covariances near the
reference during a fit.
"""
import numpy as np
import pytest
import torch

import cytovanni.rainbow.rbim as rbim
from cytovanni.rainbow import (RainbowCalibrator, RainbowFluorescenceGMMCalibrationDataset,
                               RainbowFluorescenceGMMCalibrationTrainer)

from synthetic import ShortTrainer, short_fit_fluorescence

FL = ["FL1-A", "FL2-A", "FL3-A"]


def initialised(fitted_rainbow_ads, **kw):
    """Calibrator, dataset and trainer after `fit_init` (short schedule, CPU), reference covariances set."""
    rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads])
    ds = RainbowFluorescenceGMMCalibrationDataset(rb.rboe, rb.adatas, device="cpu")
    tr = ShortTrainer(rb.rfim, **kw)
    tr.fit_init(ds)
    tr.set_cov_reference(ds)
    return rb, ds, tr


def scale_precision(rb, c):
    """Multiply every precision P = A A^T by `c` (A by sqrt(c))."""
    with torch.no_grad():
        for p in rb.rfim.param_cov.values():
            p.param_trainable.data *= np.sqrt(c)
            p.param_fixed.data *= np.sqrt(c)


def test_penalty_zero_at_reference(fitted_rainbow_ads):
    rb, ds, tr = initialised(fitted_rainbow_ads)
    for broaden_only in [True, False]:
        tr.cov_penalty_broaden_only = broaden_only
        assert abs(tr.get_cov_penalty().item()) < 1e-4


@pytest.mark.parametrize("c", [.5, .8])
def test_penalty_isotropic_broadening_equals_kl(fitted_rainbow_ads, c):
    """P = c P0 (all peaks broader by 1/c): both variants give the KL d/2 (c - 1 - ln c) per peak, summed over the
    peaks and averaged over the measurements."""
    rb, ds, tr = initialised(fitted_rainbow_ads)
    scale_precision(rb, c)
    npeak = sum(cov.shape[1] for _, cov, _, _ in tr.cov_reference.values())
    d = len(FL)
    expected = npeak * d / 2 * (c - 1 - np.log(c))
    for broaden_only in [True, False]:
        tr.cov_penalty_broaden_only = broaden_only
        assert tr.get_cov_penalty().item() == pytest.approx(expected, rel=1e-3)


def test_penalty_broaden_only_ignores_narrowing(fitted_rainbow_ads):
    rb, ds, tr = initialised(fitted_rainbow_ads)
    scale_precision(rb, 2.)                       # all peaks narrower
    tr.cov_penalty_broaden_only = True
    assert abs(tr.get_cov_penalty().item()) < 1e-4
    tr.cov_penalty_broaden_only = False
    assert tr.get_cov_penalty().item() > .1


def test_penalty_gradient_narrows_broadened_peaks(fitted_rainbow_ads):
    """A gradient step on the penalty alone moves broadened precisions back up (towards the reference)."""
    rb, ds, tr = initialised(fitted_rainbow_ads)
    scale_precision(rb, .5)
    p = next(iter(rb.rfim.param_cov.values()))
    before = p()[0].detach().diagonal(dim1=-2, dim2=-1).sum()
    tr.get_cov_penalty().backward()
    with torch.no_grad():
        p.param_trainable -= .01 * p.param_trainable.grad
    after = p()[0].detach().diagonal(dim1=-2, dim2=-1).sum()
    assert after > before


def test_fit_fluorescence_passes_penalty_to_trainer(fitted_rainbow_ads):
    rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads])
    short_fit_fluorescence(rb, cov_penalty=.3, cov_penalty_broaden_only=False)
    assert rb.rfim_trainer.cov_penalty == .3 and rb.rfim_trainer.cov_penalty_broaden_only is False
    assert rb.rfim_trainer.cov_reference is not None


def test_default_trainer_has_no_penalty(fitted_rainbow_ads):
    rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads])
    tr = RainbowFluorescenceGMMCalibrationTrainer(rb.rfim)
    assert tr.cov_penalty == 0 and tr.cov_reference is None


def test_penalty_zero_gives_the_unpenalised_fit(fitted_rainbow_ads):
    """cov_penalty=0 is the fit without the option (same seeds -> identical factors and loss history)."""
    out = []
    for kw in [{}, {"cov_penalty": 0.}]:
        torch.manual_seed(0); np.random.seed(0)
        rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads])
        short_fit_fluorescence(rb, **kw)
        out.append((rb.get_calibration_factors_all()[FL].to_numpy(), list(rb.rfim_trainer.fit_losshistory)))
        assert rb.rfim_trainer.cov_reference is None
    np.testing.assert_array_equal(out[0][0], out[1][0])
    assert out[0][1] == out[1][1]


class MediumTrainer(RainbowFluorescenceGMMCalibrationTrainer):
    def fit_init_peak(self, dataset, Niter=200, **kw):
        return super().fit_init_peak(dataset, Niter=Niter, **kw)

    def fit_init_cov(self, dataset, Niter=5, **kw):
        return super().fit_init_cov(dataset, Niter=Niter, **kw)

    def fit(self, dataset, Niter=300, **kw):
        return super().fit(dataset, Niter=Niter, **kw)


@pytest.mark.slow
def test_penalty_limits_broadening_during_fit(fitted_rainbow_ads):
    """During the full fit, the penalised run's covariances stay closer to the reference than the unpenalised run's
    (broadening-only KL to the covariances after fit_init), and the factors are still recovered."""
    kl, factors = {}, {}
    for lam in [0., 10.]:
        torch.manual_seed(0); np.random.seed(0)
        rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads])
        ds = RainbowFluorescenceGMMCalibrationDataset(rb.rboe, rb.adatas, device="cpu")
        tr = MediumTrainer(rb.rfim, cov_penalty=lam)
        tr.fit_init(ds, showprogress=False)
        tr.set_cov_reference(ds)
        tr.fit(ds, showprogress=False)
        tr.cov_penalty_broaden_only = True
        kl[lam] = tr.get_cov_penalty().item()
        factors[lam] = rb.get_calibration_factors_all()[FL]
    assert kl[10.] < kl[0.]
    np.testing.assert_allclose(factors[10.].to_numpy(), factors[0.].to_numpy(), rtol=.05)


@pytest.mark.parametrize("warmup", [0, 3])
def test_refit_uses_the_given_lr(fitted_rainbow_ads, warmup):
    """A later `fit` call uses its own `lr` after the warm-up, also with warmup=0 (before: the old lr was kept)."""
    rb, ds, tr = initialised(fitted_rainbow_ads)
    tr.fit(ds, Niter=4, lr=1e-2, warmup=2, showprogress=False)
    assert tr.optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
    tr.fit(ds, Niter=4, lr=1e-4, warmup=warmup, showprogress=False)
    assert tr.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert len(tr.fit_losshistory) == 8
