"""cytovanni.rainbow.rbim_scatter: per-batch scatter factors from bead medians (closed-form weighted least squares)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cytovanni.exceptions import CalibrationModuleException, MissingChannelException, DefaultScatterCalibrationException
from cytovanni.io import readfcs_rainbow_sample
from cytovanni.rainbow import RainbowScatterCalibrationModule, RainbowCalibrator, RainbowBatchOrdinalEncoder

import synthetic

NPEAK = len(synthetic.PEAKS)


# ---------------------------------------------------------------- module on synthetic medians (as jobs/r2_verify/f_rainbow.py)
def _module(types, batches, nchan):
    r = RainbowScatterCalibrationModule()
    r.rboe = SimpleNamespace(oes={"rainbow_type": list(range(types)), "rainbow_batch": list(range(batches))})
    r.config["channels"] = np.asarray([f"ch{i}" for i in range(nchan)])
    r.init_parameters()
    return r


def _set_data(r, tidx, bidx, y, n):
    r.data_type_idx = torch.tensor(tidx)
    r.data_batch_idx = torch.tensor(bidx)
    r.data_logmedian = torch.tensor(y, dtype=torch.float32)
    r.data_sqrtN = torch.sqrt(torch.tensor(n, dtype=torch.float32))
    r.has_training_data = True


@pytest.mark.slow
def test_closed_form_equals_converged_sgd_R2_A7(rng):
    """R2-A7: the closed-form fit is the limit of the old SGD fit (2 types, 6 batches, unequal N, not every type in
    every batch)."""
    tidx = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    bidx = [0, 1, 2, 3, 2, 3, 4, 5, 4, 0]
    y = 11 + rng.normal(scale=.4, size=(10, 2))
    n = rng.integers(5_000, 50_000, 10).astype(float)
    sgd = _module(2, 6, 2); _set_data(sgd, tidx, bidx, y, n)
    opt = torch.optim.SGD(sgd.parameters(), lr=5e-3, momentum=.5)
    for _ in range(20000):
        opt.zero_grad(); sgd.get_loss().backward(); opt.step()
    closed = _module(2, 6, 2); _set_data(closed, tidx, bidx, y, n)
    closed.fit()
    for name in ["param_logposition", "param_logshift"]:
        torch.testing.assert_close(getattr(closed, name)(), getattr(sgd, name)().detach(), atol=1e-4, rtol=0)
    assert closed.losshistory[1]["mse"] <= closed.losshistory[0]["mse"]
    assert closed.losshistory[1]["mse"] <= sgd.get_loss().item() * (1 + 1e-5)


def test_closed_form_is_a_stationary_point_R2_A7(rng):
    """R2-A7: the closed-form solution minimises the weighted loss: its gradient in the trainable parameters is 0
    (2 types, 6 batches, unequal N, not every type in every batch)."""
    tidx = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    bidx = [0, 1, 2, 3, 2, 3, 4, 5, 4, 0]
    y = 11 + rng.normal(scale=.4, size=(10, 2))
    n = rng.integers(5_000, 50_000, 10).astype(float)
    closed = _module(2, 6, 2); _set_data(closed, tidx, bidx, y, n)
    closed.fit()
    assert closed.is_trained and closed.losshistory[1]["mse"] < closed.losshistory[0]["mse"]
    probe = _module(2, 6, 2); _set_data(probe, tidx, bidx, y, n)
    with torch.no_grad():                    # same values, still trainable (the mean norm keeps a mean-0 shift)
        probe.param_logposition.param_trainable.data[:] = closed.param_logposition()
        probe.param_logshift.param_trainable.data[:] = closed.param_logshift()
    loss = probe.get_loss().double()
    loss.backward()
    assert loss.item() == pytest.approx(closed.losshistory[1]["mse"], rel=1e-5)
    for p in [probe.param_logposition.param_trainable, probe.param_logshift.param_trainable]:
        assert p.grad.abs().max().item() < 1e-3


def test_closed_form_exact_for_many_batches_and_large_files_R2_A7(rng):
    """R2-A7: 1000 batches of 500k events give the exact shifts (SGD diverged to NaN above ~360k events)."""
    r = _module(1, 1000, 2)
    shift = rng.normal(scale=.3, size=(1000, 2)); shift -= shift.mean(0)
    _set_data(r, [0] * 1000, list(range(1000)), 11.5 + shift, [500_000.] * 1000)
    r.fit()
    assert r.is_trained
    np.testing.assert_allclose(r.param_logshift().detach().numpy(), shift, atol=1e-5)
    np.testing.assert_allclose(r.param_logposition().detach().numpy(), 11.5, atol=1e-5)
    assert abs(r.param_logshift().detach().numpy().mean(0)).max() < 1e-6     # mean norm over the batches


def test_extend_keeps_frozen_parameters_R2_A7():
    """R2-A7: after freezing, a second fit only changes the new entries; the mean norm stays over the first batches."""
    r = _module(1, 3, 2)
    _set_data(r, [0, 0, 0], [0, 1, 2], 11 + np.array([[.1, .2], [0, 0], [-.1, -.2]]), [1e4] * 3)
    r.fit()
    pos0, sh0 = r.param_logposition().detach().clone(), r.param_logshift().detach().clone()
    r.rboe = SimpleNamespace(oes={"rainbow_type": [0], "rainbow_batch": [0, 1, 2, 3]})
    r.param_logposition.extend([]); r.param_logshift.extend([3])
    _set_data(r, [0, 0], [0, 3], 11 + np.array([[.1, .2], [.5, .7]]), [1e4] * 2)
    r.fit()
    sh1 = r.param_logshift().detach()
    torch.testing.assert_close(r.param_logposition().detach(), pos0)
    torch.testing.assert_close(sh1[:3], sh0, atol=1e-6, rtol=0)
    torch.testing.assert_close(sh1[3], torch.tensor([.5, .7]) + 11 - pos0[0], atol=1e-5, rtol=0)


def test_nonfinite_medians_raise_R2_A7():
    r = _module(1, 2, 1)
    _set_data(r, [0, 0], [0, 1], np.array([[11.], [np.nan]]), [1e4] * 2)
    with pytest.raises(CalibrationModuleException, match="Non-finite"):
        r.fit()
    assert not r.is_trained


def test_fit_without_data_raises():
    with pytest.raises(CalibrationModuleException, match="training data"):
        _module(1, 2, 1).fit()


def test_unweighted_loss():
    r = _module(1, 2, 1)
    _set_data(r, [0, 0], [0, 1], np.array([[11.], [12.]]), [100., 400.])
    pred = r.get_prediction().detach()
    diff = (pred - r.data_logmedian) ** 2
    assert r.get_loss(weighted=False).item() == pytest.approx(diff.mean().item())
    assert r.get_loss().item() == pytest.approx((r.data_sqrtN[:, None] * diff).mean().item())


# ---------------------------------------------------------------- from_data on rainbow files
@pytest.fixture(scope="module")
def rainbow_adatas(rainbow_files, config):
    return [readfcs_rainbow_sample(p, config, NPEAK, name=f"rb_{b}{k}", rainbow_batch=b, rainbow_type="T")
            for b, paths in rainbow_files.items() for k, p in enumerate(paths)]


@pytest.fixture
def rboe(rainbow_adatas):
    return RainbowBatchOrdinalEncoder.from_data(RainbowCalibrator._process_adatas_meta(
        [_with_gmm_stub(ad) for ad in rainbow_adatas]))


def _with_gmm_stub(ad):
    """_process_adatas_meta needs uns['GMM_asinh_cofactor'] (not used by the scatter module)."""
    ad = ad.copy()
    ad.uns["GMM_asinh_cofactor"] = 1.
    return ad


def test_from_data_recovers_scatter_gains(rboe, rainbow_adatas):
    rsim = RainbowScatterCalibrationModule.from_data(rboe, rainbow_adatas)
    assert rsim.is_trained and rsim.has_training_data
    assert list(rsim.config["channels"]) == ["FSC-A", "SSC-A"]
    assert rsim.config["scatter_calibration_target"] == ["FSC-A", "FSC-H", "SSC-A", "SSC-H"]
    f = {b: rsim.get_calibration_factor(rainbow_batch=b) for b in synthetic.BATCHES}
    for b in synthetic.BATCHES:
        assert list(f[b].index) == ["FSC-A", "FSC-H", "SSC-A", "SSC-H"]
        assert f[b]["FSC-A"] == f[b]["FSC-H"]                         # the factor fitted on FSC-A is applied to FSC-H
        np.testing.assert_allclose(f[b] / f["A"], 1 / synthetic.SCATTER_GAINS[b], rtol=.01)
    logf = np.log(pd.DataFrame(f).T[["FSC-A", "SSC-A"]])
    np.testing.assert_allclose(logf.mean(axis=0), 0., atol=1e-5)    # geometric mean 1 over the batches


def test_explicit_scatter_calibration_R2_K6(rboe, rainbow_adatas):
    """R2-K6: an explicit scatter_calibration is checked and used (it raised NameError before)."""
    rsim = RainbowScatterCalibrationModule.from_data(rboe, rainbow_adatas, scatter_calibration={"SSC-A": ["SSC-A"]})
    assert list(rsim.config["channels"]) == ["SSC-A"]
    assert list(rsim.get_calibration_factor(rainbow_batch="B").index) == ["SSC-A"]


def test_invalid_scatter_calibration_raises(rboe, rainbow_adatas):
    with pytest.raises(DefaultScatterCalibrationException):
        RainbowScatterCalibrationModule.from_data(rboe, rainbow_adatas, scatter_calibration={"FL1-A": ["FL1-A"]})


def test_missing_channel_raises(rboe, rainbow_adatas):
    ads = [ad[:, [c for c in ad.var_names if c != "SSC-A"]].copy() for ad in rainbow_adatas]
    with pytest.raises(MissingChannelException):
        RainbowScatterCalibrationModule.from_data(rboe, ads)


def test_unknown_batch_raises(rboe, rainbow_adatas):
    ads = [ad.copy() for ad in rainbow_adatas]
    ads[0].uns["rainbow_batch"] = "Z"
    with pytest.raises(CalibrationModuleException, match="rainbow_batch"):
        RainbowScatterCalibrationModule.from_data(rboe, ads)


def test_collect_data_log_median_clipped_at_min_median(rboe, rainbow_adatas):
    ads = [ad.copy() for ad in rainbow_adatas]
    ads[0].layers["raw"][:, list(ads[0].var_names).index("FSC-A")] = 0.     # e.g. laser off: log median -inf -> 0
    rsim = RainbowScatterCalibrationModule.from_data(rboe, ads)
    assert rsim.data_logmedian[0, 0].item() == 0.
    raw = rainbow_adatas[1][:, "SSC-A"].layers["raw"][:, 0]
    assert rsim.data_logmedian[1, 1].item() == pytest.approx(np.median(np.log(raw)), rel=1e-6)
    np.testing.assert_allclose(rsim.data_sqrtN.numpy(), np.sqrt([ad.shape[0] for ad in ads]))


@pytest.mark.parametrize("key", ["rainbow_batch", "rainbow_type"])
def test_collect_data_unknown_batch_or_type_raises(rboe, rainbow_adatas, key):
    rsim = RainbowScatterCalibrationModule.from_data(rboe, rainbow_adatas)
    ad = rainbow_adatas[0].copy()
    ad.uns[key] = "not set up"
    with pytest.raises(CalibrationModuleException, match=key):
        rsim.collect_data([ad])


def test_export_round_trip(rboe, rainbow_adatas):
    rsim = RainbowScatterCalibrationModule.from_data(rboe, rainbow_adatas)
    back = RainbowScatterCalibrationModule.from_exported(rsim.export())
    assert back.is_trained and not back.has_training_data
    for b in synthetic.BATCHES:
        pd.testing.assert_series_equal(back.get_calibration_factor(rainbow_batch=b), rsim.get_calibration_factor(rainbow_batch=b))
    with pytest.raises(CalibrationModuleException, match="no training data"):
        back.plot()


def test_plot_after_fit(rboe, rainbow_adatas):
    RainbowScatterCalibrationModule.from_data(rboe, rainbow_adatas).plot()


def test_plot_untrained_raises():
    with pytest.raises(CalibrationModuleException, match="not been trained"):
        RainbowScatterCalibrationModule().plot()


def test_extend_standalone_new_batch(rainbow_adatas):
    rsim2 = RainbowScatterCalibrationModule.from_data(RainbowBatchOrdinalEncoder.from_data(
        RainbowCalibrator._process_adatas_meta([_with_gmm_stub(ad) for ad in rainbow_adatas[:4]])), rainbow_adatas[:4])                # batches A, B
    fA, fB = rsim2.get_calibration_factor(rainbow_batch="A"), rsim2.get_calibration_factor(rainbow_batch="B")
    new = rainbow_adatas[4:]                                                         # batch C
    rsim2.extend_standalone(RainbowCalibrator._process_adatas_meta([_with_gmm_stub(ad) for ad in new]), new)
    pd.testing.assert_series_equal(rsim2.get_calibration_factor(rainbow_batch="A"), fA)
    pd.testing.assert_series_equal(rsim2.get_calibration_factor(rainbow_batch="B"), fB)
    fC = rsim2.get_calibration_factor(rainbow_batch="C")
    np.testing.assert_allclose(fC / fA, 1 / synthetic.SCATTER_GAINS["C"], rtol=.01)


def test_extend_standalone_orphan_type_raises_R2b_G2_05(rainbow_adatas):
    """R2b-G2-05: extend_standalone with a new type measured only in new batches raises before changing anything."""
    ads = [_with_gmm_stub(ad) for ad in rainbow_adatas]
    rboe = RainbowBatchOrdinalEncoder.from_data(RainbowCalibrator._process_adatas_meta(ads[:4]))
    rsim = RainbowScatterCalibrationModule.from_data(rboe, ads[:4])
    new = [ad.copy() for ad in ads[4:]]
    for ad in new:
        ad.uns["rainbow_type"] = "T2"
    with pytest.raises(CalibrationModuleException, match="T2"):
        rsim.extend_standalone(RainbowCalibrator._process_adatas_meta(new), new)
    assert list(rsim.rboe.oes["rainbow_batch"].labels) == ["A", "B"]
