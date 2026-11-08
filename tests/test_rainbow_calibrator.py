"""RainbowCalibrator and its fluorescence model and dataset (cytovanni.rainbow.rbim, rbim_fluorescence, rbim_data).

Ground truth (tests/synthetic.py): batch b measures ``PEAKS * GAINS[b]`` (fluorescence) and ``SCATTER_GAINS[b]``
(scatter), so the calibration factor of b is ``1 / gain``, normalised to geometric mean 1 over the batches of the
first fit. The fluorescence fits here use a shortened training schedule (`ShortTrainer`); the full schedule of
`fit_fluorescence` (12000 steps) is too slow for the test suite.
"""
import os
import warnings

import flowkit as fk
import numpy as np
import pandas as pd
import pytest
import torch

import cytovanni.rainbow.rbim as rbim
from synthetic import ShortTrainer, expected_rainbow_factors, read_rainbow_fit, short_fit_fluorescence
from cytovanni.exceptions import (CalibrationModuleException, CalibrationModuleUntrainedParameterWarning,
                                  CalibrationModuleWarning, MissingChannelException, MissingGPUWarning)
from cytovanni.io import readfcs_rainbow_sample, readfcs_sample, writefcs
from cytovanni.rainbow import (RainbowCalibrator, RainbowFluorescenceGMMCalibrationDataset,
                               RainbowFluorescenceGMMCalibrationModule, RainbowFluorescenceGMMCalibrationTrainer,
                               fit_rainbow_GMM, plot_eval_RBInt_fit, scale_spectra_wfactors)

import synthetic

FL = synthetic.CHANNELS_FLUORESCENCE
NPEAK = len(synthetic.PEAKS)
SCATTER_TARGETS = ["FSC-A", "FSC-H", "SSC-A", "SSC-H"]


@pytest.fixture
def ads(fitted_rainbow_ads):
    return [ad.copy() for ad in fitted_rainbow_ads]


@pytest.fixture(scope="module")
def rbint(fitted_rainbow_ads):
    """Calibrator on all six measurements, fluorescence fitted on the CPU with the short schedule (do not modify)."""
    rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads])
    short_fit_fluorescence(rb)
    return rb


@pytest.fixture(scope="module")
def saved_path(rbint, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rbint") / "sub" / "rb.pt")
    rbint.save(path, make_path_structure=True)
    return path


@pytest.fixture(scope="module")
def batch_D(config, tmp_path_factory):
    """Two measurements of a new batch 'D' with the gains of batch C (same type 'T'), and a new measurement of the
    known batch C (last)."""
    folder = tmp_path_factory.mktemp("rainbow_D")
    new = [read_rainbow_fit(synthetic.write_rainbow_fcs(str(folder / f"rb_D_{k}.fcs"), "C", btim=f"1{2+k}:00:00", seed=50 + k),
                    config, "D") for k in range(2)]
    new.append(read_rainbow_fit(synthetic.write_rainbow_fcs(str(folder / "rb_C_new.fcs"), "C", btim="10:00:00", seed=60),
                        config, "C"))
    return new


# ======================================================================== dataset
def test_dataset_layout(rbint, ads):
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, Nsubsample=100)
    assert list(ds.present_types) == ["T"] and ds.channels == FL
    assert ds.data["T"].shape == (sum(ad.shape[0] for ad in ads), len(FL)) and ds.data["T"].dtype == torch.float32
    assert ds.GMM_means["T"].shape == (len(ads), NPEAK, len(FL))
    assert ds.peak_usemask["T"].shape == (len(ads), NPEAK, len(FL)) and ds.peak_usemask["T"].all()
    assert ds.peak_shares["T"].shape == (len(ads), NPEAK)
    torch.testing.assert_close(ds.peak_shares["T"].sum(-1), torch.ones(len(ads)))
    assert (ds.peak_shares["T"] - 1 / NPEAK).abs().max() < .05
    assert list(ds.df_channel_train.index) == synthetic.BATCHES and ds.df_channel_train.to_numpy().all()
    np.testing.assert_array_equal(ds.peak_shares_batchidx["T"].numpy(), [0, 0, 1, 1, 2, 2])


def test_dataset_subsamples_per_measurement(rbint, ads):
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, Nsubsample=100, batch_size=250)
    inds = ds.get_sample_indices()["T"]
    assert len(inds) == 100 * len(ads)
    np.testing.assert_array_equal(np.bincount(ds.data_uid_index["T"][inds].numpy()), [100] * len(ads))
    loaders = ds.sample_loaders()
    assert len(loaders["T"]) == 600 and len(list(loaders["T"])) == 3
    batch = next(iter(loaders["T"]))
    assert batch["data"].shape == (250, len(FL)) and batch["uid_idx"].shape == (250,)
    assert batch["batch_idx"].shape == (250,) and batch["peak_islinear"].shape == (250, NPEAK, len(FL))
    full = ds.sample()
    assert full["data"]["T"].shape == (600, len(FL)) and full["peak_islinear"]["T"].shape == (600, NPEAK, len(FL))
    assert len(next(iter(ds.sample_loaders(batch_size_override=600)["T"]))["data"]) == 600


def test_dataset_peaks_above_linear_range_are_not_used(rbint, ads):
    for ad in ads[4:]:                               # batch C: gains .8, 1.3, 1.0 -> brightest peak 32000-52000
        ad.uns["cytoconfig"] = synthetic.make_config(max_linear_range=1e4)
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads)
    mask = ds.peak_usemask["T"].numpy()
    assert mask[:4].all() and mask[4:, :2].all() and not mask[4:, 2].any()
    assert ds.df_channel_train.to_numpy().all()      # the other peaks still train C


def test_dataset_untrainable_batch_warns(rbint, ads):
    for ad in ads[4:]:
        ad.uns["cytoconfig"] = synthetic.make_config(max_linear_range=100.)
    with pytest.warns(CalibrationModuleWarning, match="will not get trained"):
        ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads)
    assert not ds.df_channel_train.loc["C"].any() and ds.df_channel_train.loc[["A", "B"]].to_numpy().all()
    assert not ds.data_usemask["T"][ds.data_intype_index["T"] >= 4].any()      # beads of C never sampled
    assert all(len(ds.data_indices_batched["T"][i]) == 0 for i in [4, 5])


def test_dataset_channel_missing_cutoff(rbint, ads):
    """A channel counts as missing in a measurement if every peak is below the cutoff (brightest peak 4e4 * gain)."""
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, channel_missing_cutoff=5e4, warn_missing=False)
    missing = (synthetic.PEAKS[-1] * synthetic.GAINS < 5e4)                 # (batch, channel)
    pd.testing.assert_frame_equal(ds.df_channel_train, ~missing, check_dtype=False)
    usemask = ds.peak_usemask["T"].numpy()
    for i, b in enumerate(["A", "A", "B", "B", "C", "C"]):
        np.testing.assert_array_equal(usemask[i].all(axis=0), ~missing.loc[b].to_numpy())


def test_dataset_warn_missing_false_still_sets_df_channel_train_R2b_G2_10(rbint, ads):
    """R2b-G2-10: df_channel_train (needed by the trainer) is set with warn_missing=False too; only the warning
    depends on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds_f = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, warn_missing=False, channel_missing_cutoff=1e9)
    with pytest.warns(CalibrationModuleWarning):
        ds_t = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, warn_missing=True, channel_missing_cutoff=1e9)
    pd.testing.assert_frame_equal(ds_f.df_channel_train, ds_t.df_channel_train)
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, warn_missing=False)
    rb = RainbowCalibrator(ads)
    ShortTrainer(rb.rfim).fit_init(ds)                                   # AttributeError before the fix
    assert rb.rfim.df_batch_trained.to_numpy().all()


def test_dataset_inconsistent_peak_numbers_raise_R2_C4(ads):
    """R2-C4: different numbers of GMM peaks within a rainbow type, or a number different from `prior`, raise."""
    cls = RainbowFluorescenceGMMCalibrationDataset
    assert cls.get_type_Npeak(ads) == {"T": NPEAK}
    cls.check_type_Npeak_consistency(ads, prior={"T": NPEAK})
    with pytest.raises(CalibrationModuleException, match="not consistent"):
        cls.check_type_Npeak_consistency(ads, prior={"T": NPEAK + 1})
    ads[1].uns["GMM_means"] = ads[1].uns["GMM_means"][:2]
    with pytest.raises(CalibrationModuleException, match="not consistent"):
        cls.check_type_Npeak_consistency(ads)
    with pytest.raises(CalibrationModuleException):
        cls.get_type_Npeak(ads)
    with pytest.raises(CalibrationModuleException):
        RainbowCalibrator(ads)


def test_dataset_requires_gmm_fit(ads):
    del ads[0].obs["GMM_label"]
    with pytest.raises(CalibrationModuleException, match="GMM"):
        RainbowFluorescenceGMMCalibrationDataset.check_GMM_setup(ads)


@pytest.mark.parametrize("key", ["rainbow_batch", "rainbow_type", "uid"])
def test_dataset_unknown_labels_raise(rbint, ads, key):
    ads[0].uns[key] = "unknown"
    with pytest.raises(CalibrationModuleException):
        RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads)


def test_dataset_on_device(rbint, ads, device):
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, device=device, Nsubsample=50)
    assert ds.data["T"].device.type == device and ds.peak_usemask["T"].device.type == device
    batch = next(iter(ds.sample_loaders()["T"]))
    assert batch["data"].device.type == device and batch["peak_islinear"].device.type == device


# ======================================================================== construction and scatter
def test_construction_checks_required_entries(ads):
    del ads[0].uns["set_hash"]
    del ads[1].uns["GMM_means"]
    with pytest.warns(CalibrationModuleWarning) as rec:
        with pytest.raises(CalibrationModuleException, match="missing"):
            RainbowCalibrator(ads)
    text = " ".join(str(w.message) for w in rec)
    assert "set_hash" in text and "GMM_means" in text


def test_construction_missing_fluorescence_channel_raises(ads):
    ads[2] = ads[2][:, [c for c in ads[2].var_names if c != "FL2-A"]].copy()
    with pytest.raises(MissingChannelException):
        RainbowCalibrator(ads)


def test_empty_calibrator():
    rb = RainbowCalibrator()
    assert not hasattr(rb, "rfim") and rb.channel_missing_cutoff is None


@pytest.mark.parametrize("drop", ["obs", "layer"])
def test_construction_without_gmm_obs_or_raw_layer_warns_and_raises(ads, drop):
    """Inputs that were not set up with fit_rainbow_GMM (obs labels) or lack the raw data are refused."""
    if drop == "obs":
        del ads[1].obs[RainbowCalibrator.REQUIRED_GMM_OBS[0]]
    else:
        del ads[1].layers["raw"]
    with pytest.warns(CalibrationModuleWarning, match="missing"):
        with pytest.raises(CalibrationModuleException):
            RainbowCalibrator(ads)


def test_construction_sets_up_models(ads):
    rb = RainbowCalibrator(ads)
    assert rb.rsim.is_trained and not rb.rfim.is_trained
    assert rb.rfim.channels == FL and rb.rfim.dct_Npeak == {"T": NPEAK}
    cof = np.exp(np.mean(np.log([ad.uns["GMM_asinh_cofactor"] for ad in ads])))
    assert rb.rfim.class_cofactor["T"] == pytest.approx(cof)
    assert list(rb.rboe.oes["rainbow_batch"].labels) == synthetic.BATCHES


# ======================================================================== recovery (C)
def test_fluorescence_factors_recover_gains(rbint):
    F = rbint.get_calibration_factors_all()
    assert list(F.index) == synthetic.BATCHES and list(F.columns) == SCATTER_TARGETS + FL
    np.testing.assert_allclose(F[FL], expected_rainbow_factors(), rtol=.02)
    np.testing.assert_allclose(np.log(F[FL]).mean(axis=0), 0., atol=1e-5)          # geometric mean 1
    # scatter: factor ratio to A is 1 / scatter gain
    np.testing.assert_allclose(F["FSC-A"] / F.loc["A", "FSC-A"], 1 / synthetic.SCATTER_GAINS, rtol=.01)
    assert rbint.rfim.is_trained and rbint.rfim_trainer.is_trained


def test_calibrated_rainbow_peaks_coincide(rbint, ads):
    """After calibration the GMM peak means of all batches agree (the purpose of the calibration)."""
    means = []
    for ad in ads:
        f = rbint.get_calibration_factor(adata=ad)
        means.append(ad.uns["GMM_means"] * f[FL].to_numpy()[None])
    means = np.array(means)
    np.testing.assert_allclose(means, np.broadcast_to(means.mean(0), means.shape), rtol=.03)


def test_get_calibration_factor_by_name_and_by_adata(rbint, ads):
    f = rbint.get_calibration_factor(rainbow_batch="B")
    assert list(f.index) == SCATTER_TARGETS + FL
    pd.testing.assert_series_equal(rbint.get_calibration_factor(adata=ads[2]), f)
    f_all, mask = rbint.get_calibration_factor_all_single("B")
    pd.testing.assert_series_equal(f_all, f)
    assert mask.all() and list(mask.index) == FL


def test_get_calibration_factor_infers_batch_of_cells(rbint, cell_files, config):
    """A sample without 'rainbow_batch' gets the batch of the closest rainbow measurement with the same settings."""
    for b in synthetic.BATCHES:
        cells = readfcs_sample(cell_files[b], config)
        assert "rainbow_batch" not in cells.uns
        pd.testing.assert_series_equal(rbint.get_calibration_factor(adata=cells), rbint.get_calibration_factor(rainbow_batch=b))


@pytest.mark.slow
def test_longer_fit_recovers_gains_and_shift(fitted_rainbow_ads, config, tmp_path):
    """Longer schedule: factors within 1 %; a measurement with an added offset of +300 (raw units) gets a fitted shift
    300 lower than the other measurements of its batch (calibrated = (raw + shift) * factor)."""
    path = synthetic.write_rainbow_fcs(str(tmp_path / "rb_A_shift.fcs"), "A", btim="10:00:00", seed=77, shift=300.)
    shifted = read_rainbow_fit(path, config, "A")
    rb = RainbowCalibrator([ad.copy() for ad in fitted_rainbow_ads] + [shifted])

    class Longer(RainbowFluorescenceGMMCalibrationTrainer):
        def fit_init_peak(self, dataset, Niter=1000, **kw):
            return super().fit_init_peak(dataset, Niter=Niter, **kw)

        def fit_init_cov(self, dataset, Niter=200, **kw):
            return super().fit_init_cov(dataset, Niter=Niter, **kw)

        def fit(self, dataset, Niter=500, **kw):
            return super().fit(dataset, Niter=Niter, **kw)
    short_fit_fluorescence(rb, trainer=Longer)
    F = rb.get_calibration_factors_all()
    np.testing.assert_allclose(F[FL], expected_rainbow_factors(), rtol=.01)
    shifts = rb.get_rainbow_shifts_all()["T"]
    others = [ad.uns["uid"] for ad in fitted_rainbow_ads[:2]]
    diff = shifts.loc[shifted.uns["uid"]] - shifts.loc[others].mean()
    np.testing.assert_allclose(diff, -300., atol=60.)


# ======================================================================== applying the calibration
def test_add_calibrated_multiplies_raw(rbint, ads):
    ad = ads[2]
    rbint.add_calibrated(ad)
    f = rbint.get_calibration_factor(rainbow_batch="B").loc[list(ad.var_names)]
    np.testing.assert_allclose(ad.layers["calibrated"], ad.layers["raw"] * f.to_numpy()[None], rtol=1e-6)
    rbint.add_calibrated(ad, rainbow_batch="C", addlayer="asC")
    fC = rbint.get_calibration_factor(rainbow_batch="C").loc[list(ad.var_names)]
    np.testing.assert_allclose(ad.layers["asC"], ad.layers["raw"] * fC.to_numpy()[None], rtol=1e-6)


def test_add_calibrated_sample_without_a_calibrated_channel_R2_A2(rbint, ads):
    """R2-A2: a calibrated channel missing from the sample (FSC-H) is skipped with a warning; its factor is not
    applied to any other column (it hit the last column before)."""
    ad = ads[2][:, [c for c in ads[2].var_names if c != "FSC-H"]].copy()
    f = rbint.get_calibration_factor(rainbow_batch="B")
    with pytest.warns(CalibrationModuleWarning, match="FSC-H"):
        rbint.add_calibrated(ad)
    np.testing.assert_allclose(ad.layers["calibrated"], ad.layers["raw"] * f.loc[list(ad.var_names)].to_numpy()[None], rtol=1e-6)
    with pytest.warns(CalibrationModuleWarning, match="FSC-H"):
        rbint.add_calibrated_rainbow(ad, include_shift=False)
    np.testing.assert_allclose(ad.layers["calibrated"], ad.layers["raw"] * f.loc[list(ad.var_names)].to_numpy()[None], rtol=1e-6)
    with pytest.warns(CalibrationModuleWarning, match="FSC-H"):
        rbint.add_calibration_factor(ad)
    assert "FSC-H" not in ad.var.index
    np.testing.assert_allclose(ad.var["rainbow_calibration_factor"], f.loc[list(ad.var_names)])


def test_add_calibration_factor_nan_for_channels_without_factor(rbint):
    ad = synthetic.make_adata(np.ones((3, 4)), FL + ["X-A"], layer="raw", rainbow_batch="C")
    with pytest.warns(CalibrationModuleWarning):          # the scatter channels are not in this sample
        rbint.add_calibration_factor(ad, addkey="f")
    assert np.isnan(ad.var.loc["X-A", "f"])
    np.testing.assert_allclose(ad.var.loc[FL, "f"], rbint.get_calibration_factor(rainbow_batch="C")[FL])


def test_add_calibrated_rainbow_with_and_without_shift(rbint, ads):
    ad = ads[4]
    f = rbint.get_calibration_factor(rainbow_batch="C").loc[list(ad.var_names)].to_numpy()
    shift = rbint.rfim.get_calibration_rainbow_shift(adata=ad)
    s = pd.Series(0., index=ad.var_names); s[FL] = shift
    rbint.add_calibrated_rainbow(ad, addlayer="with")
    rbint.add_calibrated_rainbow(ad, include_shift=False, addlayer="without")
    rbint.add_calibrated_rainbow(ad, only_shift=True, addlayer="shiftonly")
    raw = ad.layers["raw"]
    np.testing.assert_allclose(ad.layers["with"], (raw + s.to_numpy()[None]) * f[None], rtol=1e-5)
    np.testing.assert_allclose(ad.layers["without"], raw * f[None], rtol=1e-6)
    np.testing.assert_allclose(ad.layers["shiftonly"], raw + s.to_numpy()[None], rtol=1e-6)


def test_rainbow_shifts_all(rbint, ads):
    shifts = rbint.get_rainbow_shifts_all()
    assert list(shifts) == ["T"]
    df = shifts["T"]
    assert list(df.columns) == FL and set(df.index) == {ad.uns["uid"] for ad in ads}
    pd.testing.assert_series_equal(df.loc[ads[3].uns["uid"]], rbint.rfim.get_calibration_rainbow_shift(adata=ads[3]),
                                   check_names=False)


def test_rainbow_peak_parameters_describe_the_shifted_data(rbint, ads):
    ad = ads[2]
    mean, cov, shares, islinear, cof = rbint.get_rainbow_peak_parameters(ad, ["FL3-A", "FL1-A"])
    assert mean.shape == (NPEAK, 2) and cov.shape == (NPEAK, 2, 2) and shares.shape == (NPEAK,)
    assert islinear.shape == (NPEAK, 2) and islinear.dtype == bool and islinear.all()
    assert cof == rbint.rfim.class_cofactor["T"]
    assert shares.sum() == pytest.approx(1., abs=1e-5)
    shift = rbint.rfim.get_calibration_rainbow_shift(adata=ad)[["FL3-A", "FL1-A"]].to_numpy()
    gmm = np.arcsinh((ad.uns["GMM_means"][:, [2, 0]] + shift[None]) / cof)
    np.testing.assert_allclose(mean, gmm, atol=.05)
    np.testing.assert_allclose(cov, np.transpose(cov, (0, 2, 1)), atol=1e-6)
    assert (np.linalg.eigvalsh(cov) > 0).all()


def test_apply_arcsinh(rbint):
    x = np.array([0., 100., 1e4])
    np.testing.assert_allclose(rbint.apply_ArcSinh(x, "T"), np.arcsinh(x / rbint.rfim.class_cofactor["T"]))


def test_calibrated_spectra(rbint):
    spectra = synthetic.SPECTRA.T
    out = rbint.get_calibrated_spectra(spectra, "B", spectral=True)
    pd.testing.assert_frame_equal(out, scale_spectra_wfactors(spectra, rbint.get_calibration_factor(rainbow_batch="B"), spectral=True))
    conv = rbint.get_calibrated_spectra(spectra, "B")
    np.testing.assert_allclose(np.diag(conv.loc[FL, synthetic.STAINS]), 1.)


# ======================================================================== untrained factors
def test_untrained_factors_are_one_with_warning(ads):
    for ad in ads[4:]:
        ad.uns["cytoconfig"] = synthetic.make_config(max_linear_range=100.)
    rb = RainbowCalibrator(ads)
    with pytest.warns(CalibrationModuleWarning):
        ds = RainbowFluorescenceGMMCalibrationDataset(rb.rboe, ads)
    rb.rfim.remember_trainable_batches(ds.df_channel_train)
    with torch.no_grad():
        rb.rfim.param_logscalefactor.param_trainable.data[:] = torch.tensor([[.3], [0.], [-.3]])   # as if fitted
    with pytest.warns(CalibrationModuleUntrainedParameterWarning, match="set to 1"):
        f = rb.get_calibration_factor(rainbow_batch="C")
    assert (f[FL] == 1.).all()
    f_raw, mask = rb.rfim.get_calibration_factor(rainbow_batch="C", fix_untrained=False, warn_untrained=False,
                                                 return_mask=True)
    assert not mask.any() and not np.allclose(f_raw, 1.)
    with pytest.warns(CalibrationModuleUntrainedParameterWarning, match="not set to 1"):
        rb.rfim.get_calibration_factor(rainbow_batch="C", fix_untrained=False)
    with pytest.warns(CalibrationModuleUntrainedParameterWarning, match="3 \\(out of 3\\) channels"):
        F = rb.get_calibration_factors_all()
    assert (F.loc["C", FL] == 1.).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rb.get_calibration_factors_all(warn_untrained=False)
        rb.get_calibration_factor_all_single("C")


# ======================================================================== fluorescence module
def test_rainbow_shift_lookup_fallbacks_and_errors(rbint, ads):
    rfim, ad = rbint.rfim, ads[0]
    ref = rfim.get_calibration_rainbow_shift(adata=ad)
    pd.testing.assert_series_equal(rfim.get_calibration_rainbow_shift(uid=ad.uns["uid"], tp="T"), ref)
    with pytest.warns(CalibrationModuleWarning, match="Uid"):
        pd.testing.assert_series_equal(rfim.get_calibration_rainbow_shift(adata=ad, uid="nope"), ref)
    with pytest.warns(CalibrationModuleWarning, match="rainbow type"):
        pd.testing.assert_series_equal(rfim.get_calibration_rainbow_shift(adata=ad, tp="nope"), ref)
    with pytest.raises(CalibrationModuleException, match="requires either"):
        rfim.get_calibration_rainbow_shift(uid=ad.uns["uid"])
    with pytest.raises(CalibrationModuleException, match="Did not find provided uid"):
        rfim.get_calibration_rainbow_shift(uid="nope", tp="T")
    with pytest.raises(CalibrationModuleException, match="Did not find provided rainbow type"):
        rfim.get_calibration_rainbow_shift(uid=ad.uns["uid"], tp="nope")
    bad = ad.copy(); bad.uns["uid"] = "nope"
    with pytest.raises(CalibrationModuleException, match="not found"):
        rfim.get_calibration_rainbow_shift(adata=bad)
    del bad.uns["uid"]
    with pytest.raises(CalibrationModuleException, match="no 'uid'"):
        rfim.get_calibration_rainbow_shift(adata=bad)
    bad = ad.copy(); del bad.uns["rainbow_type"]
    with pytest.raises(CalibrationModuleException, match="no 'rainbow_type'"):
        rfim.get_calibration_rainbow_shift(adata=bad)


def test_fluorescence_module_export_round_trip(rbint):
    back = RainbowFluorescenceGMMCalibrationModule.from_exported(rbint.rfim.export())
    assert back.is_trained and back.dct_Npeak == rbint.rfim.dct_Npeak
    assert sorted(back.frozen_batches) == [0, 1, 2]
    pd.testing.assert_frame_equal(back.df_batch_trained, rbint.rfim.df_batch_trained, check_dtype=False)
    for b in synthetic.BATCHES:
        pd.testing.assert_series_equal(back.get_calibration_factor(rainbow_batch=b),
                                       rbint.rfim.get_calibration_factor(rainbow_batch=b), rtol=1e-6)


def test_fluorescence_module_extend_with_other_peak_number_raises(ads):
    rb = RainbowCalibrator(ads)
    with pytest.raises(CalibrationModuleException, match="peaks"):
        rb.rfim.extend(rb.rboe, {"rainbow_batch": [], "uid": [], "rainbow_type": []}, dct_Npeak={"T": NPEAK + 1})


def test_dist_parameters_and_nll(rbint, ads):
    rfim = rbint.rfim
    with pytest.raises(ValueError):
        rfim.get_dist_parameters({"T": torch.tensor([0])}, {})
    ds = RainbowFluorescenceGMMCalibrationDataset(rbint.rboe, ads, Nsubsample=20)
    batch = next(iter(ds.sample_loaders()["T"]))
    rfim.update_params()
    nll = rfim.get_nll_batch(batch, "T", accumulate=False)
    assert nll.shape == (len(batch["data"]),) and torch.isfinite(nll).all()
    params = rfim.get_dist_parameters_type("T", batch["uid_idx"], batch["batch_idx"], batch["peak_islinear"])
    n = len(batch["data"])
    assert params["peak_position_scaled"].shape == (n, NPEAK, len(FL))
    assert params["precision"].shape == (n, NPEAK, len(FL), len(FL)) and params["log_classshares"].shape == (n, NPEAK)
    torch.testing.assert_close(params["peak_position_scaled"], params["peak_position"] * params["scale"])


def test_trainer_plots(rbint):
    rbint.rfim_trainer.plot_fit_init()
    rbint.rfim_trainer.plot_fit()


# ======================================================================== fit_fluorescence options, save / load
@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CPU fallback without CUDA")
def test_fit_fluorescence_gpu_fallback_and_cutoff_warning(ads):
    rb = RainbowCalibrator(ads[:4])
    rb.channel_missing_cutoff = 5.
    with pytest.warns(MissingGPUWarning):
        with pytest.warns(CalibrationModuleWarning, match="channel_missing_cutoff"):
            short_fit_fluorescence(rb, gpu=True, verbosity=2)
    assert next(rb.rfim.parameters()).device.type == "cpu" and rb.channel_missing_cutoff == -np.inf


def test_fit_fluorescence_warns_on_changed_channel_missing_cutoff(ads):
    rb = RainbowCalibrator(ads[:4])
    rb.channel_missing_cutoff = 5.
    with pytest.warns(CalibrationModuleWarning, match="channel_missing_cutoff"):
        short_fit_fluorescence(rb, gpu=False)
    assert rb.channel_missing_cutoff == -np.inf


def test_fit_fluorescence_after_loading_raises(saved_path):
    rb = RainbowCalibrator.from_saved(saved_path)
    with pytest.raises(CalibrationModuleException, match="extend"):
        rb.fit_fluorescence(gpu=False, verbosity=0)


def test_fit_fluorescence_on_device_then_peak_parameters_R2b_G2_01(ads, device, tmp_path):
    """R2b-G2-01: after a fit on the GPU, get_rainbow_peak_parameters and plot_rainbow_with_PDF work (numpy outputs)
    and equal those of the calibrator reloaded on the CPU."""
    rb = RainbowCalibrator(ads[:4])
    short_fit_fluorescence(rb, gpu=device == "cuda", device="cuda:0")
    assert next(rb.rfim.parameters()).device.type == device
    out = rb.get_rainbow_peak_parameters(ads[2], FL)
    assert all(isinstance(x, np.ndarray) for x in out[:4])
    rb.plot_rainbow_with_PDF(ads[2])
    rb.save(str(tmp_path / "rb.pt"))
    back = RainbowCalibrator.from_saved(str(tmp_path / "rb.pt"))
    for a, b in zip(out[:4], back.get_rainbow_peak_parameters(ads[2], FL)[:4]):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_save_and_load(rbint, saved_path):
    back = RainbowCalibrator.from_saved(saved_path)
    pd.testing.assert_frame_equal(back.get_calibration_factors_all(), rbint.get_calibration_factors_all())
    assert back.channel_missing_cutoff == rbint.channel_missing_cutoff == -np.inf
    assert not hasattr(back, "adatas")
    pd.testing.assert_frame_equal(back.get_rainbow_shifts_all()["T"], rbint.get_rainbow_shifts_all()["T"])
    with pytest.raises(CalibrationModuleException, match="no training data"):
        back.fit_fluorescence(gpu=False)


def test_save_without_folder_raises(rbint, tmp_path):
    with pytest.raises(Exception):
        rbint.save(str(tmp_path / "missing" / "rb.pt"))


def test_extend_fit_new_batch_keeps_known_factors(saved_path, batch_D):
    """extend_fit on a loaded calibrator: known factors unchanged, the new batch D (gains of C) gets C's factors;
    a new measurement of the known batch C is possible."""
    rb = RainbowCalibrator.from_saved(saved_path)
    before = rb.get_calibration_factors_all()
    new = [ad.copy() for ad in batch_D]
    short_fit_fluorescence_extend(rb, new)
    after = rb.get_calibration_factors_all()
    pd.testing.assert_frame_equal(after.loc[synthetic.BATCHES], before)
    np.testing.assert_allclose(after.loc["D"], after.loc["C"], rtol=.02)
    assert list(rb.rboe.oes["rainbow_batch"].labels) == synthetic.BATCHES + ["D"]
    assert rb.adatas is not None and len(rb.adatas) == 3


def short_fit_fluorescence_extend(rb, adatas):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rbim, "RainbowFluorescenceGMMCalibrationTrainer", ShortTrainer)
        rb.extend_fit(adatas, gpu=False, verbosity=0)


def test_extend_fit_orphan_type_raises_before_changes_R2b_G2_05(saved_path, batch_D):
    """R2b-G2-05: a new rainbow type measured only in new batches raises CalibrationModuleException before the
    encoder or the training data are changed."""
    rb = RainbowCalibrator.from_saved(saved_path)
    new = [ad.copy() for ad in batch_D[:2]]
    for ad in new:
        ad.uns["rainbow_type"] = "T2"
    with pytest.raises(CalibrationModuleException, match="T2"):
        short_fit_fluorescence_extend(rb, new)
    assert list(rb.rboe.oes["rainbow_batch"].labels) == synthetic.BATCHES
    assert list(rb.rboe.oes["rainbow_type"].labels) == ["T"] and not hasattr(rb, "adatas")


def test_extend_fit_new_type_in_new_batch_with_known_type(saved_path, batch_D):
    rb = RainbowCalibrator.from_saved(saved_path)
    new = [ad.copy() for ad in batch_D[:2]]
    new[1].uns["rainbow_type"] = "T2"                                  # T2 measured together with T in batch D
    short_fit_fluorescence_extend(rb, new)
    assert set(rb.rfim.dct_Npeak) == {"T", "T2"} and "T2" in rb.rfim.class_cofactor
    assert set(rb.get_rainbow_shifts_all()) == {"T", "T2"}


# ======================================================================== .fcs output
def test_make_calibrated_fcs(rbint, cell_files, config, tmp_path):
    out = str(tmp_path / "cal.fcs")
    rbint.make_calibrated_fcs(cell_files["B"], out, config)
    raw = readfcs_sample(cell_files["B"], config)
    cal = readfcs_sample(out, config)
    f = rbint.get_calibration_factor(rainbow_batch="B").loc[list(raw.var_names)].to_numpy()
    np.testing.assert_allclose(cal.layers["raw"], raw.layers["raw"] * f[None], rtol=1e-5)
    np.testing.assert_allclose(cal.obs["Time"].astype(float), raw.obs["Time"].astype(float))
    assert fk.Sample(out, preprocess=False).metadata["__cytovanni_rainbow_calibrated"] == "True"
    # a marked file is not calibrated again
    out2 = str(tmp_path / "cal2.fcs")
    with pytest.warns(UserWarning, match="already been calibrated"):
        rbint.make_calibrated_fcs(out, out2, config)
    np.testing.assert_allclose(readfcs_sample(out2, config).layers["raw"], cal.layers["raw"], rtol=1e-6)
    out3 = str(tmp_path / "cal3.fcs")
    rbint.make_calibrated_fcs(out, out3, config, force=True, rainbow_batch="B")
    np.testing.assert_allclose(readfcs_sample(out3, config).layers["raw"], cal.layers["raw"] * f[None], rtol=1e-5)


class _ThresholdGate:
    def __init__(self, channel, threshold):
        self.channel, self.threshold = channel, threshold

    def apply_df(self, df):
        return (df[self.channel] > self.threshold).to_numpy()


def test_make_calibrated_fcs_gates_on_raw_values(rbint, cell_files, config, tmp_path):
    out = str(tmp_path / "gated.fcs")
    raw = readfcs_sample(cell_files["A"], config)
    keep = raw.to_df(layer="raw")["FL1-A"].to_numpy() > 2000.
    rbint.make_calibrated_fcs(cell_files["A"], out, config, gates=[_ThresholdGate("FL1-A", 2000.)])
    cal = readfcs_sample(out, config)
    f = rbint.get_calibration_factor(rainbow_batch="A").loc[list(raw.var_names)].to_numpy()
    assert cal.shape[0] == keep.sum()
    np.testing.assert_allclose(cal.layers["raw"], raw.layers["raw"][keep] * f[None], rtol=1e-5)


def test_make_calibrated_fcs_renamed_channels_R2_A1(tmp_path):
    """R2-A1: with a configuration that renames channels (FACSAria), the file's channels are calibrated under their
    canonical names and written back with the original $PnN names."""
    from types import SimpleNamespace
    from cytovanni.utils import CytometerConfiguration_FACSAriaIII
    cfg = CytometerConfiguration_FACSAriaIII()
    inv = {v: k for k, v in cfg.replace_channelname_dict.items()}
    raw_fl = [inv[c] for c in cfg.channels_fluorescence]
    cols = ["FSC-A", "FSC-H", "SSC-A", "SSC-H"] + raw_fl + ["Time"]
    df = pd.DataFrame(np.random.default_rng(0).uniform(100, 1000, size=(50, len(cols))), columns=cols)
    fin, fout = str(tmp_path / "aria_in.fcs"), str(tmp_path / "aria_out.fcs")
    writefcs(fin, df)
    factor = pd.Series(2., index=cfg.channels_fluorescence)
    stub = SimpleNamespace(get_calibration_factor=lambda adata=None, rainbow_batch=None: factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        RainbowCalibrator.make_calibrated_fcs(stub, fin, fout, cfg)
    s = fk.Sample(fout, preprocess=False)
    d = s.as_dataframe(source="raw")
    assert list(d.columns.get_level_values(0)) == cols                   # original names kept
    np.testing.assert_allclose(d[raw_fl[0]].to_numpy().ravel(), 2 * df[raw_fl[0]].to_numpy(), rtol=1e-5)
    np.testing.assert_allclose(d["FSC-A"].to_numpy().ravel(), df["FSC-A"].to_numpy(), rtol=1e-6)


def test_make_calibrated_fcs_without_calibrated_channels_raises(tmp_path, config):
    from types import SimpleNamespace
    df, _ = synthetic.cell_events("A", n=50)
    fin = str(tmp_path / "in.fcs")
    writefcs(fin, df, add_metadata=synthetic.fcs_metadata("A"))
    stub = SimpleNamespace(get_calibration_factor=lambda adata=None, rainbow_batch=None: pd.Series(2., index=["X-A"]))
    with pytest.raises(MissingChannelException):
        RainbowCalibrator.make_calibrated_fcs(stub, fin, str(tmp_path / "out.fcs"), config)


# ======================================================================== plots
def test_plot_eval_fit_adds_layers(rbint, ads):
    plot_eval_RBInt_fit(rbint, ads[:3], legendkey="rainbow_batch", ylim=(0, 1), bins=50)
    for ad in ads[:3]:
        assert {"calibrated_rb", "calibrated_rb_noshift"} <= set(ad.layers)
    plot_eval_RBInt_fit(rbint, ads[3:4], channels=["FL2-A"])


def test_plot_rainbow_with_pdf(rbint, ads):
    rbint.plot_rainbow_with_PDF(ads[1], channels=["FL1-A"], bins=50)
    assert "_calibratedos" in ads[1].layers
    rbint.plot_rainbow_with_PDF(ads[1])
    rbint.rsim.plot()
