"""cytovanni.rainbow.utils: spectra scaling with calibration factors, laser/channel split of log factors, peak order."""
import warnings

import numpy as np
import pandas as pd
import pytest

from cytovanni.exceptions import CalibrationModuleException, SpilloverWarning
from cytovanni.rainbow import (scale_spectra_wfactors, extract_logfactor_laserfactor,
                               extract_logfactor_laserfactor_bylaser, reorder_peaks)

import synthetic


# ---------------------------------------------------------------- scale_spectra_wfactors
def _spectra_channel_dye():
    """Synthetic spectra in (channel, dye) orientation, conventional normalisation (exactly one entry 1 per dye)."""
    return synthetic.SPECTRA.T.copy()


def test_scale_spectra_conventional_keeps_the_unit_entry():
    spectra = _spectra_channel_dye()
    factors = pd.Series([2., .5, 4.], index=synthetic.CHANNELS_FLUORESCENCE)
    out = scale_spectra_wfactors(spectra, factors, spectral=False)
    assert out.shape == spectra.shape and list(out.index) == list(spectra.index) and list(out.columns) == list(spectra.columns)
    # every dye keeps exactly its main channel at 1; the other entries are scaled relative to it
    for dye, main in zip(synthetic.STAINS, synthetic.CHANNELS_FLUORESCENCE):
        assert out.loc[main, dye] == pytest.approx(1.)
        expected = spectra[dye] * factors / factors[main]
        np.testing.assert_allclose(out[dye], expected, rtol=1e-12)


def test_scale_spectra_spectral_normalises_the_maximum():
    spectra = _spectra_channel_dye()
    factors = pd.Series([1., 10., 1.], index=synthetic.CHANNELS_FLUORESCENCE)
    out = scale_spectra_wfactors(spectra, factors, spectral=True)
    np.testing.assert_allclose(out.max(axis=0), 1.)
    scaled = spectra * factors.to_numpy()[:, None]
    np.testing.assert_allclose(out, scaled / scaled.max(axis=0), rtol=1e-12)


def test_scale_spectra_ignores_extra_factors_and_aligns_by_name():
    spectra = _spectra_channel_dye()
    factors = pd.Series([4., 2., 1., 99.], index=["FL3-A", "FL2-A", "FL1-A", "FSC-A"])  # other order + extra channel
    out = scale_spectra_wfactors(spectra, factors, spectral=True)
    ref = scale_spectra_wfactors(spectra, factors.loc[synthetic.CHANNELS_FLUORESCENCE], spectral=True)
    pd.testing.assert_frame_equal(out, ref)


def test_scale_spectra_missing_factor_raises():
    with pytest.raises(CalibrationModuleException, match="missing factors"):
        scale_spectra_wfactors(_spectra_channel_dye(), pd.Series([1., 1.], index=["FL1-A", "FL2-A"]))


def test_scale_spectra_nan_factor_raises():
    factors = pd.Series([1., np.nan, 1.], index=synthetic.CHANNELS_FLUORESCENCE)
    with pytest.raises(CalibrationModuleException, match="nan"):
        scale_spectra_wfactors(_spectra_channel_dye(), factors)


def test_scale_spectra_conventional_fallback_warns_R2_C1():
    """R2-C1: conventional spectra without exactly one entry 1.0 warn (SpilloverWarning) and fall back to the spectral
    normalisation instead of raising NameError."""
    spectra = _spectra_channel_dye()
    spectra["FL1"] = [.9, .3, .05]           # no entry 1 -> invalid
    spectra["FL2"] = [1., 1., .4]            # two entries 1 -> ambiguous
    factors = pd.Series([2., 1., 1.], index=synthetic.CHANNELS_FLUORESCENCE)
    with pytest.warns(SpilloverWarning) as rec:
        out = scale_spectra_wfactors(spectra, factors, spectral=False)
    messages = " ".join(str(w.message) for w in rec)
    assert "ambiguous" in messages and "invalid" in messages
    for dye in ["FL1", "FL2"]:
        assert out[dye].max() == pytest.approx(1.)
    assert out.loc["FL3-A", "FL3"] == pytest.approx(1.)  # the valid dye keeps the conventional normalisation


# ---------------------------------------------------------------- extract_logfactor_laserfactor
def _logfactors(rng, n=30, component=(.8, 1., 1.2, 1.), noise=.01):
    a = rng.normal(size=n)
    lf = pd.DataFrame(5 + a[:, None] * np.asarray(component)[None] + noise * rng.normal(size=(n, len(component))),
                      columns=list("abcd"), index=[f"b{i}" for i in range(n)])
    return lf, a


def test_logfactor_fixed_components_is_channel_mean(rng):
    lf, _ = _logfactors(rng)
    common, res = extract_logfactor_laserfactor(lf, fixed_components=True)
    pd.testing.assert_series_equal(common, lf.mean(axis=1), check_names=False)
    pd.testing.assert_frame_equal(res, lf.sub(lf.mean(axis=1), axis=0))
    np.testing.assert_allclose(res.sum(axis=1), 0., atol=1e-10)


def test_logfactor_nan_batches_are_left_out(rng):
    lf, _ = _logfactors(rng, n=10)
    lf.iloc[3, 1] = np.nan
    for fixed in [True, False]:
        common, res = extract_logfactor_laserfactor(lf, fixed_components=fixed)
        assert np.isnan(common.iloc[3]) and res.iloc[3].isna().all()
        assert common.drop(lf.index[3]).notna().all()
        assert list(res.index) == list(lf.index) and list(res.columns) == list(lf.columns)


def test_logfactor_pca_decomposition_R2_D1(rng):
    """R2-D1: the PCA path reconstructs logfactors = common * component + residuals (offset included), its common
    factor is oriented like the data (positive correlation), and a constant component equals the fixed path."""
    lf, a = _logfactors(rng)
    common, res = extract_logfactor_laserfactor(lf, fixed_components=False)
    fixed_common, _ = extract_logfactor_laserfactor(lf, fixed_components=True)
    assert np.corrcoef(common, a)[0, 1] > .99
    assert abs(common.mean() - fixed_common.mean()) < .05            # offset included, same scale as the mean
    # residuals = logfactors - common * component, with the component rescaled to mean 1 (the data's .8, 1, 1.2, 1);
    # the rank-1 structure is explained, what is left per channel is the offset not along the component (constant)
    component = np.array([.8, 1., 1.2, 1.])
    np.testing.assert_allclose(res, lf - common.to_numpy()[:, None] * component[None], atol=.05)
    assert res.std(axis=0).max() < .05
    # with the sign flipped in the data, the common factor follows (no arbitrary PCA sign)
    common_neg, _ = extract_logfactor_laserfactor(-lf, fixed_components=False)
    np.testing.assert_allclose(common_neg, -common, atol=1e-8)
    # constant component: identical to the fixed path
    const = pd.DataFrame(5 + a[:, None] * np.ones((1, 4)), columns=list("abcd"))
    c1, r1 = extract_logfactor_laserfactor(const, True)
    c2, r2 = extract_logfactor_laserfactor(const, False)
    np.testing.assert_allclose(c1, c2, atol=1e-6)
    np.testing.assert_allclose(r1, r2, atol=1e-6)


def test_logfactor_pca_without_common_direction_raises(rng):
    a = rng.normal(size=40)
    lf = pd.DataFrame(a[:, None] * np.array([[1., -1.]]) + 1e-3 * rng.normal(size=(40, 2)), columns=["x", "y"])
    with pytest.raises(CalibrationModuleException, match="common direction"):
        extract_logfactor_laserfactor(lf, fixed_components=False)


def test_logfactor_bylaser_groups_channels_by_laser(rng):
    lf, _ = _logfactors(rng, n=8)
    lf.columns = ["r1", "b1", "r2", "b2"]
    key = np.array(["Red", "Blue", "Red", "Blue"])
    res, laser = extract_logfactor_laserfactor_bylaser(lf, key)
    assert list(laser.columns) == ["Blue", "Red"] and list(laser.index) == list(lf.index)
    assert list(res.columns) == ["b1", "b2", "r1", "r2"]               # grouped by sorted laser
    pd.testing.assert_series_equal(laser["Red"], lf[["r1", "r2"]].mean(axis=1), check_names=False)
    pd.testing.assert_frame_equal(res[["b1", "b2"]], lf[["b1", "b2"]].sub(lf[["b1", "b2"]].mean(axis=1), axis=0))


def test_logfactor_bylaser_all_nan_channel_and_laser(rng):
    lf, _ = _logfactors(rng, n=6)
    lf.columns = ["r1", "b1", "r2", "v1"]
    lf["r2"] = np.nan       # NaN in all batches: stays NaN, the laser uses r1 only
    lf["v1"] = np.nan       # the only channel of laser V: laser factor NaN
    key = pd.Series(["R", "B", "R", "V"], index=lf.columns)
    res, laser = extract_logfactor_laserfactor_bylaser(lf, key.to_numpy())
    assert res["r2"].isna().all() and res["v1"].isna().all()
    pd.testing.assert_series_equal(laser["R"], lf["r1"], check_names=False)
    assert laser["V"].isna().all()


# ---------------------------------------------------------------- reorder_peaks
def _gmm_adata(n_peak=3, n=30):
    adata = synthetic.make_adata(np.zeros((n, 3)), synthetic.CHANNELS_FLUORESCENCE, layer="raw")
    adata.obs["GMM_label"] = np.arange(n) % n_peak
    means = np.arange(n_peak)[:, None] * 10. + np.arange(3)[None]
    adata.uns.update(GMM_means=means.copy(), GMM_asinh_means=means + .5,
                     GMM_asinh_covariances=np.stack([np.eye(3) * (k + 1) for k in range(n_peak)]),
                     GMM_asinh_stds=means + .25)
    return adata, means


def test_reorder_peaks_reverses_labels_and_rows():
    adata, means = _gmm_adata()
    old_labels = adata.obs["GMM_label"].to_numpy().copy()
    reorder_peaks(adata, pd.Series([2, 1, 0], index=[0, 1, 2]))
    np.testing.assert_array_equal(adata.obs["GMM_label"], 2 - old_labels)
    np.testing.assert_array_equal(adata.uns["GMM_means"], means[::-1])
    np.testing.assert_array_equal(adata.uns["GMM_asinh_means"], means[::-1] + .5)
    np.testing.assert_array_equal(adata.uns["GMM_asinh_stds"], means[::-1] + .25)
    assert adata.uns["GMM_asinh_covariances"][0, 0, 0] == 3.


def test_reorder_peaks_general_permutation_keeps_row_label_correspondence():
    adata, means = _gmm_adata(n_peak=4, n=40)
    old_labels = adata.obs["GMM_label"].to_numpy().copy()
    new_order = pd.Series([2, 0, 3, 1], index=[0, 1, 2, 3])        # old label -> new label
    reorder_peaks(adata, new_order)
    for old, new in new_order.items():
        np.testing.assert_array_equal(adata.uns["GMM_means"][new], means[old])
        assert (adata.obs["GMM_label"].to_numpy()[old_labels == old] == new).all()


def test_reorder_peaks_wrong_length_raises():
    adata, _ = _gmm_adata()
    with pytest.raises(ValueError):
        reorder_peaks(adata, pd.Series([1, 0], index=[0, 1]))
