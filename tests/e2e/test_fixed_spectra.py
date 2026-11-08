"""End-to-end: the fixed-spectra tutorial (rainbow calibration -> calibrated samples -> scaling reference model) on
two batches of the example data (`pipeline.FIXED_BATCHES`)."""
import os

import numpy as np
import pandas as pd
import pytest

import pipeline
from conftest import run_stage
from cytovanni.io import read_rainbow_h5ad, readfcs_sample
from cytovanni.rainbow import RainbowCalibrator
from cytovanni.utils import CytometerConfiguration_A3small


@pytest.fixture(scope="session")
def prep(request, example_data, stage_cache):
    return run_stage(request, stage_cache, "fixed/rainbow_prep", pipeline.fixed_rainbow_prep, example_data)


@pytest.fixture(scope="session")
def fit(request, prep, stage_cache):
    return run_stage(request, stage_cache, "fixed/rainbow_fit", pipeline.fixed_rainbow_fit, prep, slow=True)


@pytest.fixture(scope="session")
def cal(request, example_data, fit, stage_cache):
    return run_stage(request, stage_cache, "fixed/calibrate", pipeline.fixed_calibrate, example_data, fit)


@pytest.fixture(scope="session")
def ref(request, cal, stage_cache):
    return run_stage(request, stage_cache, "fixed/ref", pipeline.fixed_ref, cal, slow=True)


def factors(fit):
    return pd.read_csv(os.path.join(fit, "factors.csv"), index_col=0)


# ------------------------------------------------------------------ rainbow preparation
def test_rainbow_prep_all_peaks_found(prep):
    files = sorted(f for f in os.listdir(prep) if f.endswith(".h5ad"))
    assert len(files) == len(pipeline.FIXED_BATCHES)
    for f in files:
        ad = read_rainbow_h5ad(os.path.join(prep, f))
        assert ad.uns["N_rainbow_peak"] == 8 and ad.uns["GMM_means"].shape[0] == 8
        shares = np.bincount(ad.obs["GMM_label"].astype(int), minlength=8) / ad.shape[0]
        assert shares.min() > .03, shares            # no collapsed peak
        assert ad.shape[0] > 5000
    assert any(f.endswith(".jpg") for f in os.listdir(prep))


# ------------------------------------------------------------------ rainbow fit
@pytest.mark.slow
def test_rainbow_fit_factors(fit):
    f = factors(fit)
    cfg = CytometerConfiguration_A3small()
    fl = f[cfg.channels_fluorescence]
    assert list(f.index) == pipeline.FIXED_BATCHES
    assert np.isfinite(fl.to_numpy()).all() and (fl.to_numpy() > 0).all()
    # geometric mean 1 over the batches of the first fit, and the batches differ by at most a few tens of percent
    np.testing.assert_allclose(np.log(fl).mean(0), 0, atol=1e-5)
    assert np.abs(np.log(fl).to_numpy()).max() < np.log(1.5)
    loss = pd.read_csv(os.path.join(fit, "losshistory.csv"), index_col=0).iloc[:, 0].to_numpy()
    assert loss[-500:].mean() < loss[:500].mean()


@pytest.mark.slow
def test_rainbow_fit_agrees_with_full_tutorial_fit(fit, reference_outputs):
    """The factor ratio between the two batches agrees with the tutorial's fit on all six batches (within 2 %)."""
    if reference_outputs is None:
        pytest.skip("CYTOVANNI_TEST_REFERENCE not set")
    path = os.path.join(reference_outputs, "integration_fixed-spectra", "rainbow", "RBInt.pt")
    try:
        full = RainbowCalibrator.from_saved(path).get_calibration_factors_all()
    except Exception as e:  # saved with an older package version
        pytest.skip(f"cannot load the tutorial fit {path}: {type(e).__name__}: {e}")
    cfg = CytometerConfiguration_A3small()
    ch = cfg.channels_fluorescence
    b0, b1 = pipeline.FIXED_BATCHES
    ours = np.log(factors(fit).loc[b1, ch] / factors(fit).loc[b0, ch])
    theirs = np.log(full.loc[b1, ch] / full.loc[b0, ch])
    d = (ours - theirs).abs()
    assert d.max() < np.log(1.02), d.sort_values().tail()


# ------------------------------------------------------------------ calibrated samples
@pytest.mark.slow
def test_calibrated_files_are_raw_times_factor(example_data, fit, cal):
    cfg = CytometerConfiguration_A3small()
    raw_dir = os.path.join(example_data, pipeline.FIXED_DIR)
    f = factors(fit)
    for b in pipeline.FIXED_BATCHES:
        gated = readfcs_sample(os.path.join(raw_dir, f"RF_{b}.fcs"), cfg, gates=pipeline.gates_minmax(cfg)).to_df(layer="raw")
        calib = readfcs_sample(os.path.join(cal, f"RF_{b}.fcs"), cfg).to_df(layer="raw")
        assert calib.shape == gated.shape
        ch = cfg.channels_fluorescence
        np.testing.assert_allclose(calib[ch].to_numpy(), gated[ch].to_numpy() * f.loc[b, ch].to_numpy()[None],
                                   rtol=1e-5, atol=1e-2)
    spectra = pd.read_csv(os.path.join(cal, "panel_spectra_calibrated.csv"), index_col=0)
    assert np.isfinite(spectra.to_numpy()).all()


# ------------------------------------------------------------------ reference model
@pytest.mark.slow
def test_ref_model_brings_aliquots_together(cal, ref):
    """After standardisation, the arcsinh quantiles of the two aliquots differ less than before, for the normalised
    markers; the anchor batch keeps factor 1."""
    from cytovanni.ref import OverlapFitModel
    from cytovanni.utils import apply_arcsinh
    sf = pd.read_csv(os.path.join(ref, "normalization_factors.csv"), index_col=0)
    np.testing.assert_allclose(sf.loc[f"RF_{pipeline.FIXED_ANCHOR}"], 1., atol=1e-6)
    sfm = OverlapFitModel.from_saved(os.path.join(ref, "standardisation_model.pt"))
    ads, _ = pipeline.fixed_read_calibrated(cal)
    for ad in ads:
        sfm.add_ad_standardised(ad, key_layer="raw", addkey="unmx_standardised")
    m = pipeline.FIXED_MARKERS_NORMALIZE
    def mismatch(key):
        q = [apply_arcsinh(ad.obsm[key][m], 1500).quantile([.5, .9, .99]) for ad in ads]
        return (q[0] - q[1]).abs().mean(0)
    before, after = mismatch("unmx"), mismatch("unmx_standardised")
    assert after.mean() < .7 * before.mean(), pd.DataFrame({"before": before, "after": after})


@pytest.mark.slow
def test_ref_model_agrees_with_full_tutorial_fit(ref, reference_outputs):
    """Stain factors of the last batch relative to the anchor agree with the tutorial's fit on six batches (10 %)."""
    if reference_outputs is None:
        pytest.skip("CYTOVANNI_TEST_REFERENCE not set")
    full = pd.read_csv(os.path.join(reference_outputs, "integration_fixed-spectra", "ref", "normalization_factors.csv"),
                       index_col=0)
    ours = pd.read_csv(os.path.join(ref, "normalization_factors.csv"), index_col=0)
    b = f"RF_{pipeline.FIXED_BATCHES[1]}"
    m = pipeline.FIXED_MARKERS_NORMALIZE
    d = np.abs(np.log(ours.loc[b, m] / full.loc[b, m]))
    assert d.max() < np.log(1.1), d.sort_values().tail()
