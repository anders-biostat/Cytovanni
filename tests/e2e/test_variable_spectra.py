"""End-to-end: the variable-spectra tutorials (rainbow calibration -> calibrated samples and spectra -> dye PCA and
panel -> panel-embedding reference model) on two dates of the example data (`pipeline.VARIABLE_DATES`), samples
measured on both dates only."""
import os

import numpy as np
import pandas as pd
import pytest

import pipeline
from conftest import run_stage
from cytovanni.io import read_rainbow_h5ad
from cytovanni.rainbow import RainbowCalibrator


@pytest.fixture(scope="session")
def meta(example_data):
    return pipeline.variable_metadata(example_data)


@pytest.fixture(scope="session")
def prep(request, example_data, stage_cache):
    return run_stage(request, stage_cache, "variable/rainbow_prep", pipeline.variable_rainbow_prep, example_data)


@pytest.fixture(scope="session")
def fit(request, prep, stage_cache):
    return run_stage(request, stage_cache, "variable/rainbow_fit", pipeline.variable_rainbow_fit, prep, slow=True)


@pytest.fixture(scope="session")
def cal(request, example_data, fit, stage_cache):
    return run_stage(request, stage_cache, "variable/calibrate", pipeline.variable_calibrate, example_data, fit, slow=True)


@pytest.fixture(scope="session")
def dye(request, example_data, cal, stage_cache):
    return run_stage(request, stage_cache, "variable/dye", pipeline.variable_dye, example_data, cal)


@pytest.fixture(scope="session")
def model(request, cal, dye, stage_cache):
    return run_stage(request, stage_cache, "variable/model", pipeline.variable_model, cal, dye, slow=True)


def test_subset_selection(meta):
    smp, ref, rbw, runs = meta
    assert set(smp["m_date"]) == set(pipeline.VARIABLE_DATES) == set(ref["m_date"])
    for uid, g in smp.groupby("ovlp_uid"):
        assert set(g["m_date"]) == set(pipeline.VARIABLE_DATES), uid
    assert set(runs["rainbow_batch"]) <= set(rbw["batch_rb"])
    assert set(smp["batch_rb"]) | set(ref["batch_rb"]) <= set(rbw["batch_rb"])
    assert smp["is_train"].any() and (~smp["is_train"]).any()


def test_rainbow_prep(prep, meta):
    rbw = meta[2]
    files = [f for f in os.listdir(prep) if f.endswith(".h5ad")]
    assert len(files) == len(rbw)
    for f in files:
        ad = read_rainbow_h5ad(os.path.join(prep, f))
        n = ad.uns["N_rainbow_peak"]
        shares = np.bincount(ad.obs["GMM_label"].astype(int), minlength=n) / ad.shape[0]
        assert shares.min() > .01, (f, shares)


@pytest.mark.slow
def test_rainbow_fit(fit, meta):
    f = pd.read_csv(os.path.join(fit, "factors.csv"), index_col=0)
    assert set(f.index) == set(meta[2]["batch_rb"])
    from cytovanni.utils import CytometerConfiguration_A3
    fl = f[CytometerConfiguration_A3().channels_fluorescence]
    ok = fl.notna() & np.isfinite(fl)
    assert ok.to_numpy().mean() > .95
    assert (fl[ok] > 0).all().all()
    loss = pd.read_csv(os.path.join(fit, "losshistory.csv"), index_col=0).iloc[:, 0].to_numpy()
    assert loss[-500:].mean() < loss[:500].mean()


@pytest.mark.slow
def test_rainbow_fit_agrees_with_full_tutorial_fit(fit, reference_outputs):
    """Log factor differences between the batches of the subset agree with the tutorial's fit on all measurements:
    after centring both over the common batches, 95 % of the entries within 3 %."""
    if reference_outputs is None:
        pytest.skip("CYTOVANNI_TEST_REFERENCE not set")
    path = os.path.join(reference_outputs, "integration_variable-spectra", "rainbow", "RBInt.pt")
    try:
        full = RainbowCalibrator.from_saved(path).get_calibration_factors_all()
    except Exception as e:  # saved with an older package version
        pytest.skip(f"cannot load the tutorial fit {path}: {type(e).__name__}: {e}")
    ours = pd.read_csv(os.path.join(fit, "factors.csv"), index_col=0)
    idx = ours.index.intersection(full.index)
    cols = ours.columns.intersection(full.columns)
    a, b = np.log(ours.loc[idx, cols]), np.log(full.loc[idx, cols])
    d = ((a - a.mean(0)) - (b - b.mean(0))).abs().to_numpy()
    d = d[np.isfinite(d)]
    assert np.quantile(d, .95) < np.log(1.03), np.quantile(d, [.5, .95, 1.])


@pytest.mark.slow
def test_calibrated_outputs(cal, meta):
    smp, ref, rbw, runs = meta
    for fp in list(smp["filepath"]) + list(ref["filepath"]):
        assert os.path.exists(os.path.join(cal, "calibrated_fcs", fp)), fp
    for r in runs["run"]:
        sp = pd.read_csv(os.path.join(cal, "single_stain", r, "spectra_calibrated.csv"), index_col=0)
        assert np.isfinite(sp.to_numpy()).all(), r


@pytest.mark.slow
def test_dye_panel(dye):
    from cytovanni.spill import PanelConfiguration
    pc = PanelConfiguration.from_saved(os.path.join(dye, "panel_HD7_TCell.pt"))
    assert pc.has_PCA_data and pc.total_components > 0
    assert pc.pca_components.shape[1] == len(pc.channels)
    assert np.isfinite(pc.pca_components.to_numpy()).all()


@pytest.mark.slow
def test_model_improves_test_aliquots(cal, dye, model):
    """On the held-out samples (is_train False), the fitted model brings the aliquots of the two dates closer than
    the initial model (arcsinh quantile differences), and the anchor batch keeps factor 1."""
    from cytovanni.ref import OverlapFitModel
    adatas = [ad for ad in pipeline.variable_load_samples(cal) if not ad.uns["is_train"]]
    for name, key in [("model_init.pt", "unmx_init"), ("model.pt", "unmx_standardised")]:
        sfm = OverlapFitModel.from_saved(os.path.join(model, name)).to(pipeline.device())
        for ad in adatas:
            sfm.add_ad_standardised(ad, key_layer="raw", addkey=key)
    before = pipeline.aliquot_mismatch(adatas, "unmx_init")
    after = pipeline.aliquot_mismatch(adatas, "unmx_standardised")
    pd.DataFrame({"init": before, "fitted": after}).to_csv(os.path.join(model, "aliquot_mismatch_test.csv"))
    assert after.mean() < before.mean(), pd.DataFrame({"init": before, "fitted": after})
    ss = pd.read_csv(os.path.join(model, "stainscale.csv"), index_col=0)
    np.testing.assert_allclose(ss.loc[pipeline.VARIABLE_ANCHORS["batch"][0]], 1., atol=1e-5)
