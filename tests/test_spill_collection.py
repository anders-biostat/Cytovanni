"""Tests of cytovanni.spill.collection: SpilloverMeasurement (one run) and SpilloverCollection (many runs, PCA per stain)."""
import os
import shutil
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.exceptions import NegativeSpilloverWarning, SpilloverException, SpilloverWarning
from cytovanni.rainbow import scale_spectra_wfactors
from cytovanni.spill import (LinearSingleStainSpectrumFit, OutliersafeLinearSingleStainSpectrumFit,
                             PosNegSingleStainSpectrumFit, SpilloverCollection, SpilloverMeasurement,
                             collect_bead_controls_regularflow, collect_single_fits, make_AutoSpill_fcs_config,
                             PanelConfiguration)
from cytovanni.spill import collection as coll_mod
from cytovanni.io import read_bead_h5ad

CH = synthetic.CHANNELS_FLUORESCENCE
FACTOR = pd.Series([1.3, 1.3, 2., 1., .5], index=["FSC-A", "FSC-H"] + CH)  # calibration factor incl. scatter channels


class StubCalibrator:
    """Stand-in for cytovanni.rainbow.RainbowCalibrator: channels and a fixed calibration factor; records its calls."""
    def __init__(self, factor=FACTOR):
        self.rfim = SimpleNamespace(channels=CH)
        self.factor = factor
        self.calls = []

    def get_calibration_factor(self, rainbow_batch=None, adata=None):
        self.calls.append((rainbow_batch, adata))
        return self.factor


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- one measurement from the package's own pipeline
@pytest.fixture(scope="module")
def custom_folder(single_stain_folder, tmp_path_factory):
    """collect_bead_controls_regularflow + linear, linearos, posneg fits + collect_single_fits on the synthetic stains."""
    folder = str(tmp_path_factory.mktemp("custom") / "run1")
    quiet(collect_bead_controls_regularflow, single_stain_folder, synthetic.make_config(), folder, verbose=False)
    for f in sorted(os.listdir(os.path.join(folder, "data"))):
        adata = read_bead_h5ad(os.path.join(folder, "data", f))
        for cls in [LinearSingleStainSpectrumFit, OutliersafeLinearSingleStainSpectrumFit, PosNegSingleStainSpectrumFit]:
            quiet(cls, adata).write(folder)
    for m in ["linear", "linearos", "posneg"]:
        collect_single_fits(folder, f"fit_{m}", f"spectra_{m}.csv")
    return folder


def copy_folder(src, tmp_path):
    dst = str(tmp_path / "run")
    shutil.copytree(src, dst)
    return dst


def test_init_needs_rbint_or_channels():
    with pytest.raises(ValueError, match="rbint or channels"):
        SpilloverMeasurement()
    spm = SpilloverMeasurement(channels=CH)
    assert spm.channels == CH and spm.spectra == {} and spm.spectra_calibrated == {}
    assert not hasattr(spm, "calibration_factor")


def test_from_custom_spectra_recover_truth(custom_folder):
    rb = StubCalibrator()
    spm = quiet(SpilloverMeasurement.from_custom, rb, custom_folder)
    assert set(spm.spectra) == {"linear", "linearos", "posneg"}
    for key, sp in spm.spectra.items():
        assert list(sp.index) == CH and list(sp.columns) == synthetic.STAINS  # (channel, dye), dyes sorted
        assert np.allclose(sp.to_numpy(), synthetic.SPECTRA.T.to_numpy(), atol=.02), key
    assert list(spm.panel_meta.index) == synthetic.STAINS
    assert "datafile" not in spm.panel_meta.columns
    assert spm.panel_meta["antigen"].tolist() == synthetic.MARKERS


def test_from_custom_calibration(custom_folder):
    """Calibrated = scale_spectra_wfactors(raw, factor); the rainbow batch is inferred from the smallest .h5ad."""
    rb = StubCalibrator()
    spm = quiet(SpilloverMeasurement.from_custom, rb, custom_folder)
    (batch, adata), = rb.calls
    assert batch is None and adata is not None and adata.uns["dye"] in synthetic.STAINS
    pd.testing.assert_series_equal(spm.calibration_factor, FACTOR)
    for key in spm.spectra:
        pd.testing.assert_frame_equal(spm.spectra_calibrated[key], scale_spectra_wfactors(spm.spectra[key], FACTOR, False))
    # conventional flow: calibrated spectrum ~ slope_c * f_c / f_main, still 1 in the main channel
    f = FACTOR[CH].to_numpy()
    exp = synthetic.SPECTRA.to_numpy() * f[None] / f[:, None]
    assert np.allclose(spm.spectra_calibrated["linear"].to_numpy(), exp.T, atol=.05)
    assert np.allclose(np.diag(spm.spectra_calibrated["linear"].to_numpy()), 1.)


def test_from_custom_explicit_rainbow_batch(custom_folder):
    rb = StubCalibrator()
    quiet(SpilloverMeasurement.from_custom, rb, custom_folder, rainbow_batch="A")
    assert rb.calls == [("A", None)]


def test_from_custom_spectral_normalisation(custom_folder):
    spm = quiet(SpilloverMeasurement.from_custom, StubCalibrator(), custom_folder, is_spectral=True)
    assert np.allclose(spm.spectra_calibrated["linear"].max(axis=0), 1.)


def test_from_custom_inconsistent_settings(custom_folder, tmp_path):
    """R2-C3: abort_inconsistent=False warns with SpilloverWarning (was TypeError: exception class as warning category)."""
    folder = copy_folder(custom_folder, tmp_path)
    meta = pd.read_csv(os.path.join(folder, "panel_meta.csv"), index_col=0)
    meta.loc[2, "set_hash"] = "other"
    meta.to_csv(os.path.join(folder, "panel_meta.csv"))
    with pytest.raises(SpilloverException, match="inconsistent settings"):
        quiet(SpilloverMeasurement.from_custom, StubCalibrator(), folder)
    with pytest.warns(SpilloverWarning, match="inconsistent settings"):
        spm = SpilloverMeasurement.from_custom(StubCalibrator(), folder, abort_inconsistent=False)
    assert "linear" in spm.spectra


def test_from_custom_missing_metadata_raises(custom_folder, tmp_path):
    folder = copy_folder(custom_folder, tmp_path)
    meta = pd.read_csv(os.path.join(folder, "panel_meta.csv"), index_col=0)
    meta.drop(index=1).to_csv(os.path.join(folder, "panel_meta.csv"))
    with pytest.raises(SpilloverException, match="Missing metadata"):
        quiet(SpilloverMeasurement.from_custom, StubCalibrator(), folder)


def test_from_custom_no_spectra_warns(custom_folder, tmp_path):
    folder = copy_folder(custom_folder, tmp_path)
    for m in ["linear", "linearos", "posneg"]:
        os.remove(os.path.join(folder, f"spectra_{m}.csv"))
    with pytest.warns(SpilloverWarning, match="custom fits"):
        spm = SpilloverMeasurement.from_custom(StubCalibrator(), folder)
    assert spm.spectra == {}


def test_from_custom_clips_negative(custom_folder, tmp_path):
    folder = copy_folder(custom_folder, tmp_path)
    sp = pd.read_csv(os.path.join(folder, "spectra_linear.csv"), index_col=0)
    sp.iloc[2, 0] = -.01
    sp.to_csv(os.path.join(folder, "spectra_linear.csv"))
    with pytest.warns(NegativeSpilloverWarning, match="1 negative"):
        spm = SpilloverMeasurement.from_custom(StubCalibrator(), folder)
    assert spm.spectra["linear"].iloc[2, 0] == 0. and spm.spectra_calibrated["linear"].iloc[2, 0] == 0.
    spm = quiet(SpilloverMeasurement.from_custom, StubCalibrator(), folder, clip_negative=False)
    assert spm.spectra["linear"].iloc[2, 0] == -.01


def write_autospill(folder, single_stain_folder, transposed_spectra, pad=False):
    """AutoSpill-like result folder: fcs_control.csv and table_spillover/autospill_spillover.csv (dye rows, '-A')."""
    os.makedirs(os.path.join(folder, "table_spillover"), exist_ok=True)
    cfg = SimpleNamespace(channels_fluorescence=CH + (["FL4-A"] if pad else []))
    quiet(make_AutoSpill_fcs_config, single_stain_folder, cfg).to_csv(os.path.join(folder, "fcs_control.csv"))
    transposed_spectra.to_csv(os.path.join(folder, "table_spillover", "autospill_spillover.csv"))
    return folder


def test_from_AutoSpill(single_stain_folder, tmp_path):
    asp = synthetic.SPECTRA.copy()
    asp.index = [s + "-A" for s in asp.index]  # AutoSpill: (dye with -A, channel)
    asp.loc["FL4-A"] = [.1, .1, .1]  # a padded channel's fit, loaded like the others
    folder = write_autospill(str(tmp_path / "asp"), single_stain_folder, asp, pad=True)
    rb = StubCalibrator()
    spm = SpilloverMeasurement.from_AutoSpill(rb, folder, rainbow_batch="A")
    assert rb.calls == [("A", None)]
    sp = spm.spectra["autospill"]
    assert list(sp.index) == CH and list(sp.columns) == ["FL1", "FL2", "FL3", "FL4"]
    assert np.allclose(sp[synthetic.STAINS].to_numpy(), synthetic.SPECTRA.T.to_numpy())
    assert spm.panel_meta.loc["FL4", "antigen"] == "_dummy_pad"
    assert "filename" not in spm.panel_meta.columns and "wavelength" not in spm.panel_meta.columns
    assert set(spm.spectra_calibrated) == {"autospill"}


def test_from_AutoSpill_missing_tables_warns(single_stain_folder, tmp_path):
    asp = synthetic.SPECTRA.copy(); asp.index = [s + "-A" for s in asp.index]
    folder = write_autospill(str(tmp_path / "asp"), single_stain_folder, asp)
    os.remove(os.path.join(folder, "table_spillover", "autospill_spillover.csv"))
    with pytest.warns(SpilloverWarning, match="AutoSpill fits"):
        SpilloverMeasurement.from_AutoSpill(StubCalibrator(), folder, rainbow_batch="A")


def test_from_custom_with_autospill(custom_folder, single_stain_folder, tmp_path):
    asp = synthetic.SPECTRA.copy(); asp.index = [s + "-A" for s in asp.index]
    folder = write_autospill(str(tmp_path / "asp"), single_stain_folder, asp)
    shutil.copy(os.path.join(folder, "table_spillover", "autospill_spillover.csv"),
                os.path.join(folder, "table_spillover", "posnegpop_spillover.csv"))
    spm = quiet(SpilloverMeasurement.from_custom, StubCalibrator(), custom_folder, asp_folder=folder)
    assert set(spm.spectra) == {"linear", "linearos", "posneg", "autospill", "autospill_posneg"}


def test_read_matrix_limits_and_orders_channels(tmp_path):
    m = pd.DataFrame([[1., .2], [.3, 1.], [.5, .5]], index=["FL2-A", "FL1-A", "X-A"], columns=["b", "a"])
    m.to_csv(tmp_path / "m.csv")
    out = coll_mod._read_matrix_custom(str(tmp_path / "m.csv"), CH)
    assert list(out.index) == CH and list(out.columns) == ["a", "b"]
    assert out.loc["FL1-A", "a"] == 1. and out.loc["FL3-A"].isna().all()
    m.T.rename(index={"a": "a-A"}).to_csv(tmp_path / "t.csv")
    out = coll_mod._read_matrix_autospill(str(tmp_path / "t.csv"), CH)
    assert list(out.columns) == ["a", "b"] and out.loc["FL1-A", "a"] == 1.


# ---------------------------------------------------------------- from_csv and in-place helpers
def write_run(folder, name, spectra, meta):
    """Write a (channel, dye) spectra csv and a dye metadata csv; return SpilloverMeasurement.from_csv of them."""
    fs, fm = os.path.join(folder, f"{name}_spectra.csv"), os.path.join(folder, f"{name}_meta.csv")
    spectra.to_csv(fs); meta.to_csv(fm)
    return quiet(SpilloverMeasurement.from_csv, fs, fm)


def default_meta(dyes=synthetic.STAINS, **cols):
    meta = pd.DataFrame({"dye": list(dyes), "fluorophore": [f"F_{d}" for d in dyes], "antigen": ["x"] * len(dyes),
                         "positive_peak": [1e4 * (i + 1) for i in range(len(dyes))]}, index=list(dyes))
    for k, v in cols.items():
        meta[k] = v
    return meta


def test_from_csv(tmp_path):
    sp = synthetic.SPECTRA.T.copy()
    sp.iloc[1, 2] = -.02
    fs, fm = str(tmp_path / "s.csv"), str(tmp_path / "m.csv")
    sp.to_csv(fs); default_meta().to_csv(fm)
    with pytest.warns(NegativeSpilloverWarning):
        spm = SpilloverMeasurement.from_csv(fs, fm)
    assert list(spm.spectra) == ["spectra"] and spm.spectra_calibrated == {}
    assert list(spm.channels) == CH
    assert spm.spectra["spectra"].iloc[1, 2] == 0.
    assert list(spm.panel_meta.index) == synthetic.STAINS


def test_clip_negative_spectra_counts_raw_and_calibrated():
    spm = SpilloverMeasurement(channels=CH)
    spm.spectra["a"] = pd.DataFrame([[-1., 1.], [1., -2.]])
    spm.spectra_calibrated["a"] = pd.DataFrame([[-1., 1.], [1., 1.]])
    with pytest.warns(NegativeSpilloverWarning, match="3 negative entries .* 2 fits"):
        spm.clip_negative_spectra()
    assert (spm.spectra["a"] >= 0).all().all() and (spm.spectra_calibrated["a"] >= 0).all().all()


def test_clip_negative_spectra_silent(no_warnings):
    spm = SpilloverMeasurement(channels=CH)
    spm.spectra["a"] = pd.DataFrame([[-1., 1.]])
    spm.clip_negative_spectra(warn=False)
    assert spm.spectra["a"].iloc[0, 0] == 0.


def test_mask_spectra_fromkey_nan_counts_as_true(tmp_path):
    meta = default_meta(is_bad=[True, np.nan, False])
    spm = write_run(str(tmp_path), "r", synthetic.SPECTRA.T, meta)
    spm.spectra_calibrated["spectra"] = spm.spectra["spectra"].copy()
    spm.mask_spectra_fromkey("is_bad", mask_meta=["positive_peak"])
    for d in [spm.spectra["spectra"], spm.spectra_calibrated["spectra"]]:
        assert d[["FL1", "FL2"]].isna().all().all() and d["FL3"].notna().all()
    assert spm.panel_meta["positive_peak"].isna().tolist() == [True, True, False]


def test_check_missing_metadata(tmp_path):
    spm = write_run(str(tmp_path), "r", synthetic.SPECTRA.T, default_meta())
    spm.check_missing_metadata()
    spm.panel_meta = spm.panel_meta.drop(index="FL2")
    with pytest.raises(SpilloverException, match="FL2"):
        spm.check_missing_metadata()


# ---------------------------------------------------------------- SpilloverCollection
DIRS = {"FL1": np.array([[0, .05, 0], [0, 0, .02]]), "FL2": np.array([[.03, 0, 0], [0, 0, .05]]),
        "FL3": np.array([[.01, 0, 0], [0, .04, 0]])}


def make_runs(folder, n=8, seed=0, scale_raw=True, meta_fn=None):
    """n measurements of the synthetic spectra varied along DIRS (two directions per stain).

    With `scale_raw`, every raw spectrum is multiplied by a random factor (not normalised), as raw fits could be.
    Returns (pandas.Series of SpilloverMeasurement, dict stain -> latent (run, 2)).
    """
    rng = np.random.default_rng(seed)
    runs, latent = {}, {s: rng.normal(size=(n, 2)) for s in synthetic.STAINS}
    for r in range(n):
        sp = synthetic.SPECTRA.copy()
        for s in synthetic.STAINS:
            sp.loc[s] = sp.loc[s] + latent[s][r] @ DIRS[s]
        sp = sp.T * (rng.uniform(.5, 2, 3)[None] if scale_raw else 1.)
        meta = default_meta() if meta_fn is None else meta_fn(r)
        runs[f"run{r}"] = write_run(folder, f"run{r}", sp, meta)
    return pd.Series(runs), latent


@pytest.fixture
def collection(tmp_path):
    runs, latent = make_runs(str(tmp_path))
    c = SpilloverCollection(runs, register_on="dye")
    return c, latent


def test_collection_keys_and_spectra_orientation(collection):
    c, _ = collection
    assert c.available_keys.tolist() == synthetic.STAINS and c.available_keys.index.tolist() == synthetic.STAINS
    assert c["FL2"] is c.single_stains["FL2"]
    s = c["FL2"]
    assert s.keys == ["spectra"]
    assert s.spectra["spectra"].shape == (8, 3) and list(s.spectra["spectra"].columns) == CH
    assert list(s.spectra["spectra"].index) == [f"run{i}" for i in range(8)]
    assert list(s.meta.index) == [f"run{i}" for i in range(8)] and (s.meta["dye"] == "FL2").all()
    assert s.spectra_calibrated["spectra"].shape[0] == 0  # from_csv: no calibrated spectra


def test_collection_register_on_other_column(tmp_path):
    runs, _ = make_runs(str(tmp_path), n=3)
    c = SpilloverCollection(runs, register_on="fluorophore")
    assert c.available_keys.tolist() == ["F_FL1", "F_FL2", "F_FL3"]
    assert c["F_FL1"].dye == "FL1" and c["F_FL1"].spectra["spectra"].shape == (3, 3)


def test_collection_stain_under_other_dye_name_warns(tmp_path):
    """R3b-B8: a stain listed under another dye name in some measurement is reported (not collected there)."""
    def meta_fn(r):
        m = default_meta()
        return m if r != 1 else m.rename(index={"FL1": "FL1x"}).assign(dye=["FL1x", "FL2", "FL3"])
    rng = np.random.default_rng(0)
    runs = {}
    for r in range(3):
        sp = synthetic.SPECTRA.T.copy()
        if r == 1:
            sp = sp.rename(columns={"FL1": "FL1x"})
        runs[f"run{r}"] = write_run(str(tmp_path), f"run{r}", sp, meta_fn(r))
    with pytest.warns(SpilloverWarning, match=r"F_FL1.*run1"):
        c = SpilloverCollection(pd.Series(runs), register_on="fluorophore")
    assert c["F_FL1"].spectra["spectra"].index.tolist() == ["run0", "run2"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SpilloverCollection(pd.Series({k: runs[k] for k in ["run0", "run2"]}), register_on="fluorophore")


def test_normalize_spectra_channel_key_and_default(collection):
    c, _ = collection
    with pytest.warns(UserWarning, match="FL3"):
        c.normalize_spectra(pd.Series({"FL1": "FL2-A", "FL2": "FL2-A"}))
    assert np.allclose(c["FL1"].spectra["spectra"]["FL2-A"], 1.)
    assert np.allclose(c["FL2"].spectra["spectra"]["FL2-A"], 1.)
    assert np.allclose(c["FL3"].spectra["spectra"]["FL3-A"], 1.)  # highest mean channel


def test_normalize_spectra_set_empty_and_default():
    empty = pd.DataFrame(columns=CH, dtype=float)
    assert coll_mod._normalize_spectra_set(empty).shape == (0, 3)
    sp = pd.DataFrame([[2., 4., 1.], [1., 2., 1.]], columns=CH)
    out = coll_mod._normalize_spectra_set(sp)
    assert np.allclose(out["FL2-A"], 1.) and np.allclose(out["FL1-A"], .5)


def test_count_channels(collection):
    c, _ = collection
    quiet(c.normalize_spectra)
    above = c.count_channels_above_cutoff("spectra", cutoff=.1)
    # FL1: FL2-A (.3) > .1; FL2: FL3-A (.4) and FL1-A (.1 +- .03, some runs > .1); FL3: FL2-A (.15)
    assert above.index.tolist() == synthetic.STAINS
    assert above["FL1"] == 1 and above["FL2"] == 2 and above["FL3"] == 1
    rng_ = c.count_channels_range_above_cutoff("spectra", cutoff=.05)
    sp = c["FL2"].spectra["spectra"]
    assert rng_["FL2"] == ((sp.max() - sp.min()) > .05).sum()


def test_pca_recovers_variation_directions(collection):
    """C: after normalisation, the PCA of every stain spans its two true variation directions."""
    c, _ = collection
    quiet(c.normalize_spectra)
    c.add_PCA_fit("spectra", calibrated=False)
    assert c.pca_method == "spectra" and c.pca_calibrated is False
    for s in synthetic.STAINS:
        f = c[s].pcafitter
        assert f.N == 2 and f.name == s
        q, _ = np.linalg.qr(DIRS[s].T)
        assert np.allclose(np.linalg.norm(f.components @ q, axis=1), 1., atol=1e-6)


def test_fit_PCA_without_calibrated_spectra_is_dummy(collection):
    """from_csv runs have no calibrated spectra: fit_PCA(calibrated=True) (its default) gives a dummy fit."""
    c, _ = collection
    quiet(c.normalize_spectra)
    quiet(c["FL1"].fit_PCA, "spectra")
    assert c["FL1"].pcafitter.is_dummy and c["FL1"].pca_calibrated is True


def test_export_pca_structure(collection, tmp_path):
    c, _ = collection
    quiet(c.normalize_spectra)
    c.add_PCA_fit("spectra")
    pd.DataFrame({"stain": synthetic.STAINS, "marker": synthetic.MARKERS, "components": ["01", "0", ""]}).to_csv(tmp_path / "p.csv", index=False)
    pc = PanelConfiguration.from_csv(str(tmp_path / "p.csv"), "p")
    e = c["FL2"].export_pca(pc)
    assert set(e) == {"embedding", "components", "offset", "components_stain", "components_use", "explained_variance",
                      "explained_variance_channelratio", "spectra"}
    assert e["embedding"].shape == (8, 2) and list(e["embedding"].columns) == ["FL2_PC0", "FL2_PC1"]
    assert e["components"].shape == (2, 3) and list(e["components"].columns) == CH
    assert np.allclose(np.linalg.norm(e["components"].to_numpy(), axis=1), 1.)
    assert list(e["offset"].index) == ["FL2"] and e["offset"].shape == (1, 3)
    assert e["components_use"] == ["FL2_PC0"]
    assert (e["components_stain"] == "FL2").all()
    assert e["spectra"].shape == (8, 3)
    assert c["FL3"].export_pca(pc)["components_use"] == []


def test_collection_plots(collection, tmp_path):
    c, _ = collection
    quiet(c.normalize_spectra)
    c.add_PCA_fit("spectra")
    pd.DataFrame({"stain": synthetic.STAINS, "marker": synthetic.MARKERS, "components": ["01", "0", ""]}).to_csv(tmp_path / "p.csv", index=False)
    pc = PanelConfiguration.from_csv(str(tmp_path / "p.csv"), "p")
    c.plot_peak_position()
    c.plot_spectrum("FL1", "spectra", hue="antigen")
    c.plot_spectrum_correlation_single("FL1", "spectra", "FL2-A", "FL3-A", hue="antigen")
    c.plot_total_variance(); c.plot_total_variance(channel_max=True)
    assert c.plot_explained_variance() is not None
    assert c.plot_explained_variance_panel(pc) is not None
    c["FL1"].plot_spill_correlations_pcafit()
    c["FL1"].plot_spill_correlations_pcafit(Npc=1)


def test_plot_spill_correlations_dummy_warns(tmp_path):
    runs, _ = make_runs(str(tmp_path), n=1)
    c = SpilloverCollection(runs)
    quiet(c.normalize_spectra)
    c.add_PCA_fit("spectra")
    assert c["FL1"].pcafitter.is_dummy
    with pytest.warns(SpilloverWarning, match="trivial PCA"):
        assert c["FL1"].plot_spill_correlations_pcafit() is None
