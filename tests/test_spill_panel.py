"""Tests of cytovanni.spill.panel.PanelConfiguration: panel csv, PCA data, embeddings and spectra, save/load."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.spill import PanelConfiguration, SpilloverCollection, SpilloverMeasurement

CH = synthetic.CHANNELS_FLUORESCENCE
DIRS = {"FL1": np.array([[0, .05, 0], [0, 0, .02]]), "FL2": np.array([[.03, 0, 0], [0, 0, .05]]),
        "FL3": np.array([[.01, 0, 0], [0, .04, 0]])}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def make_collection(folder, n=8, seed=0, skip=()):
    """SpilloverCollection of n normalised runs of the synthetic spectra varied along DIRS, PCA fitted.

    `skip`: (run, stain) pairs left out of the measurements (stain not measured in that run).
    """
    rng = np.random.default_rng(seed)
    latent = {s: rng.normal(size=(n, 2)) for s in synthetic.STAINS}
    runs = {}
    for r in range(n):
        sp = synthetic.SPECTRA.copy()
        for s in synthetic.STAINS:
            sp.loc[s] = sp.loc[s] + latent[s][r] @ DIRS[s]
        dyes = [s for s in synthetic.STAINS if (f"run{r}", s) not in skip]
        fs, fm = os.path.join(folder, f"s{r}.csv"), os.path.join(folder, f"m{r}.csv")
        sp.T[dyes].to_csv(fs)
        pd.DataFrame({"dye": dyes}, index=dyes).to_csv(fm)
        runs[f"run{r}"] = quiet(SpilloverMeasurement.from_csv, fs, fm)
    c = SpilloverCollection(pd.Series(runs))
    quiet(c.normalize_spectra)
    c.add_PCA_fit("spectra", calibrated=False)
    return c


def panel_csv(folder, components=("01", "0", ""), name="p.csv", stains=synthetic.STAINS):
    path = os.path.join(folder, name)
    pd.DataFrame({"stain": list(stains), "marker": synthetic.MARKERS[:len(stains)], "components": list(components)}).to_csv(path, index=False)
    return path


@pytest.fixture
def pc_and_collection(tmp_path):
    c = make_collection(str(tmp_path))
    pc = PanelConfiguration.from_csv(panel_csv(str(tmp_path)), "test")
    pc.collect_PCA_data(c)
    return pc, c


# ---------------------------------------------------------------- csv and repr
def test_from_csv_parsing(tmp_path):
    pc = PanelConfiguration.from_csv(panel_csv(str(tmp_path), components=("13", "0", "")), "HD")
    assert pc.name == "HD" and not pc.has_PCA_data
    assert pc.stains.tolist() == synthetic.STAINS and pc.markers.tolist() == synthetic.MARKERS
    assert pc.stain_marker_name.tolist() == [f"{s} | {m}" for s, m in zip(synthetic.STAINS, synthetic.MARKERS)]
    assert [c.tolist() for c in pc.components] == [[1, 3], [0], []]
    assert pc.total_components == 3
    r = repr(pc)
    assert "PanelConfiguration 'HD'" in r and "3 dyes" in r and "3 PC components" in r and "CD2" in r


def test_from_csv_reads_digit_strings_as_str(tmp_path):
    path = os.path.join(str(tmp_path), "p.csv")
    with open(path, "w") as f:
        f.write("stain,marker,components\nFL1,CD1,01\nFL2,CD2,\n")
    pc = PanelConfiguration.from_csv(path, "p")
    assert [c.tolist() for c in pc.components] == [[0, 1], []]


# ---------------------------------------------------------------- collect_PCA_data
def test_collect_PCA_data_attributes(pc_and_collection):
    pc, c = pc_and_collection
    comps = [f"{s}_PC{i}" for s in synthetic.STAINS for i in range(2)]
    assert pc.has_PCA_data and pc.channels == CH
    assert pc.pca_components.index.tolist() == comps and pc.pca_components.columns.tolist() == CH
    assert pc.pca_embedding.shape == (8, 6) and pc.pca_embedding.columns.tolist() == comps
    assert pc.pca_offset.index.tolist() == synthetic.STAINS and pc.pca_offset.shape == (3, 3)
    assert pc.pca_components_stain.tolist() == [s for s in synthetic.STAINS for _ in range(2)]
    assert pc.pca_components_use.tolist() == [True, True, True, False, False, False]
    assert pc.pca_explained_variance.index.tolist() == comps
    assert pc.pca_explained_variance_channelratio.shape == (6, 3)
    assert set(pc.spectra_full) == set(synthetic.STAINS) and pc.spectra_full["FL1"].shape == (8, 3)


def test_collect_PCA_data_standardised_embedding(pc_and_collection):
    """Embeddings in units of their std (ddof 0); components rescaled so embedding @ components is unchanged."""
    pc, c = pc_and_collection
    assert np.allclose(np.nanstd(pc.pca_embedding.to_numpy(), axis=0), 1.)
    f = c["FL2"].pcafitter
    rec = pc.pca_embedding[["FL2_PC0", "FL2_PC1"]].to_numpy() @ pc.pca_components.loc[["FL2_PC0", "FL2_PC1"]].to_numpy()
    assert np.allclose(rec, f.embedding @ f.components)
    assert np.allclose(pc.pca_offset.loc["FL2"].to_numpy(), f.spectra_means[0])


def test_collect_PCA_data_missing_component_raises(tmp_path):
    """R2b-G2-15: a panel that asks for a component that was not fitted raises ValueError naming it."""
    c = make_collection(str(tmp_path), n=2)  # 2 runs -> 1 component per stain
    pc = PanelConfiguration.from_csv(panel_csv(str(tmp_path), components=("01", "", "")), "p")
    with pytest.raises(ValueError, match="FL1_PC1"):
        pc.collect_PCA_data(c)


def test_collect_PCA_data_stain_not_measured_everywhere(tmp_path):
    c = make_collection(str(tmp_path), skip=[("run3", "FL2")])
    pc = PanelConfiguration.from_csv(panel_csv(str(tmp_path)), "p")
    pc.collect_PCA_data(c)
    assert pc.pca_embedding.loc["run3", ["FL2_PC0", "FL2_PC1"]].isna().all()
    assert pc.pca_embedding.loc["run3", "FL1_PC0"] == pc.pca_embedding.loc["run3", "FL1_PC0"]
    assert np.allclose(np.nanstd(pc.pca_embedding.to_numpy(), axis=0), 1.)
    rs = pc.get_run_spectra("run3")
    assert rs.loc["FL2"].isna().all()
    rs = pc.get_run_spectra("run3", pca_impute=True)
    assert np.allclose(rs.loc["FL2"].to_numpy(), pc.pca_offset.loc["FL2"].to_numpy())


# ---------------------------------------------------------------- spectra <-> embedding
def test_get_run_spectra(pc_and_collection):
    pc, c = pc_and_collection
    rs = pc.get_run_spectra("run2")
    assert rs.index.tolist() == synthetic.STAINS and rs.columns.tolist() == CH  # (stain, channel)
    assert np.allclose(rs.loc["FL1"].to_numpy(), c["FL1"].spectra["spectra"].loc["run2"].to_numpy())
    rs = pc.get_run_spectra("run2", label_marker=True)
    assert rs.index.tolist() == pc.stain_marker_name.tolist()
    assert pc.get_run_spectra("nope").isna().all().all()


def test_embedding_roundtrip_all_components_exact(pc_and_collection):
    """C: with all fitted components (full rank), measured spectra are reconstructed exactly from their embedding."""
    pc, _ = pc_and_collection
    for run in ["run0", "run5"]:
        rs = pc.get_run_spectra(run)
        emb = pc.get_spectra_embedding(rs, ensure_complete=True)
        assert emb.index.tolist() == pc.pca_components.index.tolist()
        assert np.allclose(emb.to_numpy(), pc.pca_embedding.loc[run].to_numpy())
        full = pc.get_spectra_from_embedding(emb, only_included_components=False)
        assert full.index.tolist() == synthetic.STAINS and full.columns.tolist() == CH
        assert np.allclose(full.to_numpy(), rs.to_numpy(), atol=1e-10)


def test_spectra_from_embedding_used_components_only(pc_and_collection):
    pc, _ = pc_and_collection
    emb = pc.pca_embedding.loc["run1"]
    part = pc.get_spectra_from_embedding(emb)
    # FL3 uses no components -> its mean spectrum; FL2 uses PC0 only
    assert np.allclose(part.loc["FL3"].to_numpy(), pc.pca_offset.loc["FL3"].to_numpy())
    exp = pc.pca_offset.loc["FL2"] + emb["FL2_PC0"] * pc.pca_components.loc["FL2_PC0"]
    assert np.allclose(part.loc["FL2"].to_numpy(), exp.to_numpy())


def test_unused_nonfinite_embedding_gives_finite_spectra(pc_and_collection):
    """R2d-D4: NaN in an unused component no longer turns the reconstruction NaN."""
    pc, _ = pc_and_collection
    emb = pc.pca_embedding.loc["run1"].copy()
    emb["FL2_PC1"] = np.nan; emb["FL3_PC0"] = np.inf
    out = pc.get_spectra_from_embedding(emb)
    assert np.isfinite(out.to_numpy()).all()


def test_spectra_from_embedding_missing(pc_and_collection):
    pc, _ = pc_and_collection
    emb = pc.pca_embedding.loc["run1"].drop(["FL2_PC0", "FL2_PC1"])
    with pytest.raises(ValueError, match="FL2"):
        pc.get_spectra_from_embedding(emb)
    out = pc.get_spectra_from_embedding(emb, allow_missing=True)
    assert out.loc["FL2"].isna().all() and out.loc["FL1"].notna().all()


def test_spectra_from_embedding_clip_negative(pc_and_collection):
    pc, _ = pc_and_collection
    emb = pc.pca_embedding.loc["run1"].copy()
    emb["FL1_PC1"] = -1e3  # drives FL3-A of FL1 strongly negative
    assert (pc.get_spectra_from_embedding(emb).to_numpy() >= 0).all()
    assert (pc.get_spectra_from_embedding(emb, clip_negative=False).to_numpy() < 0).any()


def test_get_single_spectrum_embedding(pc_and_collection):
    pc, _ = pc_and_collection
    rs = pc.get_run_spectra("run4")
    e = pc.get_single_spectrum_embedding("FL3", rs.loc["FL3"])
    assert e.index.tolist() == ["FL3_PC0", "FL3_PC1"]
    assert np.allclose(e.to_numpy(), pc.pca_embedding.loc["run4", ["FL3_PC0", "FL3_PC1"]].to_numpy())
    # channels are looked up by name (order of the input does not matter)
    e2 = pc.get_single_spectrum_embedding("FL3", rs.loc["FL3"][::-1])
    assert np.allclose(e2.to_numpy(), e.to_numpy())


def test_get_spectra_embedding_errors(pc_and_collection):
    pc, _ = pc_and_collection
    rs = pc.get_run_spectra("run0")
    with pytest.raises(ValueError, match="unique"):
        pc.get_spectra_embedding(pd.concat([rs, rs.iloc[:1]]))
    with pytest.raises(ValueError, match="not part of the panel"):
        pc.get_spectra_embedding(rs.rename(index={"FL1": "XX"}))
    sub = pc.get_spectra_embedding(rs.loc[["FL3", "FL1"]])  # subset, any order -> panel component order
    assert sub.index.tolist() == ["FL1_PC0", "FL1_PC1", "FL3_PC0", "FL3_PC1"]
    with pytest.raises(ValueError, match="not complete"):
        pc.get_spectra_embedding(rs.loc[["FL3", "FL1"]], ensure_complete=True)
    rs.loc["FL2", "FL1-A"] = np.nan
    with pytest.raises(ValueError, match="not complete"):
        pc.get_spectra_embedding(rs, ensure_complete=True)


def test_all_fixed_panel_embedding_empty(tmp_path):
    """R2d-D8: a panel where no stain has components (one run -> dummy PCA) gives an empty embedding."""
    c = make_collection(str(tmp_path), n=1)
    pc = PanelConfiguration.from_csv(panel_csv(str(tmp_path), components=("", "", "")), "fixed")
    pc.collect_PCA_data(c)
    assert len(pc.pca_components) == 0
    emb = pc.get_spectra_embedding(pc.get_run_spectra("run0"), ensure_complete=True)
    assert len(emb) == 0 and emb.dtype == float
    out = pc.get_spectra_from_embedding(emb)
    assert np.allclose(out.to_numpy(), pc.pca_offset.loc[synthetic.STAINS].to_numpy())


# ---------------------------------------------------------------- save / load
def assert_same_panel(a, b):
    for key in ["name", "total_components", "has_PCA_data", "channels"]:
        assert getattr(a, key) == getattr(b, key)
    for key in ["stains", "markers", "stain_marker_name"]:
        assert np.array_equal(getattr(a, key), getattr(b, key))
    assert [list(x) for x in a.components] == [list(x) for x in b.components]
    for key in ["pca_components", "pca_embedding", "pca_offset", "pca_explained_variance_channelratio", "df"]:
        pd.testing.assert_frame_equal(getattr(a, key), getattr(b, key))
    for key in ["pca_components_stain", "pca_components_use", "pca_explained_variance"]:
        pd.testing.assert_series_equal(getattr(a, key), getattr(b, key))
    assert set(a.spectra_full) == set(b.spectra_full)
    for k in a.spectra_full:
        pd.testing.assert_frame_equal(a.spectra_full[k], b.spectra_full[k])


def test_save_and_load_roundtrip(pc_and_collection, tmp_path):
    pc, _ = pc_and_collection
    pc.save(str(tmp_path / "panel.pt"))
    back = PanelConfiguration.from_saved(str(tmp_path / "panel.pt"))
    assert_same_panel(pc, back)
    assert_same_panel(pc, PanelConfiguration.from_saved(exported=pc.export()))
    emb = back.get_spectra_embedding(back.get_run_spectra("run1"))
    assert np.allclose(emb.to_numpy(), pc.pca_embedding.loc["run1"].to_numpy())


def test_save_without_PCA_data(tmp_path):
    pc = PanelConfiguration.from_csv(panel_csv(str(tmp_path)), "p")
    pc.save(str(tmp_path / "p.pt"))
    back = PanelConfiguration.from_saved(str(tmp_path / "p.pt"))
    assert not back.has_PCA_data and back.stains.tolist() == synthetic.STAINS


# ---------------------------------------------------------------- plots
def test_plot_embedding(pc_and_collection):
    pc, _ = pc_and_collection
    pc.plot_embedding()


def test_plot_test_spectra_reconstruction_aligned_rows(pc_and_collection):
    """R2-E20: rows in another order than the panel and a subset of the dyes are aligned by name."""
    pc, _ = pc_and_collection
    rs = pc.get_run_spectra("run3")
    sub = rs.loc[["FL3", "FL1"]]
    diff = pc.plot_test_spectra_reconstruction(sub)
    assert diff.index.tolist()[:3] == ["FL3 | FL1-A", "FL3 | FL2-A", "FL3 | FL3-A"]
    assert len(diff) == 6
    exp = (pc.get_spectra_from_embedding(pc.get_spectra_embedding(sub), allow_missing=True).loc[sub.index] - sub).abs()
    assert np.allclose(diff.to_numpy(), exp.to_numpy().flatten())
    # FL1 uses both of its components: error 0; FL3 uses none: error = |mean - spectrum|
    assert np.allclose(diff[[i for i in diff.index if i.startswith("FL1")]], 0., atol=1e-10)


def test_plot_test_spectra_reconstruction_guide_lines(pc_and_collection):
    """R2-E20: the 100%/10% guide lines reach the y limit (log10 exponent) of each panel separately."""
    import matplotlib.pyplot as plt
    pc, _ = pc_and_collection
    pc.plot_test_spectra_reconstruction(pc.get_run_spectra("run3"))
    ax = plt.gcf().axes
    for a in ax:
        bottom, top = a.get_ylim()
        xmax = [line.get_xdata().max() for line in a.lines]
        assert xmax == pytest.approx([top, top, -bottom, -bottom])
