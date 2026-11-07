"""Tests of cytovanni.spill.utils: finding and collecting single-stain controls (regular and spectral flow)."""
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.exceptions import SpilloverFitException, SpilloverFitWarning
from cytovanni.io import read_bead_h5ad, writefcs
from cytovanni.spill import collect_bead_controls_regularflow, collect_bead_controls_spectralflow
from cytovanni.spill.utils import _find_single_stain, _get_AutoSpill_script_path

REGEX = r"Compensation Controls_(.*)FL1 Stained Control\.fcs$"


def touch_folder(path, names):
    os.makedirs(path, exist_ok=True)
    for n in names:
        open(os.path.join(path, n), "w").close()
    return str(path)


def sim_config(channels):
    """The simulated cytometer with other fluorescence channel names."""
    from cytovanni.utils import CytometerConfiguration
    return CytometerConfiguration(channels, channels_scatter=synthetic.CHANNELS_SCATTER, channels_meta=["Time"],
                                  max_linear_range=2e5, cytometer="SIM")


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- _find_single_stain
def test_find_single_stain_returns_file_and_stripped_antigen(tmp_path):
    d = touch_folder(tmp_path, ["Compensation Controls_ CD1  FL1 Stained Control.fcs", "x.txt"])
    assert _find_single_stain(d, REGEX, "FL1") == ("Compensation Controls_ CD1  FL1 Stained Control.fcs", "CD1")


def test_find_single_stain_missing_warns(tmp_path):
    d = touch_folder(tmp_path, ["other.fcs"])
    with pytest.warns(SpilloverFitWarning, match="FL1"):
        assert _find_single_stain(d, REGEX, "FL1") == ("", "")


def test_find_single_stain_missing_silent(tmp_path, no_warnings):
    d = touch_folder(tmp_path, ["other.fcs"])
    assert _find_single_stain(d, REGEX, "FL1", warn_missing=False) == ("", "")


def test_find_single_stain_empty_folder(tmp_path):
    """R2-E10: no .fcs files -> reported as not found (was ValueError from max() of an empty list)."""
    d = touch_folder(tmp_path / "empty", ["notes.txt"])
    assert _find_single_stain(d, REGEX, "FL1", warn_missing=False) == ("", "")


def test_find_single_stain_ambiguous_raises(tmp_path):
    d = touch_folder(tmp_path, ["Compensation Controls_A FL1 Stained Control.fcs", "Compensation Controls_B FL1 Stained Control.fcs"])
    with pytest.raises(SpilloverFitException, match="multiple possible matches"):
        _find_single_stain(d, REGEX, "FL1")


def test_find_single_stain_several_matches_in_one_file_raises(tmp_path):
    d = touch_folder(tmp_path, ["aXbX.fcs"])
    with pytest.raises(SpilloverFitException, match="more than one regex match per file"):
        _find_single_stain(d, r"(X)", "FL1")


def test_autospill_script_bundled():
    p = _get_AutoSpill_script_path()
    assert p.endswith(os.path.join("AutoSpill", "run_AutoSpil.R")) and os.path.exists(p)


# ---------------------------------------------------------------- regular flow (real .fcs, io + gating)
@pytest.fixture(scope="module")
def collected_regular(single_stain_folder, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("collect_regular") / "run1")
    quiet(collect_bead_controls_regularflow, single_stain_folder, synthetic.make_config(), out, verbose=False)
    return out


def test_regularflow_panel_meta(collected_regular):
    meta = pd.read_csv(os.path.join(collected_regular, "panel_meta.csv"), index_col=0)
    assert list(meta.index) == [0, 1, 2]
    assert list(meta.columns) == ["channel", "dye", "antigen", "datafile", "set_hash"]
    assert meta["channel"].tolist() == synthetic.CHANNELS_FLUORESCENCE
    assert meta["dye"].tolist() == synthetic.STAINS
    assert meta["antigen"].tolist() == synthetic.MARKERS
    assert meta["datafile"].tolist() == [f"Compensation_Control_{c}.h5ad" for c in synthetic.CHANNELS_FLUORESCENCE]
    assert meta["set_hash"].nunique() == 1 and isinstance(meta["set_hash"][0], str)


def test_regularflow_h5ad_content(collected_regular):
    adata = read_bead_h5ad(os.path.join(collected_regular, "data", "Compensation_Control_FL2-A.h5ad"))
    assert adata.uns["dye"] == "FL2" and adata.uns["channel"] == "FL2-A"
    assert (adata.obs["dye"] == "FL2").all() and (adata.obs["channel"] == "FL2-A").all()
    assert "raw" in adata.layers
    assert 0 < adata.shape[0] < 3000  # gated on bead singlets (debris removed)
    assert adata.uns["cytoconfig"].channels_fluorescence == synthetic.CHANNELS_FLUORESCENCE
    assert os.path.exists(os.path.join(collected_regular, "gating", "scattergate_FL2.jpg"))


def test_regularflow_missing_and_drop(single_stain_folder, tmp_path):
    """R2-C15a: set_hash is an object column, NaN where there is no file; R2-C6: warn_missing honoured."""
    src = tmp_path / "src"; src.mkdir()  # FL1 and FL2 present (FL2 dropped), FL3 missing
    for name in ["Compensation Controls_CD1 FL1 Stained Control.fcs", "Compensation Controls_CD2 FL2 Stained Control.fcs"]:
        shutil.copy(os.path.join(single_stain_folder, name), src / name)
    out = str(tmp_path / "out")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        collect_bead_controls_regularflow(str(src), synthetic.make_config(), out, verbose=False, drop=["FL2"],
                                          warn_missing=False)
    assert not [x for x in w if "single stain" in str(x.message)]
    meta = pd.read_csv(os.path.join(out, "panel_meta.csv"), index_col=0)
    assert meta["datafile"].fillna("").tolist() == ["Compensation_Control_FL1-A.h5ad", "", ""]
    assert meta["set_hash"].isna().tolist() == [False, True, True]
    assert sorted(os.listdir(os.path.join(out, "data"))) == ["Compensation_Control_FL1-A.h5ad"]
    with pytest.warns(SpilloverFitWarning, match="FL3"):
        collect_bead_controls_regularflow(str(src), synthetic.make_config(), str(tmp_path / "out2"), verbose=False,
                                          drop=["FL2"])


def test_regularflow_mixed_settings_warns(single_stain_folder, tmp_path):
    src = tmp_path / "src"; src.mkdir()
    shutil.copy(os.path.join(single_stain_folder, "Compensation Controls_CD1 FL1 Stained Control.fcs"), src)
    df = synthetic.single_stain_events("FL3", "A", n=1000, seed=5)
    writefcs(str(src / "Compensation Controls_CD3 FL3 Stained Control.fcs"), df, add_metadata=synthetic.fcs_metadata("B"))
    with pytest.warns(SpilloverFitWarning, match="multiple different cytometer settings"):
        collect_bead_controls_regularflow(str(src), synthetic.make_config(), str(tmp_path / "out"), verbose=False)
    meta = pd.read_csv(tmp_path / "out" / "panel_meta.csv", index_col=0)
    assert meta["set_hash"].nunique() == 2


def test_regularflow_ambiguous_raises(single_stain_folder, tmp_path):
    src = tmp_path / "src"
    shutil.copytree(single_stain_folder, src)
    shutil.copy(src / "Compensation Controls_CD1 FL1 Stained Control.fcs", src / "Compensation Controls_CDx FL1 Stained Control.fcs")
    with pytest.raises(SpilloverFitException):
        quiet(collect_bead_controls_regularflow, str(src), synthetic.make_config(), str(tmp_path / "out"), verbose=False)


def test_regularflow_regex_escaped_channel_names(single_stain_folder, tmp_path):
    """R2-C5: channel names with regex metacharacters (FACSAria 'B1 (525)') are found."""
    src = tmp_path / "src"; src.mkdir()
    df = synthetic.single_stain_events("FL1", "A", n=1000)
    df = df.rename(columns={"FL1-A": "B1 (525)-A"})
    writefcs(str(src / "Compensation Controls_CD3 B1 (525) Stained Control.fcs"), df, add_metadata=synthetic.fcs_metadata("A"))
    chans = ["B1 (525)-A", "FL2-A", "FL3-A"]
    cfg = sim_config(chans)
    out = str(tmp_path / "out")
    quiet(collect_bead_controls_regularflow, str(src), cfg, out, verbose=False)
    meta = pd.read_csv(os.path.join(out, "panel_meta.csv"), index_col=0)
    assert meta.loc[0, "antigen"] == "CD3" and meta.loc[0, "datafile"] == "Compensation_Control_B1 (525)-A.h5ad"


# ---------------------------------------------------------------- spectral flow
def test_spectralflow_default_regex(single_stain_folder, tmp_path):
    src = tmp_path / "src"; src.mkdir()
    for name in ["Compensation Controls_CD3 FL3 Stained Control.fcs", "Compensation Controls_CD1 FL1 Stained Control.fcs",
                 "Unstained Control.fcs"]:
        shutil.copy(os.path.join(single_stain_folder, name), src / name)
    out = str(tmp_path / "out")
    quiet(collect_bead_controls_spectralflow, str(src), synthetic.make_config(), out, verbose=False)
    meta = pd.read_csv(os.path.join(out, "panel_meta.csv"), index_col=0)
    assert list(meta.columns) == ["dye", "datafile", "set_hash"]
    assert meta["dye"].tolist() == ["CD1 FL1 Stained Control", "CD3 FL3 Stained Control"]  # sorted by dye
    assert list(meta.index) == [0, 1]  # Unstained Control.fcs does not match
    adata = read_bead_h5ad(os.path.join(out, "data", meta.loc[0, "datafile"]))
    assert "channel" not in adata.uns and adata.uns["dye"] == meta.loc[0, "dye"]


def test_spectralflow_custom_regex_and_mixed_settings(single_stain_folder, tmp_path):
    """R2-C6: filename_regex is used (was overwritten by the default); R2-C15a: str hashes stored."""
    src = tmp_path / "src"; src.mkdir()
    for dye, batch in [("FL1", "A"), ("FL2", "B")]:
        writefcs(str(src / f"SS_{dye}.fcs"), synthetic.single_stain_events(dye, batch, n=1000), add_metadata=synthetic.fcs_metadata(batch))
    (src / "other.txt").write_text("")
    out = str(tmp_path / "out")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        collect_bead_controls_spectralflow(str(src), synthetic.make_config(), out, verbose=False, filename_regex=r"SS_(.*)\.fcs$")
    assert any("multiple different cytometer settings" in str(x.message) for x in w)
    meta = pd.read_csv(os.path.join(out, "panel_meta.csv"), index_col=0)
    assert meta["dye"].tolist() == ["FL1", "FL2"]
    assert meta["datafile"].tolist() == ["Compensation_Control_FL1.h5ad", "Compensation_Control_FL2.h5ad"]
    assert meta["set_hash"].nunique() == 2


def test_spectralflow_default_regex_anchored(tmp_path):
    """R2-C5: the default spectral regex is anchored at '.fcs'."""
    import re
    import inspect
    default = inspect.signature(collect_bead_controls_spectralflow).parameters["filename_regex"].default
    assert re.findall(default, "Compensation Controls_FITC.fcs") == ["FITC"]
    assert re.findall(default, "Compensation Controls_FITC.fcs.bak") == []
    assert re.findall(default, "Compensation Controls_FITCxfcs") == []


@pytest.mark.xfail(strict=True, reason="bug (todo/R4_findings/spill.md): '(.*){dye}' also matches a dye name that ends "
                                        "another channel name (V540 in UV540, V1 in UV1: S6, Aurora configs)")
def test_regularflow_channel_that_ends_another_channel_name(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    df = synthetic.single_stain_events("FL1", "A", n=1000).rename(columns={"FL1-A": "UV540-A", "FL2-A": "V540-A"})
    writefcs(str(src / "Compensation Controls_CD3 UV540 Stained Control.fcs"), df, add_metadata=synthetic.fcs_metadata("A"))
    out = str(tmp_path / "out")
    quiet(collect_bead_controls_regularflow, str(src), sim_config(["UV540-A", "V540-A", "FL3-A"]), out, verbose=False)
    meta = pd.read_csv(os.path.join(out, "panel_meta.csv"), index_col=0)
    assert meta.loc[0, "antigen"] == "CD3"
    assert pd.isna(meta.loc[1, "datafile"])  # V540 has no single stain
