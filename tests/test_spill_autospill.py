"""Tests of cytovanni.spill.autospill.make_AutoSpill_fcs_config."""
import os
from types import SimpleNamespace

import pytest

import synthetic
from cytovanni.exceptions import SpilloverFitException, SpilloverFitWarning
from cytovanni.spill import make_AutoSpill_fcs_config


def touch_folder(path, names):
    os.makedirs(path, exist_ok=True)
    for n in names:
        open(os.path.join(path, n), "w").close()
    return str(path)


def test_synthetic_single_stains_all_found(single_stain_folder, config, no_warnings):
    out = make_AutoSpill_fcs_config(single_stain_folder, config)
    assert list(out.columns) == ["filename_source", "filename", "dye", "antigen", "wavelength"]
    assert list(out.index) == [0, 1, 2]
    assert out["dye"].tolist() == synthetic.CHANNELS_FLUORESCENCE  # with -A
    assert out["antigen"].tolist() == synthetic.MARKERS
    assert out["wavelength"].tolist() == [1, 2, 3]
    assert (out["filename"] == out["filename_source"]).all()
    assert out["filename"][0] == "Compensation Controls_CD1 FL1 Stained Control.fcs"


def test_padding_distinct_names_and_regex_escape(tmp_path):
    """R2-C5, R2-C7, R2-F4: channels with parentheses are found; padded channels get _0, _1, ...; no internal column."""
    d = touch_folder(tmp_path, ["Compensation Controls_CD3 B1 (525) Stained Control.fcs",
                                "Compensation Controls_CD4 R1 (670) Stained Control.fcs"])
    cfg = SimpleNamespace(channels_fluorescence=["B1 (525)-A", "R1 (670)-A", "V1 (450)-A", "V2 (525)-A", "Y1 (586)-A"])
    with pytest.warns(SpilloverFitWarning, match="no availiable single stain"):
        out = make_AutoSpill_fcs_config(d, cfg)
    assert out["antigen"].tolist() == ["CD3", "CD4", "_dummy_pad", "_dummy_pad", "_dummy_pad"]
    pad = "Compensation Controls_CD3 B1 (525) Stained Control.fcs"
    assert out["filename_source"].tolist()[2:] == [pad] * 3
    assert out["filename"].tolist()[2:] == [pad.replace(".fcs", f"_{i}.fcs") for i in range(3)]
    assert len(set(out["filename"])) == 5
    assert "channel_name" not in out.columns and "channnel_name" not in out.columns


def test_regex_metacharacters_not_wildcards(tmp_path):
    """R2-C5: '.' in a channel name matches only a dot; the pattern is anchored at '.fcs'."""
    d = touch_folder(tmp_path, ["Compensation Controls_CD8 PE-Cy5x5 Stained Control.fcs",
                                "Compensation Controls_CD4 FITC Stained Control.fcs.bak.fcs",
                                "Compensation Controls_CD5 PE-Cy7 Stained Control.fcs"])
    cfg = SimpleNamespace(channels_fluorescence=["PE-Cy5.5-A", "FITC-A", "PE-Cy7-A"])
    with pytest.warns(SpilloverFitWarning):
        out = make_AutoSpill_fcs_config(d, cfg, pad_missing=False)
    assert out["filename"].tolist() == ["", "", "Compensation Controls_CD5 PE-Cy7 Stained Control.fcs"]


def test_no_padding_leaves_missing_empty(tmp_path):
    d = touch_folder(tmp_path, ["Compensation Controls_ FL1 Stained Control.fcs"])
    cfg = SimpleNamespace(channels_fluorescence=["FL1-A", "FL2-A"])
    out = make_AutoSpill_fcs_config(d, cfg, pad_missing=False, warn_missing=False)
    assert out["filename"].tolist() == ["Compensation Controls_ FL1 Stained Control.fcs", ""]
    assert out["antigen"].tolist() == ["", ""]  # empty antigen (stripped)


def test_warn_missing_false_is_silent(tmp_path, no_warnings):
    d = touch_folder(tmp_path, ["Compensation Controls_CD1 FL1 Stained Control.fcs"])
    out = make_AutoSpill_fcs_config(d, SimpleNamespace(channels_fluorescence=["FL1-A", "FL2-A"]), warn_missing=False)
    assert out.loc[1, "antigen"] == "_dummy_pad"


def test_channel_without_A_suffix(tmp_path):
    d = touch_folder(tmp_path, ["Compensation Controls_CD1 FL1 Stained Control.fcs"])
    out = make_AutoSpill_fcs_config(d, SimpleNamespace(channels_fluorescence=["FL1"]))
    assert out["filename"][0] == "Compensation Controls_CD1 FL1 Stained Control.fcs" and out["dye"][0] == "FL1"


def test_no_single_stains_raises(tmp_path):
    d = touch_folder(tmp_path, ["other.fcs"])
    with pytest.raises(SpilloverFitException, match="Could not find any single stains"):
        make_AutoSpill_fcs_config(d, SimpleNamespace(channels_fluorescence=["FL1-A"]), warn_missing=False)


def test_empty_folder_raises_clear_error(tmp_path):
    """R2-E10: a folder without .fcs files gives the clear 'no single stains' error (was ValueError from max())."""
    d = touch_folder(tmp_path / "empty", [])
    with pytest.raises(SpilloverFitException, match="Could not find any single stains"):
        make_AutoSpill_fcs_config(d, SimpleNamespace(channels_fluorescence=["FL1-A"]), warn_missing=False)


def test_ambiguous_match_raises(tmp_path):
    d = touch_folder(tmp_path, ["Compensation Controls_CD1 FL1 Stained Control.fcs",
                                "Compensation Controls_CD2 FL1 Stained Control.fcs"])
    with pytest.raises(SpilloverFitException, match="multiple possible matches"):
        make_AutoSpill_fcs_config(d, SimpleNamespace(channels_fluorescence=["FL1-A"]))


def test_channel_that_ends_another_channel_name_not_matched(tmp_path):
    d = touch_folder(tmp_path, ["Compensation Controls_CD3 UV540 Stained Control.fcs"])
    out = make_AutoSpill_fcs_config(d, SimpleNamespace(channels_fluorescence=["UV540-A", "V540-A"]),
                                    pad_missing=False, warn_missing=False)
    assert out["filename"].tolist() == ["Compensation Controls_CD3 UV540 Stained Control.fcs", ""]
