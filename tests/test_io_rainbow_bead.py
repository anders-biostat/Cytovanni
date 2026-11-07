"""Rainbow and bead samples: readfcs_rainbow_sample and the .h5ad writers/readers (cytovanni.io.rainbow, io.bead)."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.exceptions import ChannelMismatchException, CytometerConfigurationException, RainbowSpilloverWarning
from cytovanni.io import (read_bead_h5ad, read_rainbow_h5ad, readfcs_rainbow_sample, readfcs_sample, write_bead_h5ad,
                          write_rainbow_h5ad, writefcs)
from cytovanni.utils import CytometerConfiguration, register_hash_function
from cytovanni.utils.config import CytometerConfiguration_FACSAriaIII


def test_readfcs_rainbow_sample_adds_rainbow_metadata(rainbow_files, config):
    p = rainbow_files["B"][0]
    a = readfcs_rainbow_sample(p, config, 3, name="day2", rainbow_type="lot_A", rainbow_batch="B", metadct={"op": "x"})
    plain = readfcs_sample(p, config)
    assert np.array_equal(a.layers["raw"], plain.layers["raw"]) and a.uns["uid"] == plain.uns["uid"]
    assert a.uns["name"] == "day2" and a.uns["name_uid"] == f"day2 ({a.uns['uid']})"
    assert a.uns["N_rainbow_peak"] == 3 and a.uns["rainbow_type"] == "lot_A" and a.uns["rainbow_batch"] == "B"
    assert a.uns["op"] == "x" and a.uns["date"] == synthetic.DATES["B"]


def test_readfcs_rainbow_sample_defaults(rainbow_files, config):
    a = readfcs_rainbow_sample(rainbow_files["A"][0], config, 8)
    assert a.uns["name"] == "" and a.uns["rainbow_type"] == "default" and a.uns["rainbow_batch"] == "default"
    assert a.uns["name_uid"] == f" ({a.uns['uid']})"


def test_readfcs_rainbow_sample_spillover_warning(tmp_path, config):
    df, _ = synthetic.rainbow_events("A", n=200)
    p = str(tmp_path / "rb_spill.fcs")
    writefcs(p, df, add_metadata={**synthetic.fcs_metadata("A"), "spill": "2,FL1-A,FL2-A,1,0.3,0,1"})
    with pytest.warns(RainbowSpilloverWarning, match="compensating rainbow beads"):
        readfcs_rainbow_sample(p, config, 3, warn_spillover=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        readfcs_rainbow_sample(p, config, 3)
    assert not [x for x in w if issubclass(x.category, RainbowSpilloverWarning)]


def test_readfcs_rainbow_sample_missing_channel(tmp_path):
    df, _ = synthetic.rainbow_events("A", n=50)
    p = str(tmp_path / "rb.fcs")
    writefcs(p, df.drop(columns=["FL2-A"]))
    with pytest.raises(ChannelMismatchException):
        readfcs_rainbow_sample(p, synthetic.make_config(), 3)


# ------------------------------------------------------------------------------------------------ .h5ad round trips
WRITERS = [(write_rainbow_h5ad, read_rainbow_h5ad), (write_bead_h5ad, read_bead_h5ad)]


@pytest.fixture
def rainbow_adata(rainbow_files, config):
    return readfcs_rainbow_sample(rainbow_files["C"][1], config, 3, name="rb", rainbow_batch="C")


def _export_json(cfg):
    return json.loads(json.dumps(cfg.to_dictionary()))


@pytest.mark.parametrize("write,read", WRITERS)
def test_h5ad_round_trip(tmp_path, rainbow_adata, config, write, read):
    p = str(tmp_path / "x.h5ad")
    write(rainbow_adata, p)
    assert rainbow_adata.uns["cytoconfig"] is config  # the input is not changed
    b = read(p)
    assert np.array_equal(b.layers["raw"], rainbow_adata.layers["raw"])
    assert b.var.equals(rainbow_adata.var) or b.var.astype(str).equals(rainbow_adata.var.astype(str))
    assert list(b.obs_names) == list(rainbow_adata.obs_names)
    for k in ["date", "UTC", "uid", "set_hash", "name_uid", "N_rainbow_peak", "rainbow_batch", "Nevents"]:
        assert b.uns[k] == rainbow_adata.uns[k], k
    cfg = b.uns["cytoconfig"]
    assert type(cfg) is CytometerConfiguration
    assert cfg.channels_fluorescence == config.channels_fluorescence and cfg.timezone == config.timezone
    assert cfg.get_settings_hash(b) == rainbow_adata.uns["set_hash"]


@pytest.mark.parametrize("write,read", WRITERS)
def test_h5ad_config_is_builtin_and_json_writable(tmp_path, rainbow_adata, config, write, read):
    """R2b-G2-02: a configuration read from .h5ad holds builtins (no ndarrays), equals the original, and writes to json."""
    p = str(tmp_path / "x.h5ad")
    write(rainbow_adata, p)
    cfg = read(p).uns["cytoconfig"]
    assert isinstance(cfg.laserdict["Blue"], list) and isinstance(cfg.default_scatter_calibrate["FSC-A"], list)
    assert _export_json(cfg) == _export_json(config)
    cfg.write_json(str(tmp_path / "cfg.json"))
    back = CytometerConfiguration.load_json(str(tmp_path / "cfg.json"))
    assert back.laserdict == config.laserdict


@pytest.mark.parametrize("write,read", WRITERS)
def test_h5ad_restores_subclass_and_extras(tmp_path, write, read):
    """R2-F2: the export keeps the configuration class and its extra attributes (here pandas.Series of the FACSAria)."""
    cfg = CytometerConfiguration_FACSAriaIII()
    a = synthetic.make_adata(np.ones((3, len(cfg.channels_fluorescence))), cfg.channels_fluorescence, layer="raw",
                             cytoconfig=cfg)
    p = str(tmp_path / "aria.h5ad")
    write(a, p)
    back = read(p).uns["cytoconfig"]
    assert type(back) is CytometerConfiguration_FACSAriaIII
    assert back.lasers == cfg.lasers
    for key in ["lasers_wavelength", "lasers_color"]:
        assert isinstance(getattr(back, key), pd.Series) and getattr(back, key).equals(getattr(cfg, key)), key
    assert back.replace_channelname_dict == cfg.replace_channelname_dict


def _hash_for_r4_io_test(adata):
    return "constant"


@pytest.mark.parametrize("write,read", WRITERS)
def test_h5ad_custom_hash_function(tmp_path, write, read):
    """R2-F2: a registered custom hash function comes back by name; an unregistered one raises on reading."""
    df, _ = synthetic.rainbow_events("A", n=20)

    def unregistered(adata):
        return "x"
    cfg_bad = synthetic.make_config(fct_hash_settings=unregistered)
    a = synthetic.make_adata(np.ones((2, 3)), synthetic.CHANNELS_FLUORESCENCE, layer="raw", cytoconfig=cfg_bad)
    write(a, str(tmp_path / "bad.h5ad"))
    with pytest.raises(CytometerConfigurationException):
        read(str(tmp_path / "bad.h5ad"))
    register_hash_function(_hash_for_r4_io_test)
    a.uns["cytoconfig"] = synthetic.make_config(fct_hash_settings=_hash_for_r4_io_test)
    write(a, str(tmp_path / "good.h5ad"))
    assert read(str(tmp_path / "good.h5ad")).uns["cytoconfig"].fct_hash_settings is _hash_for_r4_io_test


def test_h5ad_target_folder_must_exist(tmp_path, rainbow_adata):
    with pytest.raises(Exception):
        write_rainbow_h5ad(rainbow_adata, str(tmp_path / "missing" / "x.h5ad"))
