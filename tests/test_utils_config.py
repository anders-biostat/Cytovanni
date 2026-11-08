"""Tests of cytovanni.utils.config: CytometerConfiguration, settings hash, export/import, instrument subclasses."""
import json
import warnings

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import cytovanni.utils.config as C
from cytovanni.exceptions import (CytometerConfigurationException, CytovanniWarning, DefaultScatterCalibrationException,
                                  DefaultScatterGatingException, SettingsHashException)
from cytovanni.io import read_rainbow_h5ad, write_rainbow_h5ad
from cytovanni.utils import (CytometerConfiguration, CytometerConfiguration_A3, CytometerConfiguration_A3small,
                             CytometerConfiguration_Aurora, CytometerConfiguration_DxFLEX1L,
                             CytometerConfiguration_FACSAriaFusion, CytometerConfiguration_FACSAriaFusion_v2,
                             CytometerConfiguration_FACSAriaIII, CytometerConfiguration_FACSAriaIII_v2,
                             CytometerConfiguration_NorthernLights, CytometerConfiguration_S6,
                             CytometerConfiguration_S8, register_hash_function)

SUBCLASSES = [CytometerConfiguration_A3, CytometerConfiguration_A3small, CytometerConfiguration_S6,
              CytometerConfiguration_S8, CytometerConfiguration_FACSAriaIII, CytometerConfiguration_FACSAriaIII_v2,
              CytometerConfiguration_FACSAriaFusion, CytometerConfiguration_FACSAriaFusion_v2,
              CytometerConfiguration_DxFLEX1L, CytometerConfiguration_Aurora, CytometerConfiguration_NorthernLights]


@pytest.fixture
def registry(monkeypatch):
    """A private copy of the settings-hash registry, so registrations do not leak into other tests."""
    monkeypatch.setattr(C, "HASH_FUNCTIONS", dict(C.HASH_FUNCTIONS))
    return C.HASH_FUNCTIONS


def settings_adata(cfg, pnv=None, png=None, drop=()):
    """Sample with the configured scatter + fluorescence channels and their $PnV / $PnG settings in var."""
    ch = [c for c in cfg.channels_scatter + cfg.channels_fluorescence if c not in drop]
    var = pd.DataFrame({"pnv": [str(300 + i) for i in range(len(ch))] if pnv is None else pnv,
                        "png": ["1.0"] * len(ch) if png is None else png}, index=ch)
    a = anndata.AnnData(np.zeros((2, len(ch)), np.float32), var=var, obs=pd.DataFrame(index=["0", "1"]))
    a.uns["cytoconfig"] = cfg
    return a


def same_config(a, b):
    """Names of the attributes in which the two configurations differ."""
    keys = ["channels_fluorescence", "channels_scatter", "channels_meta", "default_scatter_gate_channels",
            "max_linear_range", "cytometer", "try_adding_HW", "channels_fluorescence_addHW", "timezone",
            "laserdict", "default_scatter_calibrate", "replace_channelname_dict"]
    diffs = [k for k in keys if getattr(a, k) != getattr(b, k)]
    if type(a) is not type(b):
        diffs.append("class")
    if a.fct_hash_settings is not b.fct_hash_settings:
        diffs.append("fct_hash_settings")
    if not a.channel_laser_key.equals(b.channel_laser_key):
        diffs.append("channel_laser_key")
    for k in type(a)._export_extras:
        va, vb = getattr(a, k), getattr(b, k, None)
        if not (va.equals(vb) if isinstance(va, pd.Series) else va == vb):
            diffs.append(k)
    return diffs


# ------------------------------------------------------------------ construction
def test_defaults_and_added_HW_channels():
    cfg = CytometerConfiguration(["FITC-A", "PE-A", "PE-H", "Odd"])
    assert cfg.channels_scatter == ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    assert cfg.default_scatter_gate_channels == ["FSC-A", "SSC-A"] and cfg.channels_meta == ["Time"]
    assert cfg.default_scatter_calibrate == {"FSC-A": ["FSC-A", "FSC-H"], "SSC-A": ["SSC-A", "SSC-H"]}
    assert cfg.max_linear_range == 2e5 and isinstance(cfg.max_linear_range, float)
    assert cfg.cytometer == "default" and cfg.timezone == "Europe/Berlin"
    assert cfg.fct_hash_settings is C._default_hash_settings
    assert cfg.channels_fluorescence_addHW == ["FITC-H", "FITC-W", "PE-W"]  # existing PE-H and non -A names skipped
    assert CytometerConfiguration(["FITC-A"], try_adding_HW=False).channels_fluorescence_addHW == []
    assert "Cytometer configuration for cytometer default" in repr(cfg)


def test_channel_laser_key():
    cfg = CytometerConfiguration(["FITC-A", "APC-A"], laserdict={"Blue": ["FITC-A", "FSC-A", "notloaded"], "Red": ["APC-A"]})
    k = cfg.channel_laser_key
    assert list(k.index) == cfg.channels_scatter + cfg.channels_fluorescence
    assert k["FITC-A"] == "Blue" and k["FSC-A"] == "Blue" and k["APC-A"] == "Red" and k["SSC-A"] == "unknown"
    assert "notloaded" not in k.index


def test_no_shared_mutable_state():
    """R2-E5: instances copy their (default) dict/list arguments."""
    lasers = {"Blue": ["FITC-A"]}; scal = {"FSC-A": ["FSC-A"]}; chans = ["FITC-A"]
    a = CytometerConfiguration(chans, laserdict=lasers, default_scatter_calibrate=scal)
    lasers["Blue"].append("PE-A"); scal["FSC-A"].append("FSC-H"); chans.append("X-A")
    assert a.laserdict == {"Blue": ["FITC-A"]} and a.default_scatter_calibrate == {"FSC-A": ["FSC-A"]}
    assert a.channels_fluorescence == ["FITC-A"]
    b, c = CytometerConfiguration(["FITC-A"]), CytometerConfiguration(["FITC-A"])
    b.default_scatter_calibrate["FSC-A"].append("X"); b.laserdict["x"] = []; b.replace_channelname_dict["x"] = "y"
    b.channels_scatter.append("Q")
    assert c.default_scatter_calibrate["FSC-A"] == ["FSC-A", "FSC-H"] and c.laserdict == {}
    assert c.replace_channelname_dict == {} and "Q" not in c.channels_scatter


@pytest.mark.parametrize("kwargs", [
    dict(channels_fluorescence=["FSC-A", "PE-A"]),  # scatter and fluorescence overlap
    dict(channels_fluorescence=["FSC-A"], channels_scatter=["FSC-H"]),  # added -H overlaps scatter
    dict(channels_fluorescence=["PE-A"], laserdict={"Blue": ["PE-A", "PE-A"]}),  # duplicate within a laser
    dict(channels_fluorescence=["PE-A"], laserdict={"Blue": ["PE-A"], "Red": ["PE-A"]}),  # two lasers
    dict(channels_fluorescence=["PE-A"], timezone="Mars/Olympus"),
])
def test_invalid_configurations_raise(kwargs):
    with pytest.raises(CytometerConfigurationException):
        CytometerConfiguration(**kwargs)


def test_overlap_with_added_HW_is_fine_without_adding():
    cfg = CytometerConfiguration(["FSC-A"], channels_scatter=["FSC-H"], try_adding_HW=False)
    assert cfg.channels_fluorescence_addHW == []


# ------------------------------------------------------------------ checked getters
def test_default_scatter_gate_channels():
    assert CytometerConfiguration(["PE-A"]).get_default_scatter_gate_channels() == ["FSC-A", "SSC-A"]
    assert CytometerConfiguration(["PE-A"], default_scatter_gate=["PE-W", "SSC-A"]).get_default_scatter_gate_channels() == ["PE-W", "SSC-A"]
    for gate in (["FSC-A"], ["FSC-A", "SSC-A", "FSC-H"], ["FSC-A", "nope"]):
        with pytest.raises(DefaultScatterGatingException):
            CytometerConfiguration(["PE-A"], default_scatter_gate=gate).get_default_scatter_gate_channels()


def test_scatter_calibration_checks():
    cfg = CytometerConfiguration(["PE-A"])
    assert cfg.get_default_scatter_calibration() is cfg.default_scatter_calibrate  # not a copy
    cfg.check_scatter_calibration({"FSC-A": ["FSC-A", "FSC-H", "FSC-W"]})
    with pytest.raises(DefaultScatterCalibrationException, match="default"):
        CytometerConfiguration(["PE-A"], default_scatter_calibrate={"FSC-A": ["PE-A"]}).get_default_scatter_calibration()
    with pytest.raises(DefaultScatterCalibrationException):
        cfg.check_scatter_calibration({"X-A": ["FSC-A"]})
    with pytest.raises(DefaultScatterCalibrationException, match="overlap"):
        cfg.check_scatter_calibration({"FSC-A": ["FSC-A", "FSC-H"], "SSC-A": ["SSC-A", "FSC-H"]})
    with pytest.raises(DefaultScatterCalibrationException) as e:
        cfg.check_scatter_calibration({"X-A": ["FSC-A"]})
    assert "default" not in str(e.value)


# ------------------------------------------------------------------ settings hash
def test_default_hash_deterministic_and_sensitive():
    cfg = CytometerConfiguration(["PE-A", "APC-A"])
    h = cfg.get_settings_hash(settings_adata(cfg))
    assert isinstance(h, str) and len(h) == 64 and h == cfg.get_settings_hash(settings_adata(cfg))
    n = len(cfg.channels_scatter) + 2
    other_v = [str(300 + i) for i in range(n)]; other_v[-1] = "999"
    other_g = ["1.0"] * n; other_g[0] = "2.0"
    assert cfg.get_settings_hash(settings_adata(cfg, pnv=other_v)) != h
    assert cfg.get_settings_hash(settings_adata(cfg, png=other_g)) != h
    # the same numbers as other strings: values are compared as floats
    assert cfg.get_settings_hash(settings_adata(cfg, pnv=[f"{300 + i}.0" for i in range(n)])) == h


def test_default_hash_options_and_reduced_prefix():
    cfg = CytometerConfiguration(["PE-A", "APC-A"])
    n = len(cfg.channels_scatter) + 2
    v = [str(300 + i) for i in range(n)]; v[-1] = "999"
    g = ["1.0"] * n; g[0] = "2.0"
    base, changed_v, changed_g = settings_adata(cfg), settings_adata(cfg, pnv=v), settings_adata(cfg, png=g)
    h = C._default_hash_settings
    assert h(base, use_voltages=False) == h(changed_v, use_voltages=False) != h(changed_g, use_voltages=False)
    assert h(base, use_gains=False) == h(changed_g, use_gains=False) != h(changed_v, use_gains=False)
    assert h(base, include_fluorescence=False) == h(changed_v, include_fluorescence=False)  # APC-A is fluorescence
    assert h(base, include_scatter=False) == h(changed_g, include_scatter=False)  # FSC-A is scatter
    reduced = h(settings_adata(cfg, drop=("SSC-W",), pnv=v[:5] + v[6:], png=g[:5] + g[6:]))
    assert reduced.startswith("red_") and not h(base).startswith("red_")


def test_settings_hash_wraps_errors():
    def broken(adata):
        raise KeyError("pnv")
    cfg = CytometerConfiguration(["PE-A"], fct_hash_settings=broken)
    with pytest.raises(SettingsHashException) as e:
        cfg.get_settings_hash(None)
    assert isinstance(e.value.__cause__, KeyError)
    assert CytometerConfiguration(["PE-A"], fct_hash_settings=lambda a: 0).get_settings_hash(None) == 0


def test_register_hash_function(registry):
    def my_hash(adata):
        return "x"
    def other(adata):
        return "y"
    assert register_hash_function(my_hash) == "my_hash" and registry["my_hash"] is my_hash
    assert register_hash_function(my_hash) == "my_hash"  # same function again is fine
    with pytest.raises(CytometerConfigurationException):
        register_hash_function(other, name="my_hash")
    assert register_hash_function(lambda a: 0, name="const") == "const"


# ------------------------------------------------------------------ export / import
def test_to_dictionary_keys():
    d = CytometerConfiguration_FACSAriaIII().to_dictionary()
    assert d["config_class"] == "CytometerConfiguration_FACSAriaIII" and d["fct_hash_settings"] == "_default_hash_settings"
    assert d["timezone"] == "Europe/Berlin" and d["lasers_wavelength"] == {"index": ["UV", "Violet", "Blue", "Green", "Red"],
                                                                         "values": [355, 405, 488, 532, 640]}
    assert {"channels_fluorescence", "channels_scatter", "channels_meta", "default_scatter_gate_channels",
            "default_scatter_calibrate", "max_linear_range", "cytometer", "try_adding_HW", "channels_fluorescence_addHW",
            "laserdict", "replace_channelname_dict"} <= set(d)
    json.dumps(d)  # json-safe


@pytest.mark.parametrize("klass", SUBCLASSES + [CytometerConfiguration])
def test_round_trip_dict_json_h5ad(klass, tmp_path):
    """R2-F2: class, subclass extras, time zone and hash function survive dict, json and h5ad."""
    cfg = klass() if klass is not CytometerConfiguration else CytometerConfiguration(["PE-A"], laserdict={"B": ["PE-A"]})
    cfg.timezone = "America/New_York"
    r_dict = CytometerConfiguration.from_exported(cfg.to_dictionary())
    cfg.write_json(tmp_path / "c.json")
    r_json = CytometerConfiguration.load_json(tmp_path / "c.json")
    ad = anndata.AnnData(np.zeros((2, 1), np.float32), var=pd.DataFrame(index=["x"]), obs=pd.DataFrame(index=["0", "1"]))
    ad.uns["cytoconfig"] = cfg
    write_rainbow_h5ad(ad, str(tmp_path / "c.h5ad"))
    r_h5 = read_rainbow_h5ad(str(tmp_path / "c.h5ad")).uns["cytoconfig"]
    for route, r in [("dict", r_dict), ("json", r_json), ("h5ad", r_h5)]:
        assert same_config(cfg, r) == [], route
    assert ad.uns["cytoconfig"] is cfg  # writer does not change its input


def test_h5ad_config_is_json_writable(config, tmp_path):
    """R2b-G2-02: a configuration read back from .h5ad holds builtins only and can be written to json."""
    ad = anndata.AnnData(np.zeros((2, 1), np.float32), var=pd.DataFrame(index=["x"]), obs=pd.DataFrame(index=["0", "1"]))
    ad.uns["cytoconfig"] = config
    write_rainbow_h5ad(ad, str(tmp_path / "c.h5ad"))
    cfg2 = read_rainbow_h5ad(str(tmp_path / "c.h5ad")).uns["cytoconfig"]

    def has_np(v):
        if isinstance(v, (np.ndarray, np.generic)):
            return True
        if isinstance(v, dict):
            return any(has_np(k) or has_np(x) for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return any(has_np(x) for x in v)
        return False
    assert not [k for k, v in cfg2.to_dictionary().items() if has_np(v)]
    cfg2.write_json(tmp_path / "c.json")
    assert json.load(open(tmp_path / "c.json"))["laserdict"] == config.laserdict
    assert cfg2.to_dictionary() == config.to_dictionary()


def test_to_builtin_recursive():
    out = C._to_builtin({"a": np.array([1, 2]), "b": [np.float32(1.5), (np.int64(3),)], np.str_("c"): {"d": np.bool_(True)}})
    assert out == {"a": [1, 2], "b": [1.5, (3,)], "c": {"d": True}}
    assert type(out["b"][0]) is float and type(out["c"]["d"]) is bool and type(out["b"][1]) is tuple


def test_custom_hash_function_export(registry):
    """R2-F2: the hash function is stored by name; unregistered names and lambdas raise on import."""
    def my_hash(adata):
        return "custom"
    ex = CytometerConfiguration(["PE-A"], fct_hash_settings=my_hash).to_dictionary()
    assert ex["fct_hash_settings"] == "my_hash"
    with pytest.raises(CytometerConfigurationException, match="(?i)register"):
        CytometerConfiguration.from_exported(ex)
    register_hash_function(my_hash)
    assert CytometerConfiguration.from_exported(ex).fct_hash_settings is my_hash
    lam = CytometerConfiguration(["PE-A"], fct_hash_settings=lambda a: 0).to_dictionary()
    with pytest.raises(CytometerConfigurationException):
        CytometerConfiguration.from_exported(lam)


def test_registered_name_differs_from_function_name(registry):
    def fct(adata):
        return 1
    register_hash_function(fct, name="settings_v1")
    ex = CytometerConfiguration(["PE-A"], fct_hash_settings=fct).to_dictionary()
    assert ex["fct_hash_settings"] == "settings_v1"
    assert CytometerConfiguration.from_exported(ex).fct_hash_settings is fct


def test_old_export_gives_base_defaults():
    """R2-F2: an export without class / hash / time zone (written before the fix) imports as before."""
    old = {k: v for k, v in CytometerConfiguration(["PE-A"]).to_dictionary().items()
           if k not in ("config_class", "fct_hash_settings", "timezone")}
    r = CytometerConfiguration.from_exported(old)
    assert type(r) is CytometerConfiguration and r.fct_hash_settings is C._default_hash_settings
    assert r.timezone == "Europe/Berlin" and r.channels_fluorescence == ["PE-A"]


def test_import_exported_errors():
    with pytest.raises(CytometerConfigurationException, match="from_exported"):
        CytometerConfiguration(["PE-A"]).import_exported(CytometerConfiguration_A3().to_dictionary())
    bad_class = dict(CytometerConfiguration(["PE-A"]).to_dictionary(), config_class="NoSuchConfig")
    with pytest.raises(CytometerConfigurationException, match="Unknown configuration class"):
        CytometerConfiguration.from_exported(bad_class)
    bad_tz = dict(CytometerConfiguration(["PE-A"]).to_dictionary(), timezone="Nowhere/Here")
    with pytest.raises(CytometerConfigurationException):
        CytometerConfiguration.from_exported(bad_tz)
    bad_laser = dict(CytometerConfiguration(["PE-A"]).to_dictionary(), laserdict={"a": ["PE-A"], "b": ["PE-A"]})
    with pytest.raises(CytometerConfigurationException):
        CytometerConfiguration.from_exported(bad_laser)


def test_import_exported_into_instance():
    target = CytometerConfiguration(["X-A"])
    target.import_exported(CytometerConfiguration(["PE-A", "APC-A"], cytometer="mine",
                                                  laserdict={"Red": ["APC-A"]}).to_dictionary())
    assert target.channels_fluorescence == ["PE-A", "APC-A"] and target.cytometer == "mine"
    assert target.channel_laser_key["APC-A"] == "Red"  # rebuilt from the imported laserdict


def test_from_exported_subclass_init_not_run(monkeypatch):
    ex = CytometerConfiguration_A3().to_dictionary()
    monkeypatch.setattr(CytometerConfiguration_A3, "__init__", lambda self: (_ for _ in ()).throw(AssertionError))
    r = CytometerConfiguration.from_exported(ex)
    assert type(r) is CytometerConfiguration_A3 and r.alt_name_dct["BB515-A"] == "B1 (515)-A"


def test_load_json_via_subclass(tmp_path):
    """R2-F2: load_json on a subclass works (was cls([]))."""
    CytometerConfiguration_DxFLEX1L().write_json(tmp_path / "d.json")
    r = CytometerConfiguration_A3.load_json(tmp_path / "d.json")
    assert type(r) is CytometerConfiguration_DxFLEX1L


# ------------------------------------------------------------------ plot_height_clipping
def test_plot_height_clipping(config):
    ch = ["FL1-A", "FL1-H", "FL2-A"]
    a = anndata.AnnData(np.zeros((50, 3), np.float32), var=pd.DataFrame(index=ch))
    a.layers["raw"] = np.random.rand(50, 3).astype(np.float32) * 1e5
    with warnings.catch_warnings():
        warnings.simplefilter("error", CytovanniWarning)
        assert config.plot_height_clipping(a, logscale=True) is None
    ax = plt.gca()
    assert ax.get_xscale() == "log" and len(ax.collections) == 1  # only FL1 has a height channel
    assert any(np.allclose(l.get_xdata(), config.max_linear_range) for l in ax.get_lines())


def test_plot_height_clipping_warns_without_height(config):
    a = anndata.AnnData(np.zeros((5, 1), np.float32), var=pd.DataFrame(index=["FL1-A"]))
    a.layers["raw"] = a.X.copy()
    with pytest.warns(CytovanniWarning, match="height"):
        config.plot_height_clipping(a)


# ------------------------------------------------------------------ instrument subclasses
EXPECTED = {CytometerConfiguration_A3: ("A3", 23, 2e5), CytometerConfiguration_A3small: ("A3", 21, 2e5),
            CytometerConfiguration_S6: ("S6", 48, 2e5), CytometerConfiguration_S8: ("S8", 78, 4e6),
            CytometerConfiguration_FACSAriaIII: ("FACSAriaIII", 18, 2e5),
            CytometerConfiguration_FACSAriaIII_v2: ("FACSAriaIII_v2", 18, 2e5),
            CytometerConfiguration_FACSAriaFusion: ("FACSAriaFusion", 16, 2e5),
            CytometerConfiguration_FACSAriaFusion_v2: ("FACSAriaFusion_v2", 18, 2e5),
            CytometerConfiguration_DxFLEX1L: ("DxFLEX1L", 5, 2e6), CytometerConfiguration_Aurora: ("Aurora", 64, 4e6),
            CytometerConfiguration_NorthernLights: ("NorthernLights", 38, 4e6)}


@pytest.mark.parametrize("klass", SUBCLASSES)
def test_instrument_configurations_are_consistent(klass):
    cfg = klass()
    name, n_fl, lin = EXPECTED[klass]
    assert cfg.cytometer == name and len(cfg.channels_fluorescence) == n_fl and cfg.max_linear_range == lin
    assert len(set(cfg.channels_fluorescence)) == n_fl
    cfg.get_default_scatter_gate_channels()
    cfg.get_default_scatter_calibration()
    # every fluorescence channel has a laser
    assert (cfg.channel_laser_key.loc[cfg.channels_fluorescence] != "unknown").all()
    # renaming maps onto configured channel names (-A/-H/-W)
    known = set(cfg.channels_fluorescence) | set(cfg.channels_fluorescence_addHW) | set(cfg.channels_scatter)
    assert set(cfg.replace_channelname_dict.values()) <= known


def test_A3_alt_names():
    cfg = CytometerConfiguration_A3()
    assert cfg.alt_name_dct["BB515-A"] == "B1 (515)-A" and cfg.alt_name_dct["PE-CF594-W"] == "YG2 (610)-W"
    assert len(cfg.alt_name_dct) == 3 * 23
    assert not hasattr(CytometerConfiguration_A3small(), "alt_name_dct")
    assert set(CytometerConfiguration_A3().channels_fluorescence) - set(CytometerConfiguration_A3small().channels_fluorescence) == {"BUV737-A", "BV650-A"}


def test_S8_scatter_lasers():
    """R3b-B11: FSC and SSC (Imaging) are Blue, LightLoss (Violet) Violet, LightLoss (Imaging) unassigned."""
    k = CytometerConfiguration_S8(include_imaging_based=True).channel_laser_key
    assert k["FSC-A"] == "Blue" and k["SSC (Imaging)-A"] == "Blue" and k["LightLoss (Violet)-A"] == "Violet"
    assert k["LightLoss (Imaging)-A"] == "unknown" and "Imaging" not in set(k.values)


def test_S8_imaging_based_channels():
    small, big = CytometerConfiguration_S8(), CytometerConfiguration_S8(include_imaging_based=True)
    assert len(small.channels_scatter) == 9 and "LightLoss (Violet)-A" not in small.channels_scatter
    assert set(small.channels_scatter) < set(big.channels_scatter) and "Eccentricity (FSC)" in big.channels_scatter
    assert small.default_scatter_gate_channels == ["LightLoss (Imaging)-A", "SSC (Imaging)-A"]
    assert set(small.default_scatter_calibrate) == {"FSC-A", "SSC (Imaging)-A"}


@pytest.mark.parametrize("klass", [CytometerConfiguration_FACSAriaIII, CytometerConfiguration_FACSAriaIII_v2,
                                   CytometerConfiguration_FACSAriaFusion, CytometerConfiguration_FACSAriaFusion_v2])
def test_FACSAria_lasers_and_renaming(klass):
    cfg = klass()
    assert list(cfg.lasers) == list(cfg.lasers_wavelength.index) == list(cfg.lasers_color.index)
    assert set(cfg.lasers) == set(cfg.laserdict)
    assert all(c.startswith("#") for c in cfg.lasers_color)
    assert cfg.replace_channelname_dict["BL-B-A"] == ("B1 (525)-A" if "III" in klass.__name__ else "B1 (530)-A")
    assert all(k[-2:] == v[-2:] and k[-2:] in ("-A", "-H", "-W") for k, v in cfg.replace_channelname_dict.items())
    assert (cfg.channel_laser_key.loc[cfg.channels_scatter] == "unknown").all()


def test_add_AHW_replacedct():
    assert C._add_AHW_replacedct({"X": "Y"}) == {"X-A": "Y-A", "X-H": "Y-H", "X-W": "Y-W"}


def test_cytek_scatter_channels():
    for klass in (CytometerConfiguration_Aurora, CytometerConfiguration_NorthernLights):
        cfg = klass()
        assert "SSC-B-A" in cfg.channels_scatter and set(cfg.default_scatter_calibrate) == {"FSC-A", "SSC-A", "SSC-B-A"}
        assert cfg.channel_laser_key["SSC-B-A"] == "Blue" and cfg.channel_laser_key["FSC-A"] == "Violet"
    assert CytometerConfiguration_DxFLEX1L().channels_scatter == ["FSC-A", "FSC-H", "SSC-A", "SSC-H"]
