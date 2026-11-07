"""Reading and writing .fcs files (cytovanni.io.utils): samples, metadata, spillover, time stamps, repairs."""
import hashlib
import inspect
import os
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.exceptions import (ChannelMismatchException, ChannelMismatchWarning, RainbowSpilloverWarning,
                                  SettingsHashException)
from cytovanni.gating import MinMaxGate
from cytovanni.io import (read_folder_timestamps, readfcs_autospectral_spill, readfcs_available_channels,
                          readfcs_dataframe, readfcs_metadata, readfcs_sample, readfcs_spill, sample_has_nonzero_spill,
                          writefcs)
from cytovanni.io import utils as iou
from cytovanni.io.utils import (_spectra_from_autospectralstr, fix_A3_HW_bug, fix_sample_bug, read_fcs_timestamp,
                                readfcs_channel_settings_hash)
from cytovanni.utils import CytometerConfiguration


def utc(date_time, tz="Europe/Berlin"):
    return datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo(tz)).timestamp()


def small_df(n=20, columns=("FSC-A", "FSC-H", "SSC-A", "SSC-H", "FL1-A", "FL2-A", "FL3-A", "Time"), seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(100, 1000, (n, len(columns))), columns=list(columns)).astype(np.float32)


def write(tmp_path, name, df=None, meta=None, **kw):
    p = str(tmp_path / name)
    writefcs(p, small_df() if df is None else df, add_metadata=meta, **kw)
    return p


@pytest.fixture
def cell_a(cell_files):
    return cell_files["A"]


# ------------------------------------------------------------------------------------------------ readfcs_sample
def test_readfcs_sample_layout(cell_a, config):
    a = readfcs_sample(cell_a, config)
    expected_channels = synthetic.CHANNELS_SCATTER + synthetic.CHANNELS_FLUORESCENCE
    assert list(a.var_names) == expected_channels
    assert a.X is None and list(a.layers) == ["raw"]
    assert list(a.var.columns) == ["pnn", "pns", "png", "pnv", "pnr"]
    assert list(a.obs.columns) == ["Time"] and a.obs_names[0] == "0" and isinstance(a.obs_names[0], str)
    assert a.var.loc["FL2-A", "pnv"] == synthetic.VOLTAGES.loc["A", "FL2-A"]
    df, _ = synthetic.cell_events("A", seed=20, abundances=synthetic.cell_abundances(4000, np.random.default_rng(1)))
    assert np.allclose(a.to_df(layer="raw").to_numpy(), df[expected_channels].to_numpy())
    assert np.allclose(a.obs["Time"].to_numpy(), df["Time"].to_numpy())
    assert set(a.uns) == {"date", "UTC", "cytometer", "uid", "cytoconfig", "set_hash", "Nevents"}
    assert a.uns["date"] == "01-JAN-2024" and a.uns["cytometer"] == "SIM" and a.uns["Nevents"] == 4000
    assert a.uns["UTC"] == utc("2024-01-01 10:30:00")
    assert a.uns["uid"] == f"SIM-{int(a.uns['UTC'])}"
    assert a.uns["cytoconfig"] is config
    assert a.uns["set_hash"] == config.get_settings_hash(a) and isinstance(a.uns["set_hash"], str)


def test_readfcs_sample_hash_depends_on_voltages(rainbow_files, cell_files, config):
    """Files of the same batch share the settings hash, other batches (other voltages) differ."""
    h = {b: readfcs_sample(p, config).uns["set_hash"] for b, p in cell_files.items()}
    assert len(set(h.values())) == 3
    assert readfcs_sample(rainbow_files["B"][1], config).uns["set_hash"] == h["B"]


def test_readfcs_sample_metadct_and_no_shared_defaults(cell_a, config):
    """R2-E5: the io functions use None defaults; metadct entries go to uns and do not leak into later calls."""
    for f in (readfcs_sample, writefcs):
        for name, par in inspect.signature(f).parameters.items():
            assert not isinstance(par.default, (dict, list, set)), (f.__name__, name)
    from cytovanni.io import readfcs_rainbow_sample
    assert inspect.signature(readfcs_rainbow_sample).parameters["metadct"].default is None
    a = readfcs_sample(cell_a, config, metadct={"patient": "P01", "date": "overwritten"})
    assert a.uns["patient"] == "P01" and a.uns["date"] == "overwritten"
    b = readfcs_sample(cell_a, config)
    assert "patient" not in b.uns and b.uns["date"] == "01-JAN-2024"


def test_readfcs_sample_gates(cell_a, config, capsys):
    full = readfcs_sample(cell_a, config)
    gate = MinMaxGate(["FL1-A"], -1e9, float(np.median(full.to_df(layer="raw")["FL1-A"])))
    a = readfcs_sample(cell_a, config, gates=[gate, MinMaxGate(["FSC-A"], 0, 1e9)], print_gates=True)
    keep = gate.apply(full)
    assert a.n_obs == keep.sum() and a.uns["Nevents"] == full.n_obs  # Nevents counts before the gates
    assert np.allclose(a.to_df(layer="raw").to_numpy(), full[keep].to_df(layer="raw").to_numpy())
    assert "dropped" in capsys.readouterr().out
    readfcs_sample(cell_a, config, gates=[gate])
    assert capsys.readouterr().out == ""


def test_readfcs_sample_renames_channels(tmp_path, config):
    df = small_df(columns=["FSC-A", "FSC-H", "SSC-A", "SSC-H", "B1 (525)-A", "FL2-A", "FL3-A", "Time"])
    p = write(tmp_path, "renamed.fcs", df, meta={"date": "01-JAN-2024", "btim": "10:00:00"})
    cfg = synthetic.make_config(replace_channelname_dict={"B1 (525)-A": "FL1-A"})
    a = readfcs_sample(p, cfg)
    assert "FL1-A" in a.var_names and a.var.loc["FL1-A", "pnn"] == "FL1-A"
    assert np.allclose(a[:, "FL1-A"].layers["raw"][:, 0], df["B1 (525)-A"])
    assert readfcs_channel_settings_hash(p, cfg) == a.uns["set_hash"]
    assert "B1 (525)-A" in readfcs_available_channels(p)  # as stored in the file


def test_readfcs_sample_optional_HW_channels(tmp_path, config):
    df = small_df(columns=["FSC-A", "FSC-H", "SSC-A", "SSC-H", "FL1-A", "FL1-H", "FL2-A", "FL3-A", "FL3-W", "Time"])
    a = readfcs_sample(write(tmp_path, "hw.fcs", df), config)
    assert list(a.var_names) == synthetic.CHANNELS_SCATTER + synthetic.CHANNELS_FLUORESCENCE + ["FL1-H", "FL3-W"]


def test_readfcs_sample_fct_fix_df(tmp_path, config):
    """`fct_fix_df` changes the raw events before the AnnData is built; the channel checks see its output."""
    df = small_df()
    p = write(tmp_path, "fix.fcs", df)

    def swap(d):
        d["FL1-A"], d["FL2-A"] = d["FL2-A"].copy(), d["FL1-A"].copy()
        return d
    a = readfcs_sample(p, config, fct_fix_df=swap)
    assert np.allclose(a[:, "FL1-A"].layers["raw"][:, 0], df["FL2-A"])
    assert np.allclose(a[:, "FL2-A"].layers["raw"][:, 0], df["FL1-A"])
    with pytest.raises(ChannelMismatchException, match="FL3-A"):
        readfcs_sample(p, config, fct_fix_df=lambda d: d.drop(columns=["FL3-A"]))


def test_readfcs_sample_missing_fluorescence_raises(tmp_path, config):
    p = write(tmp_path, "nofl.fcs", small_df(columns=["FSC-A", "FSC-H", "SSC-A", "SSC-H", "FL1-A", "FL2-A", "Time"]))
    with pytest.raises(ChannelMismatchException, match="FL3-A"):
        readfcs_sample(p, config)


def test_readfcs_sample_missing_meta_channel_raises(tmp_path, config):
    """R2b-G2-17: a missing meta channel raises ChannelMismatchException listing the available channels."""
    p = write(tmp_path, "notime.fcs", small_df(columns=synthetic.CHANNELS_SCATTER + synthetic.CHANNELS_FLUORESCENCE))
    with pytest.raises(ChannelMismatchException) as e:
        readfcs_sample(p, config)
    assert "Time" in str(e.value) and "FL1-A" in str(e.value)


def test_readfcs_sample_missing_scatter(tmp_path, config):
    df = small_df(columns=["FSC-A", "SSC-A", "SSC-H", "FL1-A", "FL2-A", "FL3-A", "Time"])
    p = write(tmp_path, "noscatter.fcs", df)
    with pytest.warns(ChannelMismatchWarning, match="FSC-H"):
        a = readfcs_sample(p, config)
    assert list(a.var_names) == ["FSC-A", "SSC-A", "SSC-H"] + synthetic.CHANNELS_FLUORESCENCE
    assert a.uns["set_hash"].startswith("red_")
    with pytest.raises(ChannelMismatchException, match="FSC-H"):
        readfcs_sample(p, config, allow_missing_scatter=False)


SPILL_NONTRIVIAL = "3,FL1-A,FL2-A,FL3-A,1,0.1,0,0.2,1,0,0,0,1"
SPILL_IDENTITY = "3,FL1-A,FL2-A,FL3-A,1,0,0,0,1,0,0,0,1"


def test_readfcs_sample_warn_spillover(tmp_path, config):
    p = write(tmp_path, "spill.fcs", meta={"spill": SPILL_NONTRIVIAL})
    with pytest.warns(RainbowSpilloverWarning, match="custom text"):
        readfcs_sample(p, config, warn_spillover=True, warn_spillover_message="custom text")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        readfcs_sample(p, config)  # default: no warning
        readfcs_sample(write(tmp_path, "id.fcs", meta={"spill": SPILL_IDENTITY}), config, warn_spillover=True)
    assert not [x for x in w if issubclass(x.category, RainbowSpilloverWarning)]


def test_readfcs_sample_without_date(tmp_path, config):
    p = write(tmp_path, "nodate.fcs")
    a = readfcs_sample(p, config)
    assert a.uns["date"] is None and np.isnan(a.uns["UTC"])
    assert a.uns["uid"] == "SIM-" + hashlib.sha256(p.encode("utf-8")).hexdigest()


def test_readfcs_sample_unparsable_date(tmp_path, config):
    p = write(tmp_path, "baddate.fcs", meta={"date": "sometime", "btim": "10:00:00"})
    a = readfcs_sample(p, config)
    assert a.uns["date"] == "sometime" and np.isnan(a.uns["UTC"])
    assert a.uns["uid"].startswith("SIM-") and len(a.uns["uid"]) == 4 + 64


def test_readfcs_sample_date_harmonised(tmp_path, config):
    p = write(tmp_path, "date.fcs", meta={"date": "05-Mar-2024", "btim": "08:15:30"})
    a = readfcs_sample(p, config)
    assert a.uns["date"] == "05-MAR-2024" and a.uns["UTC"] == utc("2024-03-05 08:15:30")


# ------------------------------------------------------------------------------------------------ time zone (F1)
@pytest.fixture
def machine_tz(monkeypatch):
    """Set the time zone of the reading process (TZ + tzset) and restore it afterwards."""
    def set_tz(tz):
        monkeypatch.setenv("TZ", tz)
        time.tzset()
    yield set_tz
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize("posix_tz", ["UTC0", "JST-9", "EST5EDT,M3.2.0,M11.1.0"])
def test_utc_independent_of_machine_timezone(tmp_path, config, machine_tz, posix_tz):
    """R2-F1: $DATE/$BTIM are read in the instrument time zone (config.timezone), not that of the reading machine."""
    pw = write(tmp_path, "win.fcs", meta={"date": "01-JAN-2025", "btim": "12:00:00", "etim": "12:05:00"})
    ps = write(tmp_path, "sum.fcs", meta={"date": "01-JUL-2025", "btim": "12:00:00"})
    machine_tz(posix_tz)
    w, s = readfcs_sample(pw, config), readfcs_sample(ps, config)
    assert w.uns["UTC"] == 1735729200.0 and s.uns["UTC"] == 1751364000.0  # 11:00Z (CET), 10:00Z (CEST)
    assert w.uns["uid"] == "SIM-1735729200"
    assert read_fcs_timestamp(pw)["UTC"] == 1735729200.0
    assert read_folder_timestamps(str(tmp_path))["UTC"].tolist() == [1735729200.0, 1751364000.0]


def test_utc_follows_config_timezone(tmp_path):
    pw = write(tmp_path, "win.fcs", meta={"date": "01-JAN-2025", "btim": "12:00:00"})
    tokyo = synthetic.make_config(timezone="Asia/Tokyo")
    assert readfcs_sample(pw, tokyo).uns["UTC"] == 1735729200.0 - 8 * 3600
    assert read_fcs_timestamp(pw, timezone="Asia/Tokyo")["UTC"] == 1735729200.0 - 8 * 3600
    assert read_folder_timestamps(str(tmp_path), cytoconfig=tokyo)["UTC"].iloc[0] == 1735729200.0 - 8 * 3600
    assert read_folder_timestamps(str(tmp_path), cytoconfig=tokyo, timezone="UTC")["UTC"].iloc[0] == utc("2025-01-01 12:00:00", "UTC")


def test_fcs_utc_explicit_offset_takes_precedence():
    assert iou._fcs_utc("01-JAN-2025", "12:00:00+00:00", "Asia/Tokyo") == utc("2025-01-01 12:00:00", "UTC")


# ------------------------------------------------------------------------------------------------ time stamps
def test_read_fcs_timestamp(tmp_path):
    p = write(tmp_path, "t.fcs", meta={"date": "05-Jan-2024", "btim": "10:00:00", "etim": "10:20:00"})
    t = read_fcs_timestamp(p)
    assert list(t.index) == ["date", "begin_record", "end_record", "UTC"]
    assert t["date"] == "05-Jan-2024" and t["begin_record"] == "10:00:00" and t["end_record"] == "10:20:00"
    assert t["UTC"] == utc("2024-01-05 10:00:00")
    t0 = read_fcs_timestamp(write(tmp_path, "none.fcs"))
    assert t0["date"] == "" and np.isnan(t0["begin_record"]) and np.isnan(t0["end_record"]) and np.isnan(t0["UTC"])
    with pytest.raises(ValueError):
        read_fcs_timestamp(write(tmp_path, "bad.fcs", meta={"date": "garbage", "btim": "xx"}))


def test_read_folder_timestamps_sorted_chronologically_no_utc_last(tmp_path):
    """R3b-B2: sorted by UTC (not by the $DATE string); files without $DATE/$BTIM come last."""
    for name, date in [("x.fcs", "05-JAN-2024"), ("y.fcs", "01-FEB-2023"), ("z.fcs", None), ("w.fcs", "01-FEB-2024")]:
        write(tmp_path, name, meta={"date": date, "btim": "10:00:00"} if date else {})
    (tmp_path / "notes.txt").write_text("not an fcs file")
    tps = read_folder_timestamps(str(tmp_path))
    assert tps.index.tolist() == ["y.fcs", "x.fcs", "w.fcs", "z.fcs"]
    assert list(tps.columns) == ["date", "begin_record", "end_record", "UTC"]


def test_read_folder_timestamps_empty_folder(tmp_path):
    """R2-E9: a folder without .fcs files gives an empty frame with the columns (no KeyError)."""
    tps = read_folder_timestamps(str(tmp_path))
    assert tps.shape[0] == 0 and "UTC" in tps.columns


def test_read_folder_timestamps_with_hash(rainbow_files, config, tmp_path):
    folder = os.path.dirname(rainbow_files["A"][0])
    tps = read_folder_timestamps(folder, cytoconfig=config)
    assert "set_hash" in tps.columns and len(tps) == 6
    for b, paths in rainbow_files.items():
        for p in paths:
            assert tps.loc[os.path.basename(p), "set_hash"] == readfcs_sample(p, config).uns["set_hash"]
    assert tps["UTC"].is_monotonic_increasing
    # a hash function that fails gives NaN instead of an error
    bad = synthetic.make_config(fct_hash_settings=lambda ad: 1 / 0)
    assert read_folder_timestamps(folder, cytoconfig=bad)["set_hash"].isna().all()


def test_read_folder_timestamps_unparsable_raises(tmp_path):
    write(tmp_path, "bad.fcs", meta={"date": "garbage", "btim": "xx"})
    with pytest.raises(ValueError):
        read_folder_timestamps(str(tmp_path))


# ------------------------------------------------------------------------------------------------ settings hash
def test_channel_settings_hash_equals_sample_hash(cell_files, config):
    for p in cell_files.values():
        assert readfcs_channel_settings_hash(p, config) == readfcs_sample(p, config).uns["set_hash"]


def test_channel_settings_hash_raises(cell_a):
    cfg = synthetic.make_config(fct_hash_settings=lambda ad: ad.uns["missing"])
    with pytest.raises(SettingsHashException):
        readfcs_channel_settings_hash(cell_a, cfg)
    with pytest.raises(SettingsHashException):
        readfcs_sample(cell_a, cfg)


# ------------------------------------------------------------------------------------------------ metadata, channels
def test_readfcs_metadata_lowercase_keys(cell_a):
    md = readfcs_metadata(cell_a)
    assert isinstance(md, dict) and md["date"] == "01-JAN-2024" and md["btim"] == "10:30:00"
    assert md["p1n"] == "FSC-A" and all(k == k.lower() and not k.startswith("$") for k in md)


@pytest.mark.parametrize("simple", [True, False])
def test_readfcs_available_channels(cell_a, simple):
    ch = readfcs_available_channels(cell_a, simple_flowio=simple)
    assert isinstance(ch, np.ndarray) and ch.tolist() == synthetic.CHANNELS


def test_readfcs_dataframe_uses_pns_where_set(tmp_path):
    df = small_df(columns=["FSC-A", "FL1-A", "FL2-A"])
    p = write(tmp_path, "pns.fcs", df, pns=["fsc", "CD3", ""])
    out = readfcs_dataframe(p)
    assert out.columns.tolist() == ["fsc", "CD3", "FL2-A"]
    assert np.allclose(out.to_numpy(), df.to_numpy())
    assert readfcs_dataframe(write(tmp_path, "plain.fcs", df)).columns.tolist() == ["FSC-A", "FL1-A", "FL2-A"]


# ------------------------------------------------------------------------------------------------ writefcs
def test_writefcs_round_trip_creates_folders(tmp_path):
    df = small_df()
    p = str(tmp_path / "a" / "b" / "x.fcs")
    writefcs(p, df, add_metadata={"note": "calibrated", "date": "01-JAN-2024"})
    md = readfcs_metadata(p)
    assert md["note"] == "calibrated" and md["date"] == "01-JAN-2024"
    assert md["p1s"] == "FSC-A"  # default $PnS = $PnN
    back = readfcs_dataframe(p)
    assert back.columns.tolist() == df.columns.tolist() and np.allclose(back.to_numpy(), df.to_numpy())


def test_writefcs_pns_options(tmp_path):
    df = small_df(columns=["FSC-A", "FL1-A"])
    md = readfcs_metadata(write(tmp_path, "p.fcs", df, pns=["fsc", "CD3"]))
    assert md["p1s"] == "fsc" and md["p2s"] == "CD3"
    md = readfcs_metadata(write(tmp_path, "q.fcs", df, overwrite_pns=False))
    assert "p2s" not in md


# ------------------------------------------------------------------------------------------------ spillover
def test_readfcs_spill(tmp_path):
    s = readfcs_spill(write(tmp_path, "s.fcs", meta={"spill": SPILL_NONTRIVIAL}))
    assert s.columns.tolist() == ["FL1-A", "FL2-A", "FL3-A"] and s.index.tolist() == [0, 1, 2]
    assert np.allclose(s.to_numpy(), [[1, .1, 0], [.2, 1, 0], [0, 0, 1]])
    s2 = readfcs_spill(write(tmp_path, "s2.fcs", meta={"spillover": SPILL_NONTRIVIAL}))
    assert s2.equals(s)
    assert readfcs_spill(write(tmp_path, "none.fcs")) is None


@pytest.mark.parametrize("manual", [False, True])
def test_sample_has_nonzero_spill(tmp_path, manual):
    import flowkit as fk
    load = lambda name, meta: fk.Sample(write(tmp_path, name, meta=meta), preprocess=False)
    assert sample_has_nonzero_spill(load("a.fcs", {"spill": SPILL_NONTRIVIAL}), manual=manual)
    assert sample_has_nonzero_spill(load("b.fcs", {"spillover": SPILL_NONTRIVIAL}), manual=manual)
    assert not sample_has_nonzero_spill(load("c.fcs", {"spill": SPILL_IDENTITY}), manual=manual)
    assert not sample_has_nonzero_spill(load("d.fcs", {}), manual=manual)


def test_readfcs_autospectral_spill(tmp_path):
    s = "2,3,FITC,PE,FL1-A,FL2-A,FL3-A,1,0.1,0.05,0.2,1,0.3"
    m = readfcs_autospectral_spill(write(tmp_path, "as.fcs", meta={"spectra": s}))
    assert m.index.tolist() == ["FITC", "PE"] and m.columns.tolist() == ["FL1-A", "FL2-A", "FL3-A"]
    assert np.allclose(m.to_numpy(), [[1, .1, .05], [.2, 1, .3]])
    assert readfcs_autospectral_spill(write(tmp_path, "no.fcs")) is None


def test_spectra_from_autospectralstr_size_mismatch():
    assert _spectra_from_autospectralstr(" 1, 1, a , b, 2.5, ").loc["a", "b"] == 2.5
    with pytest.raises(ValueError):
        _spectra_from_autospectralstr("2,2,a,b,c,d,1,2,3")


# ------------------------------------------------------------------------------------------------ repairs
A3_COLUMNS = ["Time", "FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]


def test_fix_A3_HW_bug_swaps_only_if_time_first():
    df = small_df(columns=A3_COLUMNS)
    orig = df.copy()
    out = fix_A3_HW_bug(df)
    assert out is df
    assert np.array_equal(out["FSC-H"], orig["FSC-W"]) and np.array_equal(out["FSC-W"], orig["FSC-H"])
    assert np.array_equal(out["SSC-H"], orig["SSC-W"]) and np.array_equal(out["SSC-W"], orig["SSC-H"])
    assert np.array_equal(out["FSC-A"], orig["FSC-A"])
    df2 = small_df(columns=A3_COLUMNS[1:] + ["Time"])
    assert fix_A3_HW_bug(df2.copy()).equals(df2)


def test_fix_sample_bug_applies_fix(tmp_path):
    df = small_df(columns=A3_COLUMNS)
    p = write(tmp_path, "a3.fcs", df, meta={"date": "01-JAN-2024", "btim": "10:00:00"})
    new = str(tmp_path / "fixed" / "a3.fcs")
    fix_sample_bug(p, new, fix_A3_HW_bug)
    out = readfcs_dataframe(new)
    assert np.allclose(out["FSC-H"], df["FSC-W"]) and np.allclose(out["SSC-W"], df["SSC-H"])
    md = readfcs_metadata(new)
    assert md["date"] == "01-JAN-2024" and md["btim"] == "10:00:00"


@pytest.mark.parametrize("key", ["spill", "spillover"])
def test_fix_sample_bug_deletes_both_spill_keywords(tmp_path, key):
    """R2-E18: delete_spill=True removes $SPILLOVER as well as $SPILL; False keeps them."""
    df = small_df(columns=["FL1-A", "FL2-A", "FL3-A"])
    p = write(tmp_path, "s.fcs", df, meta={key: SPILL_NONTRIVIAL})
    fix_sample_bug(p, str(tmp_path / "del.fcs"), lambda d: d)
    md = readfcs_metadata(str(tmp_path / "del.fcs"))
    assert "spill" not in md and "spillover" not in md
    assert readfcs_spill(str(tmp_path / "del.fcs")) is None
    fix_sample_bug(p, str(tmp_path / "keep.fcs"), lambda d: d, delete_spill=False)
    assert readfcs_spill(str(tmp_path / "keep.fcs")) is not None


def test_fix_sample_bug_keeps_pns(tmp_path):
    """R3b-B1: the repaired file keeps the $PnS labels of the original ("" where there was none)."""
    df = small_df(columns=["FSC-A", "FL1-A", "FL2-A"])
    p = write(tmp_path, "a.fcs", df, pns=["fsc", "CD3", ""], meta={"spill": "1,FL1-A,1"})
    new = str(tmp_path / "fixed" / "a.fcs")
    fix_sample_bug(p, new, lambda d: d)
    md = readfcs_metadata(new)
    assert md.get("p1s") == "fsc" and md.get("p2s") == "CD3" and not md.get("p3s")
    assert readfcs_dataframe(new).columns.tolist() == ["fsc", "CD3", "FL2-A"]
