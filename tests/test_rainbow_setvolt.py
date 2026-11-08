"""cytovanni.rainbow.setvolt: target intensities from a reference rainbow measurement and the voltages reaching them.

The reference is a rainbow file of batch A, the new measurement one of batch B (detector gains `synthetic.GAINS`,
scatter gain `synthetic.SCATTER_GAINS`, voltages `synthetic.VOLTAGES`).
"""
import numpy as np
import pandas as pd
import pytest

from cytovanni.exceptions import CytovanniWarning
from cytovanni.rainbow import get_config_from_rainbow_reference, get_voltages_from_rainbow, color_voltageconfig
from cytovanni.utils import textcolor

import synthetic

FL = synthetic.CHANNELS_FLUORESCENCE
NPEAK = len(synthetic.PEAKS)
SET = ["FSC-A", "SSC-A"] + FL


@pytest.fixture(scope="module")
def targetconfig(rainbow_files, config):
    return get_config_from_rainbow_reference(config, rainbow_files["A"][0], SET, NPEAK, plot=False)


def test_target_table_layout(targetconfig):
    assert list(targetconfig.index) == SET + ["_N_peak"]
    assert list(targetconfig.columns) == ["target", "peak", "exponent"]
    assert str(targetconfig["target"].dtype) == "Int64"
    assert pd.isna(targetconfig.loc["FSC-A", "peak"]) and pd.isna(targetconfig.loc["SSC-A", "peak"])
    assert targetconfig.loc["_N_peak", "target"] == 0 and targetconfig.loc["_N_peak", "peak"] == NPEAK
    assert pd.isna(targetconfig.loc["_N_peak", "exponent"])
    assert (targetconfig.loc[SET, "exponent"] == 7.).all()


def test_targets_are_brightest_linear_peak_and_scatter_median(targetconfig):
    # all peaks are below 0.75 * max_linear_range (1.5e5): the brightest peak (3, 1-based) is the target
    assert (targetconfig.loc[FL, "peak"] == NPEAK).all()
    np.testing.assert_allclose(targetconfig.loc[FL, "target"].astype(float), synthetic.PEAKS[-1], rtol=.03)
    # scatter: median of the gated beads (FSC 1e4, SSC 5e3 in batch A)
    np.testing.assert_allclose(targetconfig.loc[["FSC-A", "SSC-A"], "target"].astype(float), [1e4, 5e3], rtol=.03)


def test_use_highest_peak_false_and_series_exponent(rainbow_files, config):
    exps = pd.Series([5., 6., 7., 8., 9.], index=SET)
    tc = get_config_from_rainbow_reference(config, rainbow_files["A"][0], SET, NPEAK, use_highest_peak=False,
                                           PMT_exponent=exps, plot=False)
    assert (tc.loc[FL, "peak"] == NPEAK - 1).all()
    np.testing.assert_allclose(tc.loc[FL, "target"].astype(float), synthetic.PEAKS[-2], rtol=.03)
    pd.testing.assert_series_equal(tc.loc[SET, "exponent"].astype(float), exps, check_names=False)


def test_safety_factor_excludes_peaks_above_linear_limit(rainbow_files, config):
    tc = get_config_from_rainbow_reference(config, rainbow_files["A"][0], FL, NPEAK, safety_factor_linear=.1,
                                           plot=False)     # limit 2e4: the 4e4 peak is excluded
    assert (tc.loc[FL, "peak"] == 2).all()


def test_unknown_channel_raises(rainbow_files, config):
    with pytest.raises(Exception, match="Could not find channels"):
        get_config_from_rainbow_reference(config, rainbow_files["A"][0], ["FL1-A", "nope"], NPEAK, plot=False)


def test_voltages_recover_gain_ratio(targetconfig, rainbow_files, config):
    vc = get_voltages_from_rainbow(config, targetconfig, rainbow_files["B"][0], plot=False)
    assert "_N_peak" not in vc.index and list(vc.index) == SET
    for col in ["value", "voltage", "factor", "error (%)", "voltage_target"]:
        assert col in vc.columns
    gain = pd.concat([pd.Series(synthetic.SCATTER_GAINS["B"], index=["FSC-A", "SSC-A"]), synthetic.GAINS.loc["B"]])
    np.testing.assert_allclose(vc["error (%)"], (gain - 1) * 100, atol=3)
    np.testing.assert_allclose(vc["voltage"], [250, 250] + synthetic.VOLTAGES.loc["B"].tolist())
    np.testing.assert_allclose(vc["factor"], (1 / gain) ** (1 / 7.), rtol=.005)
    np.testing.assert_allclose(vc["voltage_target"].astype(float), np.round(vc["voltage"] * vc["factor"]), atol=1)
    # documented formulas on the reported (rounded) numbers
    np.testing.assert_allclose(vc["error (%)"],
                               np.round((vc["value"] - vc["target"].astype(float)) / vc["target"].astype(float) * 100, 1))


def test_voltages_of_the_reference_itself_need_no_change(targetconfig, rainbow_files, config):
    vc = get_voltages_from_rainbow(config, targetconfig, rainbow_files["A"][0], plot=False)
    assert (vc["error (%)"].abs() < .5).all()
    np.testing.assert_allclose(vc["voltage_target"].astype(float), vc["voltage"], atol=1)


def test_voltages_from_csv_round_trip(targetconfig, rainbow_files, config, tmp_path):
    path = tmp_path / "targetconfig.csv"
    targetconfig.to_csv(path)
    tc = pd.read_csv(path, index_col=0)
    vc = get_voltages_from_rainbow(config, tc, rainbow_files["B"][0], plot=False)
    ref = get_voltages_from_rainbow(config, targetconfig, rainbow_files["B"][0], plot=False)
    np.testing.assert_allclose(vc["voltage_target"].astype(float), ref["voltage_target"].astype(float))


def test_saturated_channels_warn_and_give_nan_R3b_B6(rainbow_files):
    """R3b-B6: a channel without a peak below the linear limit warns (CytovanniWarning), gets target NaN, and the
    voltage step gives NaN instead of voltage 0 / error inf; the other channels are unaffected."""
    config = synthetic.make_config(max_linear_range=100.)   # every peak is above 75
    with pytest.warns(CytovanniWarning, match="No rainbow peak below the linear limit"):
        tc = get_config_from_rainbow_reference(config, rainbow_files["A"][0], SET, NPEAK, plot=False)
    assert tc.loc[FL, "target"].isna().all() and (tc.loc[FL, "peak"] == 1).all()
    assert tc.loc[["FSC-A", "SSC-A"], "target"].notna().all()
    vc = get_voltages_from_rainbow(config, tc, rainbow_files["B"][0], plot=False)
    assert vc.loc[FL, "voltage_target"].isna().all()
    assert vc.loc[["FSC-A", "SSC-A"], "voltage_target"].notna().all()
    assert not np.isinf(vc["error (%)"].astype(float)).any()


def test_color_voltageconfig(targetconfig, rainbow_files, config):
    vc = get_voltages_from_rainbow(config, targetconfig, rainbow_files["B"][0], plot=False)
    vc.loc["FL3-A", "error (%)"] = .5                     # below the error level
    out = color_voltageconfig(vc, errorlevel=1)
    assert list(out.columns) == ["target", "peak", "voltage", "value", "error (%)", "voltage_target", ""]
    red, cyan = textcolor.RED, textcolor.DARKCYAN
    assert out.loc[f"{red}FL1-A{textcolor.END}", "error (%)"].startswith(red)
    assert out.loc[f"{cyan}FL3-A{textcolor.END}", "voltage_target"].startswith(cyan)
    full = color_voltageconfig(vc, errorlevel=100, simplify=False)
    assert "factor" in full.columns
    assert all(i.startswith(cyan) for i in full.index)
