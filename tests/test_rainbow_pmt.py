"""cytovanni.rainbow.pmt: power-law fit of the detector gain in PMT voltage and laser power."""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from cytovanni.rainbow import PMTExponentFitter

V = np.array([400., 450., 500., 550., 600., 650.])
I = np.array([1., 1., 2., 2., 3., 3.])
TRUE = dict(intercept=1.5, exponent_PMT=7., exponent_LAS=.9)


def _data(noise=0., rng=None):
    ls = TRUE["intercept"] + TRUE["exponent_PMT"] * np.log(V) + TRUE["exponent_LAS"] * np.log(I)
    if noise:
        ls = ls + noise * rng.normal(size=len(V))
    df_ls = pd.DataFrame({"c1": ls, "c2": np.zeros(len(V))}, index=[f"m{i}" for i in range(len(V))])
    df_v = pd.DataFrame({"c1": V, "c2": np.full(len(V), 500.)}, index=df_ls.index)    # c2: a single voltage
    df_i = pd.DataFrame({"c1": I, "c2": I}, index=df_ls.index)
    return df_ls, df_v, df_i


def test_fit_recovers_exponents():
    fitter = PMTExponentFitter(*_data(), fit_laser_exponent=True)
    fitter.fit_all()
    assert list(fitter.parameters.columns) == ["intercept", "exponent_PMT", "exponent_LAS"]
    assert list(fitter.parameters.index) == ["c1", "c2"] and fitter.channels == ["c1", "c2"]
    np.testing.assert_allclose(fitter.parameters.loc["c1"], [TRUE["intercept"], TRUE["exponent_PMT"], TRUE["exponent_LAS"]],
                               atol=1e-4)


def test_fit_with_noise_recovers_pmt_exponent(rng):
    fitter = PMTExponentFitter(*_data(noise=.01, rng=rng))
    fitter.fit_all()
    assert fitter.parameters.loc["c1", "exponent_PMT"] == pytest.approx(7., abs=.3)


def test_single_voltage_channel_gives_nan_R2_C8():
    """R2-C8: a channel with one voltage gets NaN parameters instead of crashing fit_all."""
    fitter = PMTExponentFitter(*_data())
    fitter.fit_all()
    assert fitter.parameters.loc["c2"].isna().all()


def test_theoretical_shift_reproduces_fit_with_intercept_R2_C8():
    """R2-C8: get_theoretical_shift returns the fitted model's prediction, intercept included; NaN for channels
    without parameters."""
    df_ls, df_v, df_i = _data()
    fitter = PMTExponentFitter(df_ls, df_v, df_i)
    fitter.fit_all()
    th = fitter.get_theoretical_shift()
    assert list(th.index) == list(df_ls.index) and list(th.columns) == ["c1", "c2"]
    np.testing.assert_allclose(th["c1"], df_ls["c1"], atol=1e-4)
    assert th["c2"].isna().all()


def test_theoretical_shift_nan_for_invalid_voltage():
    df_ls, df_v, df_i = _data()
    fitter = PMTExponentFitter(df_ls, df_v, df_i)
    fitter.fit_all()
    df_v.iloc[0, 0] = 0.    # set after the fit: only the prediction is affected
    th = fitter.get_theoretical_shift()
    assert np.isnan(th["c1"].iloc[0]) and th["c1"].iloc[1:].notna().all()


def test_invalid_rows_are_left_out_of_the_fit():
    df_ls, df_v, df_i = _data()
    df_ls.loc["m0", "c1"] = np.nan
    df_v.loc["m1", "c1"] = -1.
    df_i.loc["m2", "c1"] = 0.
    fitter = PMTExponentFitter(df_ls, df_v, df_i)
    fitter.fit_all()
    np.testing.assert_allclose(fitter.parameters.loc["c1"], [1.5, 7., .9], atol=1e-3)


def test_fixed_laser_exponent_is_one_R2_C8():
    """R2-C8: with fit_laser_exponent=False the stored exponent_LAS is exactly 1."""
    df_ls, df_v, _ = _data()
    df_ls["c1"] = 2. + 6. * np.log(V)
    fitter = PMTExponentFitter(df_ls, df_v, None, fit_laser_exponent=False)
    fitter.fit_all()
    assert fitter.parameters.loc["c1", "exponent_LAS"] == 1.
    np.testing.assert_allclose(fitter.parameters.loc["c1", ["intercept", "exponent_PMT"]], [2., 6.], atol=1e-4)


def test_fixed_laser_exponent_two_voltages_R2d_D6():
    """R2d-D6: with fit_laser_exponent=False two voltages suffice (2-parameter fit; it raised TypeError before)."""
    df_ls = pd.DataFrame({"c": [3. + 7. * np.log(400.), 3. + 7. * np.log(500.)]})
    df_v = pd.DataFrame({"c": [400., 500.]})
    params = PMTExponentFitter(df_ls, df_v, None, fit_laser_exponent=False).fit_single("c")
    np.testing.assert_allclose(params, [3., 7., 1.], atol=1e-6)


def test_fixed_laser_exponent_matches_polyfit_R2d_D6(rng):
    """R2d-D6: with >= 3 voltages the 2-parameter fit is the ordinary least-squares line in ln(voltage)."""
    v = np.array([400., 450., 500., 600.])
    y = 1. + 6.5 * np.log(v) + .05 * rng.normal(size=4)
    params = PMTExponentFitter(pd.DataFrame({"c": y}), pd.DataFrame({"c": v}), None,
                               fit_laser_exponent=False).fit_single("c")
    slope, intercept = np.polyfit(np.log(v), y, 1)
    np.testing.assert_allclose(params[:2], [intercept, slope], rtol=1e-5)


def test_get_intensity_defaults_to_ones():
    df_ls, df_v, df_i = _data()
    np.testing.assert_array_equal(PMTExponentFitter(df_ls, df_v).get_intensity("c1"), np.ones(len(V)))
    pd.testing.assert_series_equal(PMTExponentFitter(df_ls, df_v, df_i).get_intensity("c1"), df_i["c1"])


def test_plots_and_write(tmp_path):
    fitter = PMTExponentFitter(*_data())
    fitter.fit_single("c1", plot=True)
    fitter.fit_all()
    fitter.plot_exponents()
    fitter.plot_fit()
    path = tmp_path / "pmt.csv"
    fitter.write(str(path))
    back = pd.read_csv(path, index_col=0)
    pd.testing.assert_frame_equal(back, fitter.parameters, check_dtype=False)
