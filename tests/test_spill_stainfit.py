"""Tests of cytovanni.spill.stainfit: single-stain spectrum fits (linear, outlier-safe, torch, pos/neg) and collection."""
import os
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.exceptions import SpilloverFitWarning
from cytovanni.spill import stainfit
from cytovanni.spill.stainfit import (LinearSingleStainSpectrumFit, OutliersafeLinearSingleStainSpectrumFit,
                                      OutliersafeTorchSingleStainSpectrumFit, PosNegSingleStainSpectrumFit,
                                      TorchSpectrumFitter, collect_single_fits, collect_peakpos_fromposneg)

CH = synthetic.CHANNELS_FLUORESCENCE


def ss_adata(dye="FL1", n=3000, seed=0, channel=True, offset=(0., 0., 0.), spectra=synthetic.SPECTRA):
    """Single-stain AnnData of `dye` (batch A: gains 1, so the true slopes are ``spectra.loc[dye]``), raw layer only."""
    df = synthetic.single_stain_events(dye, "A", n=n, seed=seed, spectra=spectra)
    x = df[CH].to_numpy(np.float64) + np.asarray(offset)[None]
    adata = ad.AnnData(x.astype(np.float32), var=pd.DataFrame(index=CH), obs=pd.DataFrame(index=[str(i) for i in range(n)]))
    adata.layers["raw"] = x
    adata.uns.update({"cytoconfig": synthetic.make_config(), "dye": dye})
    if channel:
        adata.uns["channel"] = dye + "-A"
    return adata


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- base class: main channel, linearity, input copy
def test_main_channel_from_uns_channel():
    adata = ss_adata("FL2")
    adata.uns["channel"] = "FL3-A"  # deliberately not the brightest
    f = LinearSingleStainSpectrumFit(adata)
    assert f.stain == "FL3-A"
    assert f.spectrum.loc["FL3-A", "slope"] == 1.


@pytest.mark.parametrize("dye", synthetic.STAINS)
def test_main_channel_inferred_from_95th_percentile(dye):
    adata = ss_adata(dye, channel=False)
    assert stainfit._find_max_channel(adata) == dye + "-A"
    assert LinearSingleStainSpectrumFit(adata).stain == dye + "-A"


def test_main_channel_ignored_if_not_a_channel():
    adata = ss_adata("FL2")
    adata.uns["channel"] = "XYZ-A"
    assert LinearSingleStainSpectrumFit(adata).stain == "FL2-A"


def test_assure_linearity_drops_beyond_tolerance_and_warns():
    adata = ss_adata("FL1", n=1000)
    lim = adata.uns["cytoconfig"].max_linear_range
    raw = adata.layers["raw"].copy()
    raw[:20, 0] = 1.5 * lim   # nonlinear, within tolerance (kept)
    raw[20:40, 1] = 3 * lim   # outside tolerance (dropped)
    adata.layers["raw"] = raw
    with pytest.warns(SpilloverFitWarning, match="nonlinear range"):
        f = LinearSingleStainSpectrumFit(adata, fit=False)
    assert f.adata.shape[0] == 980
    assert (f.adata.layers["raw"][:, 1] < 2 * lim).all()


def test_assure_linearity_silent_below_warn_share(no_warnings):
    adata = ss_adata("FL1", n=1000)
    raw = adata.layers["raw"].copy()
    raw[:5, 0] = 1.5 * adata.uns["cytoconfig"].max_linear_range  # 0.5% < WARN_SHARE
    adata.layers["raw"] = raw
    f = LinearSingleStainSpectrumFit(adata, fit=False)
    assert f.adata.shape[0] == 1000


def test_input_adata_not_changed():
    adata = ss_adata("FL1", n=800)
    before = adata.layers["raw"].copy()
    f = PosNegSingleStainSpectrumFit(adata)
    assert f.adata is not adata
    assert "is_positive" not in adata.obs
    assert np.array_equal(adata.layers["raw"], before)


# ---------------------------------------------------------------- linear fits
@pytest.mark.parametrize("dye", synthetic.STAINS)
@pytest.mark.parametrize("robust", [True, False])
def test_linear_fit_recovers_spectrum(dye, robust):
    f = LinearSingleStainSpectrumFit(ss_adata(dye, offset=(100., 200., -50.)), robust=robust)
    assert list(f.spectrum.columns) == ["intercept", "slope", "r"]
    assert list(f.spectrum.index) == CH
    assert np.allclose(f.spectrum["slope"], synthetic.SPECTRA.loc[dye], atol=.01)
    main = dye + "-A"
    assert f.spectrum.loc[main].tolist() == [0., 1., 1.]
    # intercept of channel c against the main channel: offset_c - slope_c * offset_main
    off = pd.Series([100., 200., -50.], index=CH)
    expected = off - synthetic.SPECTRA.loc[dye] * off[main]
    expected[main] = 0.
    assert np.allclose(f.spectrum["intercept"], expected, atol=15)
    assert (f.spectrum["r"] > .9).all()


@pytest.mark.parametrize("robust", [True, False])
def test_linear_fit_without_intercept(robust):
    f = LinearSingleStainSpectrumFit(ss_adata("FL2"), robust=robust, intercept=False)
    assert (f.spectrum["intercept"] == 0).all()
    assert np.allclose(f.spectrum["slope"], synthetic.SPECTRA.loc["FL2"], atol=.01)


def test_fit_single_channel_against_other_channel():
    f = LinearSingleStainSpectrumFit(ss_adata("FL1"))
    assert f.fit_single_channel("FL1-A") == [0., 1., 1.]
    icpt, slope, r = f.fit_single_channel("FL3-A", stain="FL2-A")
    assert slope == pytest.approx(.05 / .3, abs=.02)


def test_fit_false_does_not_fit():
    f = LinearSingleStainSpectrumFit(ss_adata("FL1"), fit=False)
    assert not hasattr(f, "spectrum")


def test_get_spectrum_shape_and_name():
    f = LinearSingleStainSpectrumFit(ss_adata("FL3"))
    s = f.get_spectrum()
    assert s.shape == (3, 1) and list(s.columns) == ["FL3"] and list(s.index) == CH
    assert "intercept" in f.spectrum.columns  # get_spectrum does not change the stored spectrum


def test_write_and_plot(tmp_path):
    f = LinearSingleStainSpectrumFit(ss_adata("FL1", n=500))
    f.write(str(tmp_path))
    back = pd.read_csv(tmp_path / "fit_linear" / "FL1.csv", index_col=0)
    assert np.allclose(back.to_numpy(), f.spectrum.to_numpy())
    f.plot(folder=str(tmp_path))
    assert (tmp_path / "plot_fit_linear" / "FL1.jpg").exists()


def test_fit_NxN_consistent_with_spectrum():
    f = LinearSingleStainSpectrumFit(ss_adata("FL1"))
    f.fit_NxN()
    for attr in ["NxN_slope", "NxN_intercept", "NxN_slope_T", "NxN_intercept_T", "NxN_slope_imp", "NxN_intercept_imp"]:
        m = getattr(f, attr)
        assert m.shape == (3, 3) and list(m.index) == CH and list(m.columns) == CH
    # column of the main channel: regression of j on the main channel = the fitted spectrum
    assert np.allclose(f.NxN_slope["FL1-A"], f.spectrum["slope"])
    assert np.allclose(f.NxN_slope_imp["FL1-A"], f.spectrum["slope"])
    assert np.allclose(np.diag(f.NxN_slope_imp), 1.)
    assert np.allclose(np.diag(f.NxN_slope), 1.)
    # implied j-against-i slope = slope_j / slope_i
    assert f.NxN_slope_imp.loc["FL3-A", "FL2-A"] == pytest.approx(f.spectrum.loc["FL3-A", "slope"] / f.spectrum.loc["FL2-A", "slope"])


def test_plot_NxN_channel_subset_any_order(tmp_path):
    """R3b-B9: plot_NxN with a channel subset looks data and lines up by channel name."""
    f = LinearSingleStainSpectrumFit(ss_adata("FL1", n=400))
    f.fit_NxN()
    import matplotlib.pyplot as plt
    f.plot_NxN(channels=["FL3-A", "FL1-A"])
    fig = plt.gcf()
    ax = fig.axes[1]  # i=0 (FL3-A on x), j=1 (FL1-A on y)
    assert ax.get_xlabel() == "FL3-A" and ax.get_ylabel() == "FL1-A"
    x, y = ax.collections[0].get_offsets().T
    assert np.allclose(x, f.adata[:, "FL3-A"].layers["raw"][:, 0])
    assert np.allclose(y, f.adata[:, "FL1-A"].layers["raw"][:, 0])
    line = ax.lines[0].get_ydata()
    xord = np.sort(f.adata[:, "FL3-A"].layers["raw"][:, 0])
    expected = xord * f.NxN_slope.loc["FL1-A", "FL3-A"] + f.NxN_intercept.loc["FL1-A", "FL3-A"]
    assert np.allclose(line, expected)
    plt.close("all")
    f.plot_NxN(savepath=str(tmp_path / "nxn.jpg"), clipNchannel=2)
    assert (tmp_path / "nxn.jpg").exists()


# ---------------------------------------------------------------- outlier-safe linear fit
def test_outliersafe_removes_outliers_and_recovers_spectrum():
    adata = ss_adata("FL2", n=3000)
    raw = adata.layers["raw"].copy()
    rng = np.random.default_rng(3)
    idx = rng.choice(3000, 30, replace=False)
    raw[idx, 0] += rng.uniform(1e4, 1e5, 30)  # 1% isolated events with strong FL1 contamination (off the FL2 line)
    raw[idx, 2] += rng.uniform(-3e4, 3e4, 30)
    adata.layers["raw"] = raw
    lin = LinearSingleStainSpectrumFit(adata.copy(), robust=False)
    f = quiet(OutliersafeLinearSingleStainSpectrumFit, adata, robust=False)
    assert f.adata.shape[0] < 3000
    kept = set(f.adata.obs_names)
    assert sum(str(i) in kept for i in idx) < 5  # most contaminated events removed
    assert abs(f.spectrum.loc["FL1-A", "slope"] - .1) < abs(lin.spectrum.loc["FL1-A", "slope"] - .1)
    assert np.allclose(f.spectrum["slope"], synthetic.SPECTRA.loc["FL2"], atol=.02)


def test_outliersafe_warns_on_large_outlier_share():
    adata = ss_adata("FL1", n=1000)
    raw = adata.layers["raw"].copy()
    raw[:100, 2] += 3e4  # 10% outliers
    adata.layers["raw"] = raw
    with pytest.warns(SpilloverFitWarning, match="outlier fraction"):
        OutliersafeLinearSingleStainSpectrumFit(adata)


@pytest.mark.parametrize("n", [60, 300, 1000])
def test_outliersafe_small_files_use_at_least_5_neighbours(n):
    """R2-E6: LocalOutlierFactor uses max(N // neighbors_factor, 5) neighbours; < 100 events no longer crash."""
    f = quiet(OutliersafeLinearSingleStainSpectrumFit, ss_adata("FL1", n=n))
    assert f._lof_n_neighbors() == max(f.adata.shape[0] // 100, 5)
    assert np.isfinite(f.spectrum["slope"]).all()


def test_outliersafe_neighbors_factor():
    f = quiet(OutliersafeLinearSingleStainSpectrumFit, ss_adata("FL1", n=2000), neighbors_factor=50)
    assert f._lof_n_neighbors() == max(f.adata.shape[0] // 50, 5)


def test_filter_outlier_residuals_removes_events():
    f = quiet(OutliersafeLinearSingleStainSpectrumFit, ss_adata("FL1", n=1500))
    n = f.adata.shape[0]
    quiet(f.filter_outlier_residuals)
    assert 0 < f.adata.shape[0] < n


def test_outliersafe_write_folder(tmp_path):
    f = quiet(OutliersafeLinearSingleStainSpectrumFit, ss_adata("FL3", n=500))
    f.write(str(tmp_path))
    assert (tmp_path / "fit_linearos" / "FL3.csv").exists()


# ---------------------------------------------------------------- torch fitter
def test_torch_fitter_invalid_criterion():
    with pytest.raises(ValueError, match="Criterion"):
        TorchSpectrumFitter(np.ones((10, 3)), CH, criterion="huber")


@pytest.mark.parametrize("criterion", ["mse", "L1"])
def test_torch_fitter_shapes_and_short_fit(device, criterion):
    data = ss_adata("FL1", n=500).layers["raw"]
    tsf = TorchSpectrumFitter(data, CH, device=device, criterion=criterion)
    assert tsf.spectrum.shape == (3,) and tsf.offset.shape == (3,) and tsf.staining.shape == (500,)
    assert tsf.reconstruction.shape == (500, 3)
    assert (tsf.spectrum > 0).all()
    assert str(tsf.data.device).startswith(device)
    l0 = tsf.get_loss().item()
    tsf.fit(Niter=50)
    assert len(tsf.losshistory) == 50
    # MSE: starts at the polyfit solution and does not get worse; L1 (sign gradients) fluctuates by a few percent
    assert np.isfinite(tsf.losshistory).all()
    assert tsf.losshistory[-1] <= l0 * (1.01 if criterion == "mse" else 1.1)


def test_torch_fitter_init_matches_polyfit():
    data = ss_adata("FL2", n=800).layers["raw"]
    tsf = TorchSpectrumFitter(data, CH)
    imax = data.mean(0).argmax()
    slope, intercept = np.polyfit(data[:, imax], data, 1)
    assert np.allclose(tsf.spectrum.detach().numpy(), np.clip(slope, 1e-14, None), rtol=1e-4)
    assert np.allclose(tsf.offset.detach().numpy(), intercept, rtol=1e-3, atol=1e-2)
    assert np.allclose(tsf.staining.detach().numpy(), data[:, imax], rtol=1e-4)


def test_torch_fitter_debug_history_and_plots():
    """R2-C11: TorchSpectrumFitter(debug=True).fit no longer raises NameError (tqdm imported)."""
    tsf = TorchSpectrumFitter(ss_adata("FL1", n=300).layers["raw"], CH, debug=True)
    tsf.fit(Niter=10)
    assert len(tsf.history_spectrum) == 3 and len(tsf.history_offset) == 3
    tsf.plot_loss(); tsf.plot_parameters(); tsf.plot_reconstruction_error()
    tsf.plot_NxN(channels=["FL2-A", "FL1-A"])


def test_torch_fitter_warmup_offset_runs():
    tsf = TorchSpectrumFitter(ss_adata("FL1", n=300).layers["raw"], CH)
    tsf.fit(Niter=20, warmup_offset=10)
    assert len(tsf.losshistory) == 20


@pytest.fixture
def short_torch_fit(monkeypatch):
    """Make OutliersafeTorchSingleStainSpectrumFit use a short TorchSpectrumFitter.fit (default part of the suite)."""
    orig = TorchSpectrumFitter.fit
    monkeypatch.setattr(TorchSpectrumFitter, "fit", lambda self, Niter=300, **kw: orig(self, Niter=Niter, **kw))


def test_outliersafe_torch_spectrum_and_intercept_columns(short_torch_fit, tmp_path):
    """R2-C11: the torch fit stores an intercept, so the inherited plot works."""
    f = quiet(OutliersafeTorchSingleStainSpectrumFit, ss_adata("FL1", n=1000, offset=(200., 500., -100.)), device="cpu")
    assert set(f.spectrum.columns) == {"slope", "intercept"}
    assert f.spectrum.loc["FL1-A", "slope"] == pytest.approx(1.)
    assert f.spectrum.loc["FL1-A", "intercept"] == pytest.approx(0., abs=1e-3)
    assert isinstance(f.tsfitter, TorchSpectrumFitter)
    f.plot(folder=str(tmp_path))
    assert (tmp_path / "plot_fit_torchos" / "FL1.jpg").exists()
    f.write(str(tmp_path))
    assert (tmp_path / "fit_torchos" / "FL1.csv").exists()


def test_outliersafe_torch_device_fallback(short_torch_fit):
    f = quiet(OutliersafeTorchSingleStainSpectrumFit, ss_adata("FL1", n=300))
    assert f.device == ("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.mark.slow
def test_outliersafe_torch_recovers_spectrum_and_intercept(device):
    """R2-C11 (recovery): full default fit; intercept = offset_c - slope_c * offset_main (true [0, 440, -110])."""
    f = quiet(OutliersafeTorchSingleStainSpectrumFit, ss_adata("FL1", n=3000, offset=(200., 500., -100.)), device=device)
    lin = quiet(OutliersafeLinearSingleStainSpectrumFit, ss_adata("FL1", n=3000, offset=(200., 500., -100.)))
    assert np.allclose(f.spectrum["slope"], synthetic.SPECTRA.loc["FL1"], atol=.01)
    assert np.allclose(f.spectrum["intercept"], [0, 440, -110], atol=25)
    assert np.allclose(f.spectrum["intercept"], lin.spectrum["intercept"], atol=25)


# ---------------------------------------------------------------- pos/neg fit
@pytest.mark.parametrize("dye", synthetic.STAINS)
def test_posneg_recovers_spectrum(dye):
    f = PosNegSingleStainSpectrumFit(ss_adata(dye, offset=(100., 200., -50.)))
    assert list(f.spectrum.columns) == ["median_positive", "median_negative", "slope", "intercept"]
    assert np.allclose(f.spectrum["slope"], synthetic.SPECTRA.loc[dye], atol=.02)
    assert f.spectrum.loc[dye + "-A", "slope"] == pytest.approx(1.)
    assert f.spectrum.loc[dye + "-A", "intercept"] == pytest.approx(0., abs=1e-9)
    # about half the events are stained (single_stain_events), the positive median is far above the negative one
    assert .4 < f.adata.obs["is_positive"].mean() < .6
    m = f.spectrum.loc[dye + "-A"]
    assert m["median_positive"] > 10 * abs(m["median_negative"])


def test_posneg_positive_population_is_higher_component():
    f = PosNegSingleStainSpectrumFit(ss_adata("FL2"))
    x = f.adata[:, "FL2-A"].layers["raw"][:, 0]
    pos = f.adata.obs["is_positive"].to_numpy()
    assert x[pos].min() > np.median(x[~pos])


def test_posneg_asinh_cofactor_is_used():
    f = PosNegSingleStainSpectrumFit(ss_adata("FL1"), asinh_cofactor=150)
    assert f.asinh_cofactor == 150
    assert np.allclose(f.spectrum["slope"], synthetic.SPECTRA.loc["FL1"], atol=.02)


# ---------------------------------------------------------------- collecting the fits of one measurement
def test_collect_single_fits_sorted_dyes(tmp_path):
    for dye in ["FL3", "FL1", "FL2"]:
        LinearSingleStainSpectrumFit(ss_adata(dye, n=500)).write(str(tmp_path))
    collect_single_fits(str(tmp_path), "fit_linear", "spectra_linear.csv")
    sp = pd.read_csv(tmp_path / "spectra_linear.csv", index_col=0)
    assert list(sp.columns) == ["FL1", "FL2", "FL3"]  # (channel, dye), sorted
    assert list(sp.index) == CH
    assert np.allclose(sp.to_numpy(), synthetic.SPECTRA.T.to_numpy(), atol=.02)


def test_collect_peakpos_fromposneg(tmp_path):
    meta = pd.DataFrame({"channel": [c for c in CH] + ["FL4-A"], "dye": synthetic.STAINS + ["FL4"]})
    meta.to_csv(tmp_path / "panel_meta.csv")
    fits = {}
    for dye in ["FL1", "FL3"]:
        fits[dye] = PosNegSingleStainSpectrumFit(ss_adata(dye, n=800))
        fits[dye].write(str(tmp_path))
    collect_peakpos_fromposneg(str(tmp_path))
    back = pd.read_csv(tmp_path / "panel_meta.csv", index_col=0)
    assert list(back.index) == [0, 1, 2, 3] and list(back["dye"]) == synthetic.STAINS + ["FL4"]
    assert back["positive_peak"].isna().tolist() == [False, True, False, True]
    assert back.loc[0, "positive_peak"] == np.round(fits["FL1"].spectrum["median_positive"].max())


def test_posneg_random_state_reaches_the_mixture():
    adata = ss_adata("FL1", n=800)
    f = PosNegSingleStainSpectrumFit(adata, random_state=3)
    assert f.gmm.random_state == 3
    g = PosNegSingleStainSpectrumFit(adata, random_state=3)
    assert f.spectrum.equals(g.spectrum)
