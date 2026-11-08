"""Tests of cytovanni.ref.factor: per-batch scale factors (Sinkhorn, mean and percentile matching)."""
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
import cytovanni.ref.factor as factor_module
from cytovanni.ref import (fit_factor_MeanMatching, fit_factor_PercentileMatching, fit_factor_Sinkhorn,
                           fit_factor_Sinkhorn_grouped, fit_factors_Sinkhorn)
from cytovanni.ref.factor import _triu_indices

FAST = dict(verbose=False, Nsamp=300, Niter=40)


def marker_events(scale, n=1500, seed=0):
    """Events of one marker: a negative population and lognormal positives, multiplied by `scale`."""
    rng = np.random.default_rng(seed)
    x = np.where(rng.random(n) < .5, rng.lognormal(9.5, .4, n), rng.normal(100, 60, n))
    return x * scale


# ------------------------------------------------------------------ helpers
def test_triu_indices_all_pairs_or_random_subset():
    assert _triu_indices(3).tolist() == [[0, 0, 1], [1, 2, 2]]
    sub = _triu_indices(5, Nmax=4)
    assert sub.shape == (2, 4) and torch.all(sub[0] < sub[1])
    assert _triu_indices(3, Nmax=10).shape == (2, 3)


# ------------------------------------------------------------------ mean / percentile matching
def test_mean_matching_formula_and_clipping():
    d = {"a": np.array([-10., 2., 4.]), "b": np.array([6., 6.])}
    f = fit_factor_MeanMatching(d)
    m = pd.Series({"a": 2., "b": 6.})  # negative values clipped to 0 before the mean
    assert np.allclose(f.to_numpy(), (m.mean() / m).to_numpy())
    assert f.index.tolist() == ["a", "b"]


def test_mean_matching_recovers_relative_scales_and_anchor():
    d = {b: marker_events(s, seed=0) for b, s in zip(synthetic.BATCHES, [1., 2., .5])}
    f = fit_factor_MeanMatching(d, anchor="A")
    assert f["A"] == 1.
    assert np.allclose(f.to_numpy(), [1., .5, 2.])


def test_mean_matching_unknown_anchor_raises_keyerror():
    with pytest.raises(KeyError):
        fit_factor_MeanMatching({"a": np.ones(3)}, anchor="z")


def test_mean_matching_warns_for_zero_mean():
    with pytest.warns(UserWarning, match="negative, zero, or infinite"):
        fit_factor_MeanMatching({"a": np.ones(3), "b": -np.ones(3)})


def test_mean_and_percentile_matching_dummy_returns_ones():
    d = {"a": np.ones(3), "b": 2 * np.ones(3)}
    for fct in [fit_factor_MeanMatching, fit_factor_PercentileMatching]:
        assert fct(d, dummy=True).tolist() == [1., 1.]


def test_percentile_matching_recovers_scales_and_flattens():
    d = {"a": marker_events(1.), "b": marker_events(3.)}
    f = fit_factor_PercentileMatching(d, percentile=90, anchor="a")
    assert np.isclose(f["b"], 1 / 3)
    f2 = fit_factor_PercentileMatching(d, percentile=90)
    q = pd.Series({k: np.percentile(v, 90) for k, v in d.items()})
    assert np.allclose(f2.to_numpy(), (q.mean() / q).to_numpy())


def test_percentile_matching_accepts_any_array_like():
    """R2-E11: lists, Series and 2-D arrays are accepted (values pooled)."""
    d = {"a": [1., 2., 3., 4.], "b": pd.Series([2., 4., 6., 8.]), "c": np.array([[1., 2.], [3., 4.]])}
    f = fit_factor_PercentileMatching(d, percentile=50, anchor="a")
    assert np.allclose(f.to_numpy(), [1., .5, 1.])


def test_percentile_matching_warns_for_nonpositive_percentile():
    with pytest.warns(UserWarning, match="negative, zero, or infinite"):
        fit_factor_PercentileMatching({"a": np.ones(5), "b": -np.ones(5)})


# ------------------------------------------------------------------ Sinkhorn, one marker
def test_sinkhorn_dummy_returns_ones_without_checks():
    f = fit_factor_Sinkhorn({"a": np.zeros(0), "b": np.ones(2)}, dummy=True, anchor="missing")
    assert f.tolist() == [1., 1.] and f.index.tolist() == ["a", "b"]


def test_sinkhorn_unknown_anchor_raises():
    with pytest.raises(ValueError, match="Anchor"):
        fit_factor_Sinkhorn({"a": np.ones(5), "b": np.ones(5)}, anchor="z", device="cpu")


def test_sinkhorn_unknown_init_raises():
    """R2-K1: an unknown `init` raises NotImplementedError (was TypeError from ``raise NotImplemented``)."""
    d = {"a": marker_events(1.), "b": marker_events(1.)}
    with pytest.raises(NotImplementedError):
        fit_factor_Sinkhorn(d, init="zero", device="cpu", **FAST)
    with pytest.raises(NotImplementedError):
        fit_factor_Sinkhorn_grouped({"a": {"s": d["a"]}, "b": {"s": d["b"]}}, init="zero", device="cpu", **FAST)
    with pytest.raises(NotImplementedError):
        fit_factors_Sinkhorn({k: pd.DataFrame({"m": v}) for k, v in d.items()}, init="zero", device="cpu", **FAST)


def test_sinkhorn_empty_batch_raises():
    """R2b-G2-12: an empty batch (or batch/sample entry) raises ValueError in all three Sinkhorn fits."""
    x = marker_events(1., n=300)
    with pytest.raises(ValueError, match="No events"):
        fit_factor_Sinkhorn({"a": x, "e": np.zeros(0), "b": 2 * x}, device="cpu", **FAST)
    with pytest.raises(ValueError, match="No events"):
        fit_factor_Sinkhorn_grouped({"a": {"s": x}, "b": {"s": np.zeros(0)}}, device="cpu", **FAST)
    df = lambda v: pd.DataFrame({"m1": v, "m2": v})
    with pytest.raises(ValueError, match="No events"):
        fit_factors_Sinkhorn({"a": df(x), "b": df(np.zeros(0))}, device="cpu", **FAST)


def test_sinkhorn_lbfgs_variant_removed():
    """R2b-G2-11: the diverging L-BFGS variant is gone."""
    assert not hasattr(factor_module, "fit_factor_Sinkhorn_LBFGS")


def test_sinkhorn_geometric_mean_one_without_anchor(device):
    d = {"a": marker_events(1., n=300), "b": marker_events(2., n=300, seed=1)}
    f = fit_factor_Sinkhorn(d, device=device, verbose=False, Nsamp=200, Niter=10)
    assert np.isclose(np.log(f.to_numpy()).mean(), 0., atol=1e-5)
    assert f.index.tolist() == ["a", "b"]


def test_sinkhorn_verbose_plots(device):
    d = {"a": marker_events(1., n=200), "b": marker_events(2., n=200, seed=1)}
    import matplotlib.pyplot as plt
    f = fit_factor_Sinkhorn(d, device=device, verbose=True, Nsamp=100, Niter=3)
    assert len(f) == 2 and len(plt.get_fignums()) >= 1


@pytest.mark.slow
@pytest.mark.parametrize("aggregate", [True, False])
@pytest.mark.parametrize("init", ["null", "mean"])
def test_sinkhorn_recovers_scale_factor(aggregate, init):
    """C: factors recover 1 / true scale (relative to the anchor) within 10 %."""
    scales = {"A": 1., "B": 2., "C": .6}
    d = {b: marker_events(s, seed=i) for i, (b, s) in enumerate(scales.items())}
    torch.manual_seed(0)
    f = fit_factor_Sinkhorn(d, device="cpu", verbose=False, Nsamp=500, Niter=120, aggregate=aggregate, init=init, anchor="A")
    assert f["A"] == pytest.approx(1.)
    for b, s in scales.items():
        assert f[b] == pytest.approx(1 / s, rel=.1), f.to_dict()


def test_sinkhorn_aggregate_loss_divided_by_nbatch(monkeypatch):
    """R1-D2: with ``aggregate=True`` the summed loss is divided by the number of batches."""
    seen = []
    calls = []
    orig = factor_module.GeomSampleLoss

    class CountingLoss(orig):
        def forward(self, *a, **k):
            out = super().forward(*a, **k)
            calls.append(out.item())
            return out

    monkeypatch.setattr(factor_module, "GeomSampleLoss", CountingLoss)
    d = {b: marker_events(s, n=200, seed=i) for i, (b, s) in enumerate(zip("abc", [1., 2., 3.]))}
    torch.manual_seed(0)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: seen.append(plt.gcf()))
    fit_factor_Sinkhorn(d, device="cpu", verbose=True, Nsamp=200, Niter=1, aggregate=True)
    loss_plotted = seen[0].axes[0].lines[0].get_ydata()[0]
    assert len(calls) == 3
    assert loss_plotted == pytest.approx(sum(calls) / 3, rel=1e-5)


# ------------------------------------------------------------------ Sinkhorn, grouped
def test_sinkhorn_grouped_no_shared_sample_raises():
    """R2-K4: no sample present in more than one batch raises ValueError (was ZeroDivisionError)."""
    with pytest.raises(ValueError, match="No sample"):
        fit_factor_Sinkhorn_grouped({"a": {"s1": np.ones(10)}, "b": {"s2": np.ones(10)}}, device="cpu", **FAST)


def test_sinkhorn_grouped_warns_for_unshared_batch():
    """R2-K4: a batch that shares no sample with another batch is named in a warning."""
    x = marker_events(1., n=300)
    d = {"a": {"s1": x}, "b": {"s1": x}, "c": {"s9": x}}
    with pytest.warns(UserWarning, match="share no sample") as rec:
        f = fit_factor_Sinkhorn_grouped(d, device="cpu", **FAST)
    assert "'c'" in str(rec[0].message)
    assert f.index.tolist() == ["a", "b", "c"]


def test_sinkhorn_grouped_dummy_and_anchor_check():
    assert fit_factor_Sinkhorn_grouped({"a": {}, "b": {}}, dummy=True).tolist() == [1., 1.]
    with pytest.raises(ValueError, match="Anchor"):
        fit_factor_Sinkhorn_grouped({"a": {"s": np.ones(3)}, "b": {"s": np.ones(3)}}, anchor="z", device="cpu")


@pytest.mark.slow
@pytest.mark.parametrize("init", ["null", "mean"])
def test_sinkhorn_grouped_recovers_scale_with_different_compositions(init):
    """C: two donors with different compositions, one factor per day; the shared factor is recovered within 10 %."""
    donor1 = lambda s, seed: marker_events(s, seed=seed)
    donor2 = lambda s, seed: np.random.default_rng(seed).lognormal(7.5, .3, 1500) * s
    d = {"day1": {"d1": donor1(1., 0), "d2": donor2(1., 1)},
         "day2": {"d1": donor1(2.5, 2), "d2": donor2(2.5, 3)}}
    torch.manual_seed(0)
    f = fit_factor_Sinkhorn_grouped(d, device="cpu", verbose=False, Nsamp=500, Niter=120, init=init, anchor="day1")
    assert f["day1"] == pytest.approx(1.)
    assert f["day2"] == pytest.approx(1 / 2.5, rel=.1), f.to_dict()


# ------------------------------------------------------------------ Sinkhorn, several markers
def frames(scales, n=1500, seed=0, columns=("m1", "m2")):
    """Per batch a DataFrame (events, marker) of independent markers, each multiplied by its scale."""
    out = {}
    for i, (b, s) in enumerate(scales.items()):
        out[b] = pd.DataFrame({c: marker_events(s[j], n=n, seed=seed + 10 * i + j) for j, c in enumerate(columns)})
    return out


def test_fit_factors_sinkhorn_dummy_returns_ones_and_checks_columns():
    """R2-C10: ``dummy=True`` returns the frame of ones (the fit ran before)."""
    d = {"a": pd.DataFrame(np.ones((5, 2)), columns=["m1", "m2"]), "b": pd.DataFrame(np.ones((5, 2)), columns=["m2", "m1"])}
    r = fit_factors_Sinkhorn(d, dummy=True, device="cpu")
    assert (r.to_numpy() == 1).all() and r.columns.tolist() == ["m1", "m2"] and r.index.tolist() == ["a", "b"]


def test_fit_factors_sinkhorn_column_checks():
    """R2d-D1: duplicate columns or a different marker set raise ValueError."""
    good = pd.DataFrame(np.ones((5, 2)), columns=["m1", "m2"])
    with pytest.raises(ValueError, match="Duplicate"):
        fit_factors_Sinkhorn({"a": pd.DataFrame(np.ones((5, 2)), columns=["m1", "m1"]), "b": good}, dummy=True)
    with pytest.raises(ValueError, match="expected the same set"):
        fit_factors_Sinkhorn({"a": good, "b": pd.DataFrame(np.ones((5, 2)), columns=["m1", "m3"])}, dummy=True)
    with pytest.raises(ValueError, match="Anchor"):
        fit_factors_Sinkhorn({"a": good, "b": good}, anchor="z", device="cpu")


def test_fit_factors_sinkhorn_aligns_columns_by_name():
    """R2d-D1: frames with the same markers in another column order are aligned by name (identical data -> 1)."""
    x = frames({"a": (1., 1.)}, columns=("m1", "m2"))["a"]
    x["m2"] = x["m2"] * 20  # very different marker scales, so a positional mix-up would give factors far from 1
    d = {"a": x, "b": x[["m2", "m1"]].copy()}
    torch.manual_seed(0)
    f = fit_factors_Sinkhorn(d, device="cpu", verbose=False, Nsamp=300, Niter=30)
    assert f.columns.tolist() == ["m1", "m2"]
    assert np.allclose(f.to_numpy(), 1., atol=.03), f


def test_fit_factors_sinkhorn_unfitted_markers_keep_factor_one():
    d = frames({"a": (1., 1.), "b": (2., 3.)})
    torch.manual_seed(0)
    f = fit_factors_Sinkhorn(d, fit_markers=["m1", "unknown"], device="cpu", verbose=False, Nsamp=300, Niter=40, init="mean")
    assert np.allclose(f["m2"].to_numpy(), 1.)
    assert f.loc["b", "m1"] < f.loc["a", "m1"]


def test_fit_factors_sinkhorn_no_matching_fit_marker_raises_runtimeerror():
    d = frames({"a": (1., 1.), "b": (2., 3.)}, n=200)
    with pytest.raises(RuntimeError):
        fit_factors_Sinkhorn(d, fit_markers=["nothing"], device="cpu", verbose=False, Nsamp=100, Niter=2)


def test_fit_factors_sinkhorn_verbose_and_geometric_mean(device):
    d = frames({"a": (1., 1.), "b": (2., .5)}, n=300)
    f = fit_factors_Sinkhorn(d, device=device, verbose=True, Nsamp=100, Niter=3)
    assert np.allclose(np.log(f.to_numpy()).mean(0), 0., atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("separate, aggregate", [(False, True), (True, True), (True, False)])
def test_fit_factors_sinkhorn_recovers_stain_scales(separate, aggregate):
    """C: per-batch, per-marker factors recover 1 / STAIN_SCALES (relative to batch A) within 10 %."""
    ab = synthetic.cell_abundances(2000, np.random.default_rng(0))
    d = {b: pd.DataFrame(ab * synthetic.STAIN_SCALES.loc[b].to_numpy()[None], columns=synthetic.STAINS)
         for b in synthetic.BATCHES}
    torch.manual_seed(0)
    f = fit_factors_Sinkhorn(d, device="cpu", verbose=False, Nsamp=600, Niter=150, separate=separate,
                             aggregate=aggregate, init="mean", anchor="A")
    expected = 1 / synthetic.STAIN_SCALES
    assert np.allclose(f.loc["A"].to_numpy(), 1.)
    assert np.allclose(f.to_numpy(), expected.loc[f.index, f.columns].to_numpy(), rtol=.1), f
