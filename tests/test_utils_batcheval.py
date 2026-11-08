"""Tests of cytovanni.utils.batcheval: per-sample summaries, paired CV tables (MFI, sigmaFI, proportions), counting-noise
corrections, the Gaussian-negative control and replicate correlations, on synthetic data with known answers."""
import warnings

import numpy as np
import pandas as pd
import pytest

from cytovanni.exceptions import CytovanniWarning
from cytovanni.utils.batcheval import (celltype_counts, celltype_marker_summary, gaussian_negative, mfi_cv_table,
                                       mfi_cv_counting_noise, proportion_cv_counting_noise, proportion_cv_table,
                                       replicate_correlation_table, resample_pair_shares, subtract_counting_noise,
                                       width_cv_table, wide_by_marker)
from cytovanni.utils.evl import get_asinh_single_CV, get_positive_single_CV


def lognormal_sd(cv):
    return np.sqrt(np.log(1 + cv**2))


@pytest.fixture
def pairs():
    n = 400
    return pd.DataFrame({"left": [f"L{i}" for i in range(n)], "right": [f"R{i}" for i in range(n)]},
                        index=[f"pair{i}" for i in range(n)])


@pytest.fixture
def mfi(rng, pairs):
    """arcsinh-scale means well above the cofactor (asinh ~ log(2z/f)): measurement CV 0.2 for marker m1 (both
    types) and 0.05 for m2; true values spread uniformly in [3, 5]."""
    n = len(pairs)
    samples = np.r_[pairs.left, pairs.right]
    means = {}
    for m, cv in [("m1", .2), ("m2", .05)]:
        cols = {}
        for c in ["T", "B"]:
            s = rng.uniform(3, 5, n)
            sd = lognormal_sd(cv)
            cols[c] = np.r_[s + rng.normal(scale=sd, size=n), s + rng.normal(scale=sd, size=n)]
        means[m] = pd.DataFrame(cols, index=samples)
    return means


# ------------------------------------------------------------------ per-sample summaries
def test_celltype_counts(rng):
    labels = pd.DataFrame({"major": rng.choice(["T", "B"], 1000, p=[.9, .1]),
                           "fine": rng.choice(["a", "b", "rare"], 1000, p=[.5, .49999, .00001])})
    res = celltype_counts(labels, min_share=1e-3)
    assert list(res.columns) == ["level", "celltype", "count", "share"]
    major = res[res.level == "major"].set_index("celltype")
    assert major["count"].sum() == 1000 and major.share.sum() == pytest.approx(1)
    assert (res.share >= 1e-3).all()


def test_celltype_marker_summary(rng):
    values = pd.DataFrame(rng.normal(3000, 1000, size=(500, 2)), columns=["CD3", "CD19"])
    labels = pd.Series(pd.Categorical(rng.choice(["T", "B"], 500), categories=["T", "B", "empty"]), name="lineage")
    res = celltype_marker_summary(values, labels, stats=("mean", "median", "std"))
    assert list(res.columns) == ["level", "stat", "celltype", "CD3", "CD19"]
    y = np.clip(np.arcsinh(values / 1500), -12, 12)
    mean = res[res.stat == "mean"].set_index("celltype")
    assert mean.loc["T", "CD3"] == pytest.approx(y.CD3[labels == "T"].mean())
    std = res[res.stat == "std"].set_index("celltype")
    assert std.loc["B", "CD19"] == pytest.approx(y.CD19[labels == "B"].std(ddof=1))
    assert np.isnan(mean.loc["empty", "CD3"])  # observed=False keeps empty categories
    observed = celltype_marker_summary(values, labels, stats=("std",), observed=True)
    assert "empty" not in set(observed.celltype)
    f32 = celltype_marker_summary(values.astype(np.float32), labels, stats=("mean",)).set_index("celltype")
    assert np.allclose(f32.loc[["T", "B"], ["CD3", "CD19"]], mean.loc[["T", "B"], ["CD3", "CD19"]], rtol=1e-5)


def test_wide_by_marker_roundtrip(rng):
    frames = []
    for s in ["s1", "s2"]:
        values = pd.DataFrame(rng.normal(3000, 1000, size=(100, 2)), columns=["CD3", "CD19"])
        part = celltype_marker_summary(values, {"lineage": rng.choice(["T", "B"], 100)}, stats=("mean",))
        frames.append(part.assign(sample=s))
    wide = wide_by_marker(pd.concat(frames))
    assert set(wide) == {"CD3", "CD19"} and list(wide["CD3"].index) == ["s1", "s2"]
    assert wide["CD3"].loc["s2", "T"] == frames[1].set_index("celltype").loc["T", "CD3"]


# ------------------------------------------------------------------ MFI CV
def test_mfi_cv_table_recovers_measurement_cv(mfi, pairs):
    res = mfi_cv_table(mfi, pairs).set_index(["marker", "celltype"])
    assert list(res.columns) == ["CV", "mean", "CV_diag", "CV_ratio"]
    for c in ["T", "B"]:
        assert res.loc[("m1", c), "CV"] == pytest.approx(.2, rel=.1)
        assert res.loc[("m2", c), "CV"] == pytest.approx(.05, rel=.1)
        assert res.loc[("m1", c), "CV_ratio"] == pytest.approx(res.loc[("m1", c), "CV_diag"] / res.loc[("m1", c), "CV"])
    # CV_diag: spread of the true values, larger than the measurement noise here
    assert (res.CV_diag > res.CV).all()
    pct = mfi_cv_table(mfi, pairs, percent=True)
    assert np.allclose(pct.CV, 100 * res.CV.to_numpy())


def test_mfi_cv_table_options(mfi, pairs):
    res = mfi_cv_table(mfi, pairs, exclude_markers=["m2"], exclude_celltypes=["B"])
    assert list(res.marker) == ["m1"] and list(res.celltype) == ["T"]
    single = mfi_cv_table(mfi, samples=pairs.left).set_index(["marker", "celltype"])
    assert list(single.columns) == ["CV", "mean"]
    x = mfi["m1"].loc[pairs.left, "T"]
    assert single.loc[("m1", "T"), "CV"] == pytest.approx(get_asinh_single_CV(x, 1500))
    with pytest.raises(ValueError):
        mfi_cv_table(mfi)


def test_mfi_cv_table_unpaired_values(mfi, pairs):
    means = {m: df.copy() for m, df in mfi.items()}
    means["m1"].loc["L0", "T"] = np.nan
    means["m1"].loc["R1", "T"] = 100.   # finite, but its partner is missing below
    means["m1"].loc["L1", "T"] = np.nan
    paired = mfi_cv_table(means, pairs).set_index(["marker", "celltype"])
    unpaired = mfi_cv_table(means, pairs, drop_unpaired=False).set_index(["marker", "celltype"])
    assert paired.loc[("m1", "T"), "CV"] == unpaired.loc[("m1", "T"), "CV"]  # CVs always use complete pairs
    assert unpaired.loc[("m1", "T"), "mean"] > paired.loc[("m1", "T"), "mean"]  # the lone 100 enters only here


def test_mfi_cv_table_warns_once_for_nan(mfi, pairs):
    means = {m: df.copy() for m, df in mfi.items()}
    means["m1"]["T"] = np.nan
    with pytest.warns(CytovanniWarning, match="1 of 4"):
        res = mfi_cv_table(means, pairs)
    assert res.CV.isna().sum() == 1


# ------------------------------------------------------------------ sigmaFI CV
def test_width_cv_table(rng, pairs, mfi):
    n = len(pairs)
    samples = np.r_[pairs.left, pairs.right]
    true = np.exp(rng.uniform(-1, 0, n))
    sd = lognormal_sd(.1)
    widths = {m: pd.DataFrame({c: np.r_[true * np.exp(rng.normal(scale=sd, size=n)),
                                        true * np.exp(rng.normal(scale=sd, size=n))] for c in ["T", "B"]},
                              index=samples) for m in mfi}
    res = width_cv_table(mfi, widths, pairs).set_index(["marker", "celltype"])
    assert list(res.columns) == ["mean", "mean_abs_diff", "mean_abs_log_ratio", "CV", "CV_diag", "CV_ratio"]
    assert np.allclose(res.CV, .1, rtol=.1)
    w = widths["m1"]["T"]
    assert res.loc[("m1", "T"), "mean_abs_diff"] == pytest.approx(
        np.mean(np.abs(w.loc[pairs.left].to_numpy() - w.loc[pairs.right].to_numpy())))
    assert res.loc[("m1", "T"), "mean"] == pytest.approx(mfi["m1"]["T"].mean())
    single = width_cv_table(mfi, widths, samples=pairs.left[:5]).set_index(["marker", "celltype"])
    v = w.loc[pairs.left[:5]].to_numpy()
    assert single.loc[("m1", "T"), "mean_abs_diff"] == pytest.approx(
        np.mean([abs(a - b) for i, a in enumerate(v) for b in v[i + 1:]]))
    assert single.loc[("m1", "T"), "CV"] == pytest.approx(get_positive_single_CV(v))


# ------------------------------------------------------------------ proportion CV
@pytest.fixture
def shares(rng, pairs):
    """Log-normal proportions with measurement CV 0.3, true log shares in [-6, -1]."""
    n = len(pairs)
    sd = lognormal_sd(.3)
    cols = {}
    for c in ["T", "B", "NK"]:
        s = rng.uniform(-6, -1, n)
        cols[c] = np.exp(np.r_[s + rng.normal(scale=sd, size=n), s + rng.normal(scale=sd, size=n)])
    return pd.DataFrame(cols, index=np.r_[pairs.left, pairs.right])


def test_proportion_cv_table(shares, pairs):
    res = proportion_cv_table(shares, pairs).set_index("celltype")
    assert list(res.columns) == ["mean", "log_std", "r_sym", "CV", "CV_diag", "CV_ratio"]
    assert np.allclose(res.CV, .3, rtol=.1)
    assert res.loc["T", "mean"] == pytest.approx(np.exp(np.log(shares["T"]).mean()))
    logs = np.log(np.r_[shares["T"].loc[pairs.left], shares["T"].loc[pairs.right]])
    assert res.loc["T", "log_std"] == pytest.approx(logs.std(ddof=1))
    assert list(proportion_cv_table(shares, pairs, exclude_celltypes=["NK"]).celltype) == ["T", "B"]
    single = proportion_cv_table(shares, samples=pairs.left).set_index("celltype")
    assert list(single.columns) == ["mean", "log_std", "CV"]
    assert single.loc["B", "log_std"] == pytest.approx(np.log(shares.B.loc[pairs.left]).std(ddof=1))


# ------------------------------------------------------------------ counting noise
@pytest.fixture
def counts(rng, pairs):
    """Both aliquots drawn from the same composition: all disagreement is counting noise."""
    n = len(pairs)
    comp = rng.dirichlet([20, 5, 1, .3], n)
    sizes_l, sizes_r = rng.integers(2000, 5000, n), rng.integers(2000, 20000, n)
    left = np.array([rng.multinomial(s, p) for s, p in zip(sizes_l, comp)])
    right = np.array([rng.multinomial(s, p) for s, p in zip(sizes_r, comp)])
    return pd.DataFrame(np.r_[left, right], index=np.r_[pairs.left, pairs.right], columns=["T", "B", "NK", "rare"])


def test_resample_pair_shares(counts, pairs):
    a = resample_pair_shares(counts, pairs, rng=3)
    b = resample_pair_shares(counts, pairs, rng=np.random.RandomState(3))
    np.random.seed(3)
    c = resample_pair_shares(counts, pairs)  # global legacy state, as np.random.multinomial
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, c)
    assert np.allclose(a.sum(1), 1, atol=1e-3) and (a >= 5e-5).all().all()
    assert not a.equals(resample_pair_shares(counts, pairs, rng=np.random.default_rng(3)))


def test_proportion_counting_noise_explains_pure_counting_noise(counts, pairs):
    shares = np.clip(counts / counts.to_numpy().sum(1, keepdims=True), 5e-5, None)
    table = proportion_cv_table(shares, pairs)
    noise = proportion_cv_counting_noise(counts, pairs, n_resamples=40, rng=0)
    assert noise.shape == (4, 40) and list(noise.index) == list(table.celltype)
    ratio = noise.median(axis=1).to_numpy()[:3] / table.CV.to_numpy()[:3]  # "rare" is dominated by the clipping
    assert np.all((ratio > .8) & (ratio < 1.25))
    corrected = subtract_counting_noise(table, noise)
    assert (corrected.CV_corrected < .5 * corrected.CV)[:3].all()


def test_proportion_counting_noise_vanishes_for_large_counts(counts, pairs):
    noise = proportion_cv_counting_noise(counts * 10_000, pairs, n_resamples=5, rng=0)
    assert (noise.loc[["T", "B", "NK"]] < .01).all().all()


def test_mfi_counting_noise(rng, pairs):
    """Means of n cells with width 0.5: observed pair CVs are pure counting noise, so the simulated noise matches."""
    n = len(pairs)
    samples = np.r_[pairs.left, pairs.right]
    cnt = pd.DataFrame({"T": rng.integers(20, 200, 2 * n), "B": 100_000}, index=samples)
    width = pd.DataFrame(.5, index=samples, columns=["T", "B"])
    s = rng.uniform(3, 5, n)
    means = {"m1": pd.DataFrame({c: np.r_[s, s] + rng.normal(size=2 * n) * width[c] / np.sqrt(cnt[c])
                                 for c in ["T", "B"]}, index=samples)}
    table = mfi_cv_table(means, pairs)
    noise = mfi_cv_counting_noise(means, {"m1": width}, cnt, pairs, n_redraws=20, rng=1)
    assert noise.shape == (2, 20) and list(noise.index.get_level_values("celltype")) == list(table.celltype)
    ratio = noise.median(axis=1).to_numpy() / table.CV.to_numpy()
    assert np.all((ratio > .8) & (ratio < 1.25))
    again = mfi_cv_counting_noise(means, {"m1": width}, cnt, pairs, n_redraws=20, rng=1)
    pd.testing.assert_frame_equal(noise, again)
    excluded = mfi_cv_counting_noise(means, {"m1": width}, cnt, pairs, n_redraws=2, rng=1, exclude_celltypes=["B"])
    assert len(excluded) == 1


def test_subtract_counting_noise_formula():
    table = pd.DataFrame({"CV": [.5, .1]})
    noise = np.array([[.3, .4, .0], [.2, .05, .05]])
    res = subtract_counting_noise(table, noise, quantiles=(0, 50, 100))
    assert res.CV_corrected.tolist() == pytest.approx([.4, np.sqrt(.01 - .0025)])
    assert res.CV_corrected_minus.tolist() == pytest.approx([.3, 0])
    assert res.CV_noise_rms[0] == pytest.approx(np.sqrt((.09 + .16) / 3))
    with pytest.raises(ValueError):
        subtract_counting_noise(table, noise[:1])


def test_subtract_counting_noise_checks_row_keys(counts, pairs):
    """Noise from proportion_cv_counting_noise must belong to the table's cell types, in the same order."""
    table = proportion_cv_table(counts / counts.to_numpy().sum(1, keepdims=True), pairs)
    noise = proportion_cv_counting_noise(counts, pairs, n_resamples=3, rng=0)
    subtract_counting_noise(table, noise)
    with pytest.raises(ValueError, match="celltype"):
        subtract_counting_noise(table, noise.iloc[::-1])


# ------------------------------------------------------------------ Gaussian negative
def test_gaussian_negative(rng):
    values = pd.DataFrame(rng.normal(500, 2000, size=(20000, 3)), columns=list("abc"), index=np.arange(20000) + 7)
    res = gaussian_negative(values, rng=5)
    low = values < 1000
    assert res.index.equals(values.index) and res.columns.equals(values.columns)
    assert (res[~low] == values[~low]).sum().sum() == (~low).sum().sum()
    assert res.to_numpy()[low.to_numpy()].std() == pytest.approx(200, rel=.03)
    # one draw for the whole matrix before masking: the noise at a position does not depend on the other values
    other = gaussian_negative(values * 0, rng=5)
    assert np.array_equal(res.to_numpy()[low.to_numpy()], other.to_numpy()[low.to_numpy()])
    arr = gaussian_negative(values.to_numpy(), cutoff=0, scale=25, rng=np.random.RandomState(5))
    noise = np.random.RandomState(5).normal(0, 25, values.shape)
    assert np.array_equal(arr, np.where(values.to_numpy() < 0, noise, values.to_numpy()))


def test_gaussian_negative_float32_frame_and_nan(rng):
    """float32 frames (as read from .fcs files) are upcast instead of failing on the float64 noise; NaN stays NaN."""
    values = pd.DataFrame(rng.normal(500, 2000, size=(500, 3)).astype(np.float32), columns=list("abc"))
    values.iloc[0, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = gaussian_negative(values, rng=5)
    noise = np.random.RandomState(5).normal(0, 200, values.shape)
    expected = np.where(values.to_numpy() < 1000, noise, values.to_numpy())
    np.testing.assert_array_equal(res.to_numpy(), expected)
    assert np.isnan(res.iloc[0, 0])


# ------------------------------------------------------------------ replicate correlation
def test_replicate_correlation_table(mfi, pairs):
    from scipy.stats import spearmanr
    res = replicate_correlation_table(mfi["m1"], pairs).set_index("celltype")
    assert list(res.columns) == ["pearson_r", "pearson_r2", "spearman_rho", "spearman_rho2"]
    x, y = mfi["m1"]["T"].loc[pairs.left].to_numpy(), mfi["m1"]["T"].loc[pairs.right].to_numpy()
    assert res.loc["T", "pearson_r"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert res.loc["T", "spearman_rho2"] == pytest.approx(spearmanr(x, y).statistic ** 2)
    same = pd.concat([mfi["m1"].loc[pairs.left], mfi["m1"].loc[pairs.left].set_axis(pairs.right)])
    perfect = replicate_correlation_table(same, pairs, celltypes=["B"])
    assert list(perfect.celltype) == ["B"] and perfect.spearman_rho[0] == pytest.approx(1)
