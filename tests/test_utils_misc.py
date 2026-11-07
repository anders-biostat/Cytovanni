"""Tests of cytovanni.utils.misc: colours, legends, subplots, arcsinh axis labels, subsampling, concatenation."""
import re

import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cytovanni.utils import (anndata_concat, anndata_subsample, check_pulse_width_independence, cmap_to_legendhandles,
                             cmap_to_legendhandles_wmarker, df_subsample, get_cmap, get_cmap_and_markers,
                             label_axis_ArcSinh, labelscolors_to_legendhandles, palette_20, palette_28, printwtime,
                             silent_log, subsample_label_indices, textcolor, unrolled_subplots, wavelength_to_hex,
                             wavelength_to_rgb)
from cytovanni.utils.misc import MARKERS, get_hex_distinct, get_scanpy_palette_20

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ------------------------------------------------------------------ textcolor
def test_textcolor_wraps_text():
    assert textcolor.toBOLD("x") == textcolor.BOLD + "x" + textcolor.END
    assert textcolor.toCOLOR("x", "RED") == "\033[91mx\033[0m"


# ------------------------------------------------------------------ palettes and colours
def test_palettes():
    assert len(palette_20) == 20 and len(palette_28) == 28
    assert get_scanpy_palette_20() == palette_20
    assert all(HEX.match(c) for c in palette_20) and len(set(palette_20)) == 20
    assert palette_20[2] == "#279e68" and palette_20[4] == "#aa40fc"  # scanpy's replacements


@pytest.mark.parametrize("n", [1, 20, 21, 28, 29, 60])
def test_get_hex_distinct(n):
    cols = get_hex_distinct(n)
    assert len(cols) == n and len(set(cols)) == n and all(HEX.match(c) for c in cols)
    if n <= 20:
        assert cols == palette_20[:n]
    elif n <= 28:
        assert cols == palette_28[:n]


def test_get_hex_distinct_reproducible_and_reseeds_global_state():
    """Documented side effect for N > 28: np.random.seed(0)."""
    a = get_hex_distinct(40); r1 = np.random.rand()
    np.random.seed(123)
    b = get_hex_distinct(40); r2 = np.random.rand()
    assert a == b and r1 == r2


@pytest.mark.parametrize("wl, rgb", [(380, None), (440, (0, 0, 255)), (490, (0, 255, 255)), (510, (0, 255, 0)),
                                     (580, (255, 255, 0)), (645, (255, 0, 0)), (700, None), (300, (0, 0, 0)), (800, (0, 0, 0))])
def test_wavelength_to_rgb(wl, rgb):
    out = wavelength_to_rgb(wl)
    assert isinstance(out, tuple) and len(out) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in out)
    if rgb is not None:
        assert out == rgb


def test_wavelength_to_hex():
    assert wavelength_to_hex(550) == "#a2ff00"  # docstring example
    assert wavelength_to_hex(800) == "#000000"


def test_get_cmap_sorted_unique_labels():
    cmap = get_cmap(["b", "a", "b", "c"])
    assert list(cmap) == ["a", "b", "c"] and list(cmap.values()) == palette_20[:3]
    rgb = get_cmap(["b", "a"], ishex=False)
    assert rgb["a"] == matplotlib.colors.to_rgb(palette_20[0])


def test_get_cmap_and_markers_cycles():
    keys = [f"k{i:02d}" for i in range(12)]
    cmap, markers = get_cmap_and_markers(keys + keys[:3])
    assert list(cmap) == keys and list(markers) == keys
    assert [markers[k] for k in keys] == MARKERS + MARKERS[:2]


def test_legend_handles():
    cmap = {"a": "#ff0000", "b": "#00ff00"}
    h = cmap_to_legendhandles(cmap, markersize=7)
    assert [x.get_label() for x in h] == ["a", "b"] and h[0].get_markerfacecolor() == "#ff0000"
    assert h[0].get_markeredgecolor() == "#ff0000" and h[0].get_markersize() == 7
    h = cmap_to_legendhandles(cmap, include_label=False, outline=True)
    assert [x.get_label() for x in h] == ["", ""] and h[1].get_markeredgecolor() == "black"
    h = cmap_to_legendhandles_wmarker(cmap, {"a": "s", "b": "v"}, outline=True)
    assert [x.get_marker() for x in h] == ["s", "v"] and h[0].get_markeredgecolor() == "black"
    with pytest.raises(KeyError):
        cmap_to_legendhandles_wmarker(cmap, {"a": "s"})
    h = labelscolors_to_legendhandles(["x", "y"], ["red", "blue"], markersize=3)
    assert [x.get_label() for x in h] == ["x", "y"] and h[1].get_markerfacecolor() == "blue"


# ------------------------------------------------------------------ unrolled_subplots
@pytest.mark.parametrize("n, ncol, grid", [(5, 2, (3, 2)), (3, 4, (1, 3)), (3, 1, (3, 1)), (8, 4, (2, 4))])
def test_unrolled_subplots(n, ncol, grid):
    fig, axes = unrolled_subplots(n, Ncol=ncol, elsize=(2, 1))
    assert len(axes) == n
    assert np.allclose(fig.get_size_inches(), [2 * grid[1], 1 * grid[0]])
    assert sum(a.axison for a in fig.axes) == n and len(fig.axes) == grid[0] * grid[1]
    if n > 1 and (grid[0] == 1 or grid[1] == 1):
        assert isinstance(axes, np.ndarray)


def test_unrolled_subplots_single():
    fig, axes = unrolled_subplots(1)
    assert len(axes) == 1 and isinstance(axes[0], matplotlib.axes.Axes)


# ------------------------------------------------------------------ small helpers
def test_printwtime(capsys):
    printwtime("hello")
    assert re.match(r"^\d\d:\d\d:\d\d - hello$", capsys.readouterr().out.strip())


def test_silent_log(no_warnings):
    out = silent_log(np.array([np.e, 0., -1., np.nan]))
    assert out[0] == pytest.approx(1.) and out[1] == -np.inf and np.isnan(out[2]) and np.isnan(out[3])
    assert silent_log(1.) == 0.


# ------------------------------------------------------------------ label_axis_ArcSinh
def _ticks(ax, onx=True):
    axis = ax.xaxis if onx else ax.yaxis
    major = axis.get_majorticklocs()
    fmt = axis.get_major_formatter()  # the labels set with the fixed locations, by position
    labels = [fmt(loc, pos) for pos, loc in enumerate(major)]
    return major, labels, axis.get_minorticklocs()


def test_label_axis_ArcSinh_major_ticks_and_labels():
    cof = 1500.
    fig, ax = plt.subplots()
    lim = (np.arcsinh(-2e4 / cof), np.arcsinh(3e5 / cof))
    ax.set_xlim(*lim)
    label_axis_ArcSinh(ax, cof, minpower=3)
    major, labels, minor = _ticks(ax)
    assert ax.get_xlim() == pytest.approx(lim)  # limits kept
    lin = np.sinh(major) * cof
    shown = [(lb, v) for lb, v in zip(labels, lin) if lim[0] <= np.arcsinh(v / cof) <= lim[1]]
    assert [lb for lb, _ in shown] == [r"$-10^4$", r"$-10^3$", r"$0$", r"$10^3$", r"$10^4$", r"$10^5$"]
    for lb, v in shown:  # tick positions are arcsinh(10^p / cofactor)
        p = 0 if lb == "$0$" else int(lb.strip("$").split("^")[1])
        assert v == pytest.approx(0 if p == 0 else np.sign(v) * 10.**p, rel=1e-9)
    minor_lin = np.sinh(minor) * cof
    assert np.any(np.isclose(minor_lin, 2e3)) and np.any(np.isclose(minor_lin, -5e3)) and np.any(np.isclose(minor_lin, 100.))


def test_label_axis_ArcSinh_y_axis_and_cofactor_marks():
    fig, ax = plt.subplots()
    ax.set_ylim(np.arcsinh(-1e3 / 10), np.arcsinh(1e5 / 10))
    label_axis_ArcSinh(ax, 10., onx=False, minpower=2, mark_cofactor=True)
    _, labels, _ = _ticks(ax, onx=False)
    assert "$10^2$" in labels and "$10^1$" not in labels and "$0$" in labels
    marks = [l.get_xydata()[0, 1] for l in ax.get_lines()]  # axhline at y = arcsinh(+-1)
    assert sorted(np.round(marks, 6)) == sorted(np.round([np.arcsinh(1), np.arcsinh(-1)], 6))


def test_label_axis_ArcSinh_x_cofactor_marks():
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 8)
    label_axis_ArcSinh(ax, 100., mark_cofactor=True)
    assert sorted(round(l.get_xdata()[0], 6) for l in ax.get_lines()) == [round(-np.arcsinh(1), 6), round(np.arcsinh(1), 6)]


@pytest.mark.parametrize("lim", [(-.1, .1), (.01, .02), (-.02, -.01), (0, .5), (-3, 8), (2, 9)])
def test_label_axis_ArcSinh_narrow_and_one_sided_ranges(lim):
    """R2-E3: a single major tick (e.g. only 0) gives no minor ticks instead of crashing."""
    fig, ax = plt.subplots()
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    label_axis_ArcSinh(ax, 1.)
    label_axis_ArcSinh(ax, 1., onx=False)
    assert ax.get_xlim() == pytest.approx(lim)


# ------------------------------------------------------------------ anndata_concat / subsample
def _ad(n, val=0., names=None):
    a = anndata.AnnData(np.full((n, 2), val, dtype=np.float32) + np.arange(n, dtype=np.float32)[:, None],
                        var=pd.DataFrame({"pnn": ["a", "b"]}, index=["a", "b"]))
    if names is not None:
        a.obs_names = names
    return a


def test_anndata_concat_inner_outer_merge_unique(recwarn):
    a, b = _ad(3), _ad(4, 10.)
    c = anndata_concat([a, b])
    assert not [w for w in recwarn if "not unique" in str(w.message)]
    assert c.shape == (7, 2) and list(c.obs_names) == ["0", "1", "2", "0", "1", "2", "3"]  # duplicate warning silenced
    assert "pnn" not in c.var  # merge=None drops var columns
    assert "pnn" in anndata_concat([a, b], merge="same").var
    assert anndata_concat([a, b], make_unique=True).obs_names.is_unique
    b2 = anndata.AnnData(np.ones((2, 3), dtype=np.float32), var=pd.DataFrame(index=["a", "b", "x"]))
    assert anndata_concat([a, b2]).shape[1] == 2 and anndata_concat([a, b2], join="outer").shape[1] == 3


def test_anndata_concat_permute_with_non_unique_names():
    """R3b-B10: permute selects by position, also with duplicate event names."""
    a = _ad(10); b = _ad(10, 100.)
    c = anndata_concat([a, b], permute=True)
    assert c.shape == (20, 2) and not c.is_view
    assert sorted(c.X[:, 0].tolist()) == sorted(np.r_[np.arange(10), 100 + np.arange(10)].tolist())


def test_anndata_subsample_copy_semantics():
    """R2-E4: copy=True always returns a new object; copy=False returns the input when small, else a view."""
    a = _ad(50)
    assert anndata_subsample(a, 100) is not a and anndata_subsample(a, 100, copy=False) is a
    s = anndata_subsample(a, 10)
    assert s.shape == (10, 2) and not s.is_view
    assert anndata_subsample(a, 10, copy=False).is_view
    assert len(np.unique(s.X[:, 0])) == 10  # without replacement


def test_df_subsample_copy_semantics():
    """R2-E4: same for DataFrames (a subsample is always a copy)."""
    df = pd.DataFrame(np.arange(100.).reshape(50, 2))
    assert df_subsample(df, 100) is not df and df_subsample(df, 100).equals(df)
    assert df_subsample(df, 100, copy=False) is df
    s = df_subsample(df, 10, copy=False)
    assert s.shape == (10, 2) and s.index.is_unique
    s.iloc[0, 0] = -1.
    assert (df.to_numpy() >= 0).all()


def test_subsample_by_position_with_non_unique_names():
    """R3b-B10: duplicate names are fine; same draws as the old name-based selection for unique names."""
    a = _ad(10, names=["x"] * 10)
    assert anndata_subsample(a, N=4).shape[0] == 4
    d = pd.DataFrame({"x": np.arange(10)}, index=["a"] * 10)
    s = df_subsample(d, N=4)
    assert s.shape[0] == 4 and len(set(s["x"])) == 4
    u = _ad(10)
    np.random.seed(3); new = anndata_subsample(u, N=4).X
    np.random.seed(3); old = u[u.obs.index[np.random.permutation(10)[:4]]].X
    assert np.array_equal(new, old)


# ------------------------------------------------------------------ subsample_label_indices
def test_subsample_label_indices_to_rarest():
    labels = np.array(["a"] * 2 + ["b"] * 10 + ["c"] * 3)
    idx = subsample_label_indices(labels)
    assert len(idx) == 6 and len(set(idx)) == 6
    assert sorted(pd.Series(labels[idx]).value_counts().to_dict().items()) == [("a", 2), ("b", 2), ("c", 2)]


def test_subsample_label_indices_max_multiple():
    labels = pd.Series(["a"] * 2 + ["b"] * 10 + ["c"] * 3, index=["i"] * 15)  # positions, not labels
    idx = subsample_label_indices(labels, max_multiple=2)
    counts = pd.Series(labels.to_numpy()[idx]).value_counts().to_dict()
    assert counts == {"a": 2, "b": 4, "c": 3}  # c: 3 <= 2*2 kept complete
    assert idx.dtype.kind == "i"


# ------------------------------------------------------------------ check_pulse_width_independence
def _pulse_adata(rng, independent):
    n = 2000
    A = rng.uniform(1e3, 1e5, n); H = A / rng.uniform(1, 3, n)
    W = rng.uniform(1, 3, n) if independent else A / H
    a = anndata.AnnData(np.zeros((n, 3), np.float32), var=pd.DataFrame(index=["FSC-A", "FSC-H", "FSC-W"]))
    a.layers["raw"] = np.vstack([A, H, W]).T.astype(np.float32)
    a.layers["other"] = a.layers["raw"].copy()
    return a


def test_check_pulse_width_independence(rng, monkeypatch):
    assert check_pulse_width_independence(_pulse_adata(rng, True), "FSC") in (True, np.True_)
    assert not check_pulse_width_independence(_pulse_adata(rng, False), "FSC", layer="other")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    check_pulse_width_independence(_pulse_adata(rng, False), "FSC", plot=True)


@pytest.mark.xfail(strict=True, reason="R4 utils bug 2: one event with height 0 makes the correlation NaN, so an "
                                       "independently measured width is reported as computed (False)")
def test_check_pulse_width_independence_with_zero_height(rng):
    a = _pulse_adata(rng, True)
    a.layers["raw"][0, 1] = 0.  # FSC-H of one event
    assert check_pulse_width_independence(a, "FSC")


@pytest.mark.xfail(strict=True, reason="R4 utils bug 3: permute=True indexes the concatenated object outside the "
                                       "warning filter, so the duplicate-name warning the docstring says is silenced appears")
def test_anndata_concat_permute_silences_duplicate_name_warning(recwarn):
    a, b = _ad(5), _ad(5, 100.)
    anndata_concat([a, b], permute=True)
    assert not [w for w in recwarn if "not unique" in str(w.message)]
