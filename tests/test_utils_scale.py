"""Tests of cytovanni.utils.scale: the "arcsinh" axis scale (raw data) and `set_arcsinh_ticks` (transformed
data)."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cytovanni  # noqa: F401  (registers the scale)
from cytovanni.utils import ArcSinhFormatter, set_arcsinh_ticks


def _ticks(axis):
    """Major ticks (data units, label) inside the view, and the minor ticks inside the view."""
    fig = axis.axes.figure
    fig.canvas.draw()
    lo, hi = sorted(axis.get_view_interval())
    major = [(loc, lb.get_text()) for loc, lb in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
             if lo <= loc <= hi]
    minor = [m for m in axis.get_minorticklocs() if lo <= m <= hi]
    return major, minor


def _value(label):
    """Linear value of a tick label (``$\\mathdefault{-10^{3}}$`` -> -1000, the 0 label -> 0)."""
    s = label.replace(r"$\mathdefault{", "").rstrip("}$")
    if s.startswith("0"):
        return 0.
    sign = -1. if s.startswith("-") else 1.
    m = float(s.lstrip("-").split(r"\times")[0]) if r"\times" in s else 1.
    return sign * m * 10.**int(s.split("^{")[1])


def _plot(cofactor, lim, figsize=(5, 4), onx=True, **kw):
    fig, ax = plt.subplots(figsize=figsize)
    (ax.set_xscale if onx else ax.set_yscale)("arcsinh", cofactor=cofactor, **kw)
    (ax.set_xlim if onx else ax.set_ylim)(*lim)
    return fig, ax, ax.xaxis if onx else ax.yaxis


def test_transform_roundtrip():
    fig, ax, axis = _plot(1500., (-1e4, 1e6))
    tr = axis.get_transform()
    x = np.array([-1e5, -10., 0., 3., 1e6])
    np.testing.assert_allclose(tr.transform(x), np.arcsinh(x / 1500.))
    np.testing.assert_allclose(tr.inverted().transform(tr.transform(x)), x, atol=1e-9)


@pytest.mark.parametrize("onx", [True, False])
def test_major_ticks_are_zero_and_powers_of_ten(onx):
    fig, ax, axis = _plot(5000., (-2e4, 1e6), onx=onx)
    major, minor = _ticks(axis)
    values = [_value(lb) for _, lb in major]
    np.testing.assert_allclose([loc for loc, _ in major], values)
    assert 0. in values and 1e5 in values and 1e6 in values and -1e4 in values
    assert all(v == 0 or np.log10(abs(v)) % 1 == 0 for v in values)
    assert np.any(np.isclose(minor, 2e5)) and np.any(np.isclose(minor, 9e4))  # 2..9 x 10^p between the labels
    plt.close(fig)


def test_labels_use_tick_font_mathtext():
    f = ArcSinhFormatter(1000.)
    assert f(1e4) == r"$\mathdefault{10^{4}}$" and f(-1e3) == r"$\mathdefault{-10^{3}}$"
    assert f(0.) == r"$\mathdefault{0}$" and f(2e3) == r"$\mathdefault{2\times10^{3}}$" and f(2.5e3) == ""
    assert f(-5e-2) == r"$\mathdefault{-5\times10^{-2}}$"
    assert f.format_data(1234.5) == "1234"


def test_small_axis_leaves_out_more_powers_near_zero():
    """The powers near 0 crowd together: a short axis labels fewer of them than a long one (never overlapping)."""
    smallest = {}
    for size in [1.5, 8.]:
        fig, ax, axis = _plot(1500., (-1e4, 1e6), figsize=(size, 3))
        major, _ = _ticks(axis)
        smallest[size] = min(abs(_value(lb)) for _, lb in major if _value(lb) > 0)
        fig.canvas.draw()
        boxes = sorted((t.get_window_extent() for t in axis.get_majorticklabels() if t.get_text()
                        and ax.get_xlim()[0] <= t.get_position()[0] <= ax.get_xlim()[1]), key=lambda b: b.x0)
        assert all(a.x1 <= b.x0 for a, b in zip(boxes[:-1], boxes[1:]))   # no overlapping labels
        plt.close(fig)
    assert smallest[1.5] > smallest[8.]


def test_minor_ticks_continue_below_the_labels():
    fig, ax, axis = _plot(1500., (-1e3, 1e6), figsize=(8, 3))
    major, minor = _ticks(axis)
    lowest = min(_value(lb) for _, lb in major if _value(lb) > 0)
    assert any(0 < m < lowest for m in minor)
    plt.close(fig)


def test_minpower_fixes_the_labels():
    fig, ax, axis = _plot(10., (-1e3, 1e5), minpower=2)
    values = [_value(lb) for _, lb in _ticks(axis)[0]]
    assert values == [-1e3, -1e2, 0., 1e2, 1e3, 1e4, 1e5]
    plt.close(fig)


@pytest.mark.parametrize("lim", [(-50., 50.), (100., 1e6), (-1e6, -100.), (2e3, 8e3), (0., 1.)])
def test_narrow_and_one_sided_ranges(lim):
    fig, ax, axis = _plot(5000., lim)
    major, minor = _ticks(axis)
    assert all(lim[0] <= loc <= lim[1] for loc, _ in major)
    plt.close(fig)


def test_scale_with_plotted_data_and_autoscale():
    fig, ax = plt.subplots()
    x = np.random.default_rng(0).normal(0, 1, 1000) * 1e4
    ax.scatter(x, x)
    ax.set_xscale("arcsinh", cofactor=5000.)
    fig.canvas.draw()
    lo, hi = ax.get_xlim()
    assert lo < x.min() and hi > x.max()
    plt.close(fig)


def test_set_arcsinh_ticks_matches_the_scale():
    """Transformed data labelled with `set_arcsinh_ticks` gets the ticks of the scale, at arcsinh(x / cofactor)."""
    cof, lim = 1500., (-2e4, 3e5)
    fig1, ax1, axis1 = _plot(cof, lim)
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.set_xlim(*np.arcsinh(np.asarray(lim) / cof))
    set_arcsinh_ticks(ax2, cof)
    m1, n1 = _ticks(axis1)
    m2, n2 = _ticks(ax2.xaxis)
    assert [lb for _, lb in m1] == [lb for _, lb in m2]
    np.testing.assert_allclose(np.arcsinh(np.array([loc for loc, _ in m1]) / cof), [loc for loc, _ in m2], atol=1e-9)
    np.testing.assert_allclose(np.arcsinh(np.array(n1) / cof), n2, atol=1e-9)
    assert ax2.get_xlim() == pytest.approx(tuple(np.arcsinh(np.asarray(lim) / cof)))
    plt.close(fig1); plt.close(fig2)


def test_set_arcsinh_ticks_y_axis_and_cofactor_marks():
    fig, ax = plt.subplots()
    ax.set_ylim(np.arcsinh(-1e3 / 10), np.arcsinh(1e5 / 10))
    set_arcsinh_ticks(ax, 10., axis="y", minpower=2, mark_cofactor=True)
    values = [_value(lb) for _, lb in _ticks(ax.yaxis)[0]]
    assert 1e2 in values and 10. not in values and 0. in values
    marks = [l.get_xydata()[0, 1] for l in ax.get_lines()]  # axhline at y = arcsinh(+-1)
    assert sorted(np.round(marks, 6)) == sorted(np.round([np.arcsinh(1), np.arcsinh(-1)], 6))
    plt.close(fig)


@pytest.mark.parametrize("onx", [True, False])
def test_zero_label_baseline_aligned_with_power_labels(onx):
    """The 0 label is moved down (by the extra height of the superscripts: all of it on x, half on y); the power
    labels are not moved."""
    fig, ax, axis = _plot(5000., (-2e4, 1e6), onx=onx)
    fig.canvas.draw()
    lo, hi = sorted(axis.get_view_interval())
    shifts = {}
    for tick in axis.get_major_ticks():
        if not tick.label1.get_text() or not lo <= tick.get_loc() <= hi:
            continue
        base = tick._get_text1_transform()[0]
        p0 = np.array(tick.label1.get_transform().transform((0, 0))) - np.array(base.transform((0, 0)))
        shifts[tick.label1.get_text()] = p0[1] * 72 / fig.dpi                # points, negative = down
    zero = shifts.pop(r"$\mathdefault{0}$")
    assert all(abs(v) < 1e-9 for v in shifts.values())
    assert (-4 < zero < -1) if onx else (-2 < zero < -.5)
    plt.close(fig)


@pytest.mark.parametrize("lim, expected", [((2e3, 8e3), [2e3, 5e3]),          # no power of ten: 2 and 5 x 10^p
                                           ((-8e3, -2e3), [-5e3, -2e3]),
                                           ((3e3, 5e4), None)])                 # one power, no 0: one more label
@pytest.mark.parametrize("onx", [True, False])
def test_two_labels_when_powers_are_missing(lim, expected, onx):
    fig, ax, axis = _plot(5000., lim, onx=onx)
    values = sorted(_value(lb) for _, lb in _ticks(axis)[0])
    assert len(values) == 2
    if expected is not None:
        assert values == expected
    else:
        assert 1e4 in values
    plt.close(fig)


def test_zero_and_one_power_are_enough():
    fig, ax, axis = _plot(5000., (-50., 5e3))
    values = sorted(_value(lb) for _, lb in _ticks(axis)[0])
    assert values == [0., 1e3]
    plt.close(fig)


def test_set_arcsinh_ticks_rejects_unknown_axis():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="axis"):
        set_arcsinh_ticks(ax, 10., axis="z")
    plt.close(fig)


def test_zero_gap_only_widens_the_space_next_to_zero():
    """A larger zero_gap drops powers next to the 0, not the decades further out."""
    out = {}
    for zg in [0., 6.]:
        fig, ax, axis = _plot(5000., (-1e6, 1e6), onx=False, zero_gap=zg, label_gap=.5)
        out[zg] = sorted(_value(lb) for _, lb in _ticks(axis)[0])
        plt.close(fig)
    near = lambda vals: min(abs(v) for v in vals if v != 0)
    assert near(out[6.]) > near(out[0.])
    assert 1e6 in out[6.] and 1e5 in out[6.]
