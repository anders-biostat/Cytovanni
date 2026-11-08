"""Manual gates: MinMaxGate, FluorescenceMinMaxGate, PolygonGate, LineGate, CutoffUnmxGate, UnmxMinGate, CombinedGate."""
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.gating import (CombinedGate, CutoffUnmxGate, FluorescenceMinMaxGate, LineGate, MinMaxGate,
                              PolygonGate, UnmxMinGate)
from cytovanni.gating.line import _closest_on_line, _distance_from_line


def gate_adata(x, columns=("FSC-A", "SSC-A"), layer="raw", unmx=None):
    """AnnData with `x` in ``layers[layer]`` (and X), optional unmixed DataFrame in ``obsm['unmx']``."""
    x = np.asarray(x, dtype=float)
    a = ad.AnnData(x.copy(), var=pd.DataFrame(index=list(columns)),
                   obs=pd.DataFrame(index=[str(i) for i in range(len(x))]))
    a.layers[layer] = x.copy()
    if unmx is not None:
        a.obsm["unmx"] = pd.DataFrame(np.asarray(unmx, dtype=float), index=a.obs_names,
                                      columns=[f"m{i}" for i in range(np.shape(unmx)[1])])
    return a


@pytest.fixture
def points(rng):
    return rng.normal(100, 50, (400, 2))


# ------------------------------------------------------------------------------------------------ MinMaxGate
def test_minmax_bounds_inclusive():
    a = gate_adata([[0, 5], [10, 10], [10.0001, 5], [-0.1, 5], [5, 5]])
    m = MinMaxGate(["FSC-A", "SSC-A"], 0, 10).apply(a)
    assert m.dtype == bool and m.shape == (5,)
    assert m.tolist() == [True, True, False, False, True]


def test_minmax_all_channels_must_be_inside():
    a = gate_adata([[5, 11], [5, 5]])
    assert MinMaxGate(["FSC-A", "SSC-A"], 0, 10).apply(a).tolist() == [False, True]
    assert MinMaxGate(["FSC-A"], 0, 10).apply(a).tolist() == [True, True]


def test_minmax_uses_layer_not_X():
    a = gate_adata([[5, 5], [50, 50]], layer="calibrated")
    a.X[:] = 1000
    g = MinMaxGate(["FSC-A", "SSC-A"], 0, 10, layer="calibrated")
    assert g.apply(a).tolist() == [True, False]
    with pytest.raises(KeyError):
        MinMaxGate(["FSC-A"], 0, 10).apply(a)  # default layer "raw" is not there


def test_minmax_addkey_writes_obs(points):
    a = gate_adata(points)
    g = MinMaxGate(["FSC-A", "SSC-A"], 50, 150)
    m = g.apply(a, addkey="keep")
    assert np.array_equal(a.obs["keep"].to_numpy(), m)
    m2 = g.apply(a)
    assert list(a.obs.columns) == ["keep"] and np.array_equal(m, m2)


def test_minmax_apply_df_equals_apply(points):
    a = gate_adata(points)
    df = pd.DataFrame(points, columns=["FSC-A", "SSC-A"])
    g = MinMaxGate(["SSC-A", "FSC-A"], 50, 150, layer="something_else")  # apply_df ignores layer
    a.layers["something_else"] = a.layers["raw"]
    assert np.array_equal(g.apply(a), g.apply_df(df))
    assert 0 < g.apply_df(df).mean() < 1


def test_minmax_copies_channels_and_repr():
    ch = ["FSC-A"]
    g = MinMaxGate(ch, 0, 1)
    ch.append("SSC-A")
    assert g.channels == ["FSC-A"]
    assert "[0, 1]" in repr(g) and "FSC-A" in repr(g)


def test_fluorescence_minmax_defaults(config):
    g = FluorescenceMinMaxGate(config)
    assert g.channels == synthetic.CHANNELS_FLUORESCENCE  # not the -H/-W channels
    assert g.v_min == -1e3 and g.v_max == config.max_linear_range * 1.5 and g.layer == "raw"
    assert FluorescenceMinMaxGate(config, max_tolerance=2).v_max == config.max_linear_range * 2
    assert FluorescenceMinMaxGate(config, max_tolerance=2, v_max=7.).v_max == 7.


def test_fluorescence_minmax_passes_layer_on(config):
    """R1-gate-layer: FluorescenceMinMaxGate uses its `layer` argument (it always used "raw")."""
    x = np.array([[1., 1., 1.], [1e9, 1., 1.]])
    a = gate_adata(x, columns=synthetic.CHANNELS_FLUORESCENCE, layer="calibrated")
    a.layers["raw"] = np.zeros_like(x)  # raw would keep both
    g = FluorescenceMinMaxGate(config, layer="calibrated")
    assert g.layer == "calibrated"
    assert g.apply(a).tolist() == [True, False]


def test_fluorescence_minmax_on_cell_file(cell_files, config):
    """As `gates` of readfcs_sample, the gate removes nothing from in-range synthetic cells, and exactly the outliers."""
    from cytovanni.io import readfcs_sample
    a = readfcs_sample(cell_files["A"], config)
    raw = a.to_df(layer="raw")[synthetic.CHANNELS_FLUORESCENCE].to_numpy()
    expected = np.all((raw >= -1e3) & (raw <= 3e5), axis=1)
    assert np.array_equal(FluorescenceMinMaxGate(config).apply(a), expected)


# ------------------------------------------------------------------------------------------------ PolygonGate
SQUARE = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])


def test_polygon_inside_outside():
    g = PolygonGate("FSC-A", "SSC-A", SQUARE)
    pts = np.array([[5, 5], [11, 5], [-1, 5], [5, 11], [9.9, 0.1]])
    assert g.apply_points(pts).tolist() == [True, False, False, False, True]


def test_polygon_apply_layer_and_df(points):
    v = np.array([[50., 50.], [150., 50.], [150., 150.], [50., 150.]])
    a = gate_adata(points, layer="calibrated")
    g = PolygonGate("FSC-A", "SSC-A", v, layer="calibrated")
    m = g.apply(a, addkey="poly")
    expected = np.all((points > 50) & (points < 150), axis=1)
    assert m.dtype == bool and np.array_equal(m, expected)
    assert np.array_equal(a.obs["poly"].to_numpy(), m)
    assert np.array_equal(g.apply_df(pd.DataFrame(points, columns=["FSC-A", "SSC-A"])), m)


def test_polygon_axis_order_matters():
    """x and y name the axes: vertices are (x, y); swapping the channels changes the gate for an asymmetric polygon."""
    tri = np.array([[0., 0.], [10., 0.], [10., 2.]])
    df = pd.DataFrame({"a": [8.], "b": [1.]})
    assert PolygonGate("a", "b", tri).apply_df(df).tolist() == [True]
    assert PolygonGate("b", "a", tri).apply_df(df).tolist() == [False]


def test_polygon_onlog_straight_edges_in_log_space_and_nonpositive_outside():
    v = np.array([[1., 1.], [100., 1.], [100., 100.], [1., 100.]])
    g = PolygonGate("a", "b", v, onlog=True)
    pts = np.array([[10., 10.], [0., 10.], [-5., 10.], [10., 0.], [-1e-3, -1e-3]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = g.apply_points(pts)
    assert m.tolist() == [True, False, False, False, False]
    # a triangle edge from (1, 100) to (100, 1): straight in log space, so (10, 10) lies on it and (5, 5) inside
    tri = PolygonGate("a", "b", np.array([[1., 1.], [100., 1.], [1., 100.]]), onlog=True)
    lin = PolygonGate("a", "b", np.array([[1., 1.], [100., 1.], [1., 100.]]))
    assert tri.apply_points(np.array([[5., 5.], [20., 20.]])).tolist() == [True, False]
    assert lin.apply_points(np.array([[20., 20.]])).tolist() == [True]


def test_polygon_obsm_axes():
    a = gate_adata(np.full((3, 2), 5.), unmx=[[5., 5.], [50., 5.], [5., 50.]])
    g = PolygonGate("m0", "SSC-A", SQUARE, x_obsm="unmx")  # x from obsm, y from the raw layer (all 5)
    pts = g.get_points(a)
    assert pts.shape == (3, 2) and np.allclose(pts[:, 0], [5, 50, 5]) and np.allclose(pts[:, 1], [5, 5, 5])
    assert g.apply(a).tolist() == [True, False, True]
    g2 = PolygonGate("m0", "m1", SQUARE, x_obsm="unmx", y_obsm="unmx")
    assert g2.apply(a).tolist() == [True, False, False]


def test_polygon_plot_draws_closed_polygon_and_log_axes():
    fig, ax = plt.subplots()
    PolygonGate("a", "b", SQUARE + 1, name="sq", onlog=True).plot(ax)
    line = ax.get_lines()[0]
    assert len(line.get_xdata()) == 5 and line.get_xdata()[0] == line.get_xdata()[-1]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log" and line.get_label() == "sq"


def test_polygon_plot_adata_runs(points):
    a = gate_adata(points)
    g = PolygonGate("FSC-A", "SSC-A", SQUARE * 10, name="p")
    g.plot_adata(a, s=1)
    assert "p" in plt.gca().get_title()
    assert "Polygon gate 'p'" in repr(g)


def test_polygon_list_vertices_work_for_apply_and_plot():
    g = PolygonGate("a", "b", SQUARE.tolist())
    assert isinstance(g.vertices, np.ndarray)
    assert g.apply_points(np.array([[5., 5.]])).tolist() == [True]
    fig, ax = plt.subplots()
    g.plot(ax)
    assert len(ax.lines) == 1


# ------------------------------------------------------------------------------------------------ LineGate
PTS_DIAG = np.array([[1., 0.], [0., 1.], [3., 1.], [-2., -3.], [-3., -2.]])  # below, above, below, below, above y=x
BELOW_DIAG = np.array([True, False, True, True, False])


@pytest.mark.parametrize("p1,p2", [((0, 0), (1, 1)), ((1, 1), (0, 0)), ((2, 2), (-1, -1))])
@pytest.mark.parametrize("below", [True, False])
def test_line_below_means_lower_y_for_any_point_order(p1, p2, below):
    """R2b-G2-08: `below=True` keeps lower y whatever the order of p1 and p2; `below=False` the higher y."""
    g = LineGate("x", "y", p1, p2, below=below)
    m = g.get_distance_points(PTS_DIAG) > 0
    assert np.array_equal(m, BELOW_DIAG if below else ~BELOW_DIAG)
    assert g.p1[0] < g.p2[0]


def test_line_vertical_raises():
    """R2b-G2-08: a vertical line (same x) has no below/above and raises ValueError."""
    with pytest.raises(ValueError):
        LineGate("x", "y", (1, 0), (1, 5))


def test_line_points_on_line_removed():
    g = LineGate("a", "b", (0, 0), (1, 1))
    assert (g.get_distance_points([[0.5, 0.5], [0.5, 0.2], [0.5, 0.8]]) > 0).tolist() == [False, True, False]
    assert g.apply_df(pd.DataFrame({"a": [0.5], "b": [0.5]})).tolist() == [False]


def test_line_distance_value_and_scaling():
    g = LineGate("a", "b", (0, 0), (2, 0))  # y = 0
    assert np.allclose(g.get_distance_points([[1., -3.], [1., 2.]]), [3., -2.])
    g.scale_x, g.scale_y = 1., 10.
    assert np.allclose(g.get_distance_points([[1., -3.]]), [30.])


def test_line_set_distance_scaling_factors_changes_distance_not_side(rng):
    x = np.abs(rng.normal(1000, 300, (300, 2))) * [1, 50]
    a = gate_adata(x)
    g = LineGate("FSC-A", "SSC-A", (0, 0), (2000, 100000))
    before = g.apply(a)
    d_before = g.get_distance(a)
    g.set_distance_scaling_factors(a)
    assert np.isclose(g.scale_x, 1 / x[:, 0].mean()) and np.isclose(g.scale_y, 1 / x[:, 1].mean())
    assert np.array_equal(g.apply(a), before)
    assert not np.allclose(g.get_distance(a), d_before)


def test_line_apply_df_equals_apply_and_addkey(points):
    """R2-E21: LineGate.apply_df exists and equals apply (also with scaling)."""
    a = gate_adata(points)
    g = LineGate("FSC-A", "SSC-A", [0, 0], [200, 180], below=False)
    g.set_distance_scaling_factors(a)
    m = g.apply(a, addkey="line")
    assert m.dtype == bool and 0 < m.mean() < 1
    assert np.array_equal(a.obs["line"].to_numpy(), m)
    assert np.array_equal(g.apply_df(pd.DataFrame(points, columns=["FSC-A", "SSC-A"])), m)


def test_line_distance_uses_explicit_2d_cross_product():
    """R2-C16a: `_distance_from_line` runs without the numpy 2 deprecation of 2-D np.cross; sign: positive right of p1->p2."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = _distance_from_line(np.array([0., 0.]), np.array([1., 0.]), np.array([[0., -1.], [0., 2.]]))
    assert np.allclose(d, [1., -2.])


def test_closest_on_line():
    p = _closest_on_line(np.array([0., 0.]), np.array([2., 2.]), np.array([[2., 0.], [0., 4.]]))
    assert np.allclose(p, [[1., 1.], [2., 2.]])


@pytest.mark.parametrize("below", [True, False])
def test_line_plot_shades_removed_side(below):
    """R2b-G2-08: plot shades above the line for below=True and below it (to the axis limit) for below=False."""
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 2); ax.set_ylim(-1, 2)
    LineGate("x", "y", (1, 1), (0, 0), below=below, name="L").plot(ax)
    v = ax.collections[0].get_paths()[0].vertices
    if below:
        assert np.all(v[:, 1] >= v[:, 0] - 1e-9)
    else:
        assert np.all(v[:, 1] <= v[:, 0] + 1e-9) and np.isclose(v[:, 1].min(), -1)
    assert ax.get_xlim() == (-1, 2) and ax.get_lines()[0].get_label() == "L"


def test_line_plot_without_shading_and_plot_adata(points):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    LineGate("x", "y", (0, 0), (1, 1)).plot(ax, mark_outside=False, addlegend=False)
    assert len(ax.collections) == 0
    a = gate_adata(points)
    g = LineGate("FSC-A", "SSC-A", (0, 0), (200, 200), name="l")
    g.plot_adata(a, s=1)
    assert "'l'" in plt.gca().get_title() and "Line gate 'l'" in repr(g)


# ------------------------------------------------------------------------------------------------ unmixed gates
@pytest.fixture
def unmx_adata():
    return gate_adata(np.zeros((5, 2)), unmx=[[0., 0.], [1., 0.], [2., 3.], [3., -1.], [4., 0.]])


@pytest.mark.parametrize("above,expected", [(True, [False, False, True, True, True]), (False, [True, True, True, False, False])])
def test_cutoff_unmx_inclusive(unmx_adata, above, expected):
    g = CutoffUnmxGate("m0", 2, above=above)
    r = g.apply(unmx_adata, addkey="cut")
    assert isinstance(r, np.ndarray) and r.dtype == bool
    assert r.tolist() == expected and unmx_adata.obs["cut"].tolist() == expected
    assert g.apply_df(unmx_adata.obsm["unmx"]).tolist() == expected  # R2-E21


def test_cutoff_unmx_key_and_repr(unmx_adata):
    unmx_adata.obsm["unmx_arcsinh"] = unmx_adata.obsm["unmx"] * 10
    g = CutoffUnmxGate("m0", 15, key="unmx_arcsinh")
    assert g.apply(unmx_adata).tolist() == [False, False, True, True, True]
    assert "above 15" in repr(g)
    assert "below" in repr(CutoffUnmxGate("m0", 1, above=False))


def test_unmx_min_strict_any_marker(unmx_adata):
    g = UnmxMinGate("unmx", 2.)
    m = g.apply(unmx_adata, addkey="anyhigh")
    assert isinstance(m, np.ndarray) and m.tolist() == [False, False, True, True, True]  # > 2 strict: (2, 3) via m1
    assert unmx_adata.obs["anyhigh"].tolist() == [False, False, True, True, True]
    assert g.apply_df(unmx_adata.obsm["unmx"]).tolist() == [False, False, True, True, True]  # R2-E21
    assert g.apply_df(unmx_adata.obsm["unmx"].to_numpy()).tolist() == [False, False, True, True, True]
    assert UnmxMinGate("unmx", 4.).apply_df(unmx_adata.obsm["unmx"]).tolist() == [False] * 5


def test_unmx_plots_run(unmx_adata):
    CutoffUnmxGate("m0", 2).plot_adata(unmx_adata)
    assert plt.gca().get_xlabel() == "m0"
    UnmxMinGate("unmx", 2.).plot_adata(unmx_adata)
    assert "unmx" in plt.gca().get_xlabel()
    assert "unmx" in repr(UnmxMinGate("unmx", 2.))


# ------------------------------------------------------------------------------------------------ CombinedGate
def test_combined_is_logical_and(points, rng):
    """R2-E21: CombinedGate.apply_df exists; apply and apply_df are the AND of the gates."""
    a = gate_adata(points)
    g1 = MinMaxGate(["FSC-A", "SSC-A"], 20, 180)
    g2 = LineGate("FSC-A", "SSC-A", [0, 0], [200, 180], below=False)
    cg = CombinedGate([g1, g2])
    m = cg.apply(a)
    assert np.array_equal(m, g1.apply(a) & g2.apply(a))
    assert 0 < m.mean() < min(g1.apply(a).mean(), g2.apply(a).mean())
    df = pd.DataFrame(points, columns=["FSC-A", "SSC-A"])
    assert np.array_equal(cg.apply_df(df), m)


def test_combined_with_unmx_gate_series(unmx_adata):
    cg = CombinedGate([CutoffUnmxGate("m0", 1), UnmxMinGate("unmx", 2.5)])
    assert cg.apply(unmx_adata).tolist() == [False, False, True, True, True]


def test_gates_on_raw_fcs_dataframe(cell_files):
    """`apply_df` works on the raw event DataFrame of an .fcs file (what make_calibrated_fcs passes)."""
    from cytovanni.io import readfcs_dataframe
    df = readfcs_dataframe(cell_files["A"])
    cg = CombinedGate([MinMaxGate(["FSC-A", "SSC-A"], 0, 4e5),
                       PolygonGate("FSC-A", "SSC-A", np.array([[0., 0.], [3e4, 0.], [3e4, 3e4], [0., 3e4]])),
                       LineGate("FSC-A", "SSC-A", (0, 0), (1e4, 5e3))])
    m = cg.apply_df(df)
    assert m.shape == (len(df),) and 0 < m.mean() < 1
