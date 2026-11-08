"""AutoSpill tessellation gating (gating/autospill.py) and gate_scatter_bead_sample (gating/bead.py)."""
import os
import warnings
from unittest import mock

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

import synthetic
from cytovanni.exceptions import ScatterGatingChannelException
from cytovanni.gating import AutoSpillTesselationGating, gate_scatter_bead_sample
from cytovanni.gating.autospill import DensityGate, HistogramTesselator, ScaledVoronoi, _BinnedKDE


def bead_scatter(n=20000, seed=0, doublet_fraction=.1, debris_fraction=.08):
    """FSC-A, SSC-A of beads: singlets around (1e5, 5e4), doublets around (1.6e5, 8e4), uniform debris.

    Returns x, y and the true class of every event (0 singlet, 1 doublet, 2 debris).
    """
    rng = np.random.default_rng(seed)
    nd, nb = int(n * doublet_fraction), int(n * debris_fraction)
    ns = n - nd - nb
    x = np.concatenate([rng.normal(1e5, 4e3, ns), rng.normal(1.6e5, 6e3, nd), rng.uniform(1e3, 3e5, nb)])
    y = np.concatenate([rng.normal(5e4, 2e3, ns), rng.normal(8e4, 3e3, nd), rng.uniform(1e3, 1.5e5, nb)])
    label = np.repeat([0, 1, 2], [ns, nd, nb])
    return x, y, label


def bead_adata(x, y, extra=None):
    cols = {"FSC-A": x, "SSC-A": y, **({} if extra is None else extra)}
    X = np.vstack(list(cols.values())).T
    a = ad.AnnData(var=pd.DataFrame(index=list(cols)), obs=pd.DataFrame(index=[str(i) for i in range(len(x))]))
    a.layers["raw"] = X
    a.uns["name_uid"] = "beads (SIM-0)"
    return a


@pytest.fixture(scope="module")
def beads():
    return bead_scatter()


@pytest.fixture(scope="module")
def gate(beads):
    x, y, _ = beads
    return AutoSpillTesselationGating(x, y, trim_beadmode=True, final_density_cutoff=.2, bound_density_mad_factor=4.)


# ------------------------------------------------------------------------------------------------ binned KDE
def test_binned_kde_matches_exact_kde(rng):
    """R1-D1: the binned KDE (512 grid, FFT) approximates scipy's exact gaussian_kde (error ~1e-3 of the maximum)."""
    X = np.vstack([np.concatenate([rng.normal(0, 1, 3000), rng.normal(4, .5, 1000)]),
                   np.concatenate([rng.normal(0, 2, 3000), rng.normal(1, .5, 1000)])])
    pts = np.vstack([rng.uniform(-3, 6, 500), rng.uniform(-5, 5, 500)])
    exact = gaussian_kde(X)(pts)
    approx = _BinnedKDE(X)(pts)
    assert approx.shape == (500,) and np.all(approx >= 0)
    assert np.abs(approx - exact).max() < 2e-3 * exact.max()
    # bandwidth and covariance are those of gaussian_kde
    assert np.allclose(_BinnedKDE(X).covariance, gaussian_kde(X).covariance)


def test_binned_kde_weighted_and_far_away_zero(rng):
    X = rng.normal(0, 1, (2, 2000))
    w = rng.uniform(.1, 1, 2000)
    pts = np.array([[0., .5, 100.], [0., -.5, 100.]])
    exact = gaussian_kde(X, weights=w)(pts)
    approx = _BinnedKDE(X, weights=w)(pts)
    assert np.allclose(approx[:2], exact[:2], rtol=.02)
    assert approx[2] == 0.


# ------------------------------------------------------------------------------------------------ HistogramTesselator
def two_clusters(rng, n=4000, sep=40., sizes=(.7, .3)):
    a, b = int(n * sizes[0]), int(n * sizes[1])
    x = np.concatenate([rng.normal(30, 4, a), rng.normal(30 + sep, 4, b)])
    y = np.concatenate([rng.normal(50, 4, a), rng.normal(50, 4, b)])
    return x, y, np.repeat([0, 1], [a, b])


def test_tesselator_single_cluster_one_tile(rng):
    x, y = rng.normal(50, 5, 3000), rng.normal(50, 5, 3000)
    h = HistogramTesselator(x, y, (0, 100), (0, 100))
    assert h.voronoi is None and h.maxima.shape == (1, 2)
    assert np.allclose(h.maxima[0], [50, 50], atol=4)
    assert h.voronoi_label_ismain.all() and h.saddle_ratio == {}
    assert h.density.shape == (100, 100) and h.logdensity.min() >= h.MIN_LOGDENSITY


def test_tesselator_two_clusters_main_by_count_and_density(rng):
    x, y, lab = two_clusters(rng, sizes=(.7, .3))
    h = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=1.)
    assert len(h.maxima) == 2 and isinstance(h.voronoi, ScaledVoronoi)
    # dummy sites were added for the two-maxima case
    assert h.maxima_indices.shape[0] > 2
    assert h.voronoi_label.shape == x.shape
    # main tile by count: the larger cluster
    assert (h.voronoi_label_ismain == (lab == 0)).mean() > .99
    # by density: a dense small cluster wins over a wide large one
    x2 = np.concatenate([rng.normal(30, 8, 3000), rng.normal(75, 1.5, 800)])
    y2 = np.concatenate([rng.normal(50, 8, 3000), rng.normal(50, 1.5, 800)])
    hc = HistogramTesselator(x2, y2, (0, 100), (0, 100), bw_factor=1.)
    hd = HistogramTesselator(x2, y2, (0, 100), (0, 100), bw_factor=1., main_tile="density")
    assert hc.voronoi_label_ismain[:3000].mean() > .9 and hc.voronoi_label_ismain[3000:].mean() < .1
    assert hd.voronoi_label_ismain[3000:].mean() > .9 and hd.voronoi_label_ismain[:3000].mean() < .2


def test_tesselator_collinear_maxima(rng):
    """Three collinear maxima make qhull fail; dummy sites are added and every cluster gets its own tile."""
    x = np.concatenate([rng.normal(c, 3, 1500) for c in (20, 50, 80)])
    y = rng.normal(50, 3, 4500)
    h = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=.5)
    assert len(h.maxima) == 3 and np.allclose(h.maxima[:, 1], h.maxima[0, 1])
    assert h.maxima_indices.shape[0] == 6
    lab = np.repeat([0, 1, 2], 1500)
    for k in range(3):
        assert len(np.unique(h.voronoi_label[lab == k])) == 1
    assert len(np.unique(h.voronoi_label)) == 3


def test_tesselator_invalid_main_tile(rng):
    x, y, _ = two_clusters(rng)
    with pytest.raises(ValueError):
        HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=1., main_tile="largest")


def test_tesselator_merge_tiles_by_saddle(rng):
    """Connected maxima (ridge >= merge_tiles_min_saddle x main maximum) are merged into the main tiles."""
    x, y, lab = two_clusters(rng, sep=14., sizes=(.6, .4))
    h = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=.7, neigh_size=2)
    assert len(h.maxima) >= 2
    r = h.saddle_ratio
    assert set(r) == set(range(len(h.maxima))) - {int(np.unique(h.voronoi_label[h.voronoi_label_ismain])[0])}
    assert all(0 <= v <= 1 for v in r.values())
    merged = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=.7, neigh_size=2, merge_tiles_min_saddle=0.)
    assert merged.voronoi_label_ismain.all()
    strict = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=.7, neigh_size=2, merge_tiles_min_saddle=1.01)
    assert np.array_equal(strict.voronoi_label_ismain, h.voronoi_label_ismain)


def test_tesselator_lower_edge_exclusion(rng):
    """Maxima in the low-x or low-y strip (density_exclusion_corner) are dropped; none left raises RuntimeError."""
    x = np.concatenate([rng.normal(3, 1, 2000), rng.normal(60, 4, 1000)])
    y = np.concatenate([rng.normal(60, 4, 2000), rng.normal(60, 4, 1000)])  # big cluster in the low-x strip only
    h = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=1., density_exclusion_corner=.1)
    assert len(h.maxima) == 1 and h.maxima[0, 0] > 40
    with pytest.raises(RuntimeError):
        HistogramTesselator(rng.normal(3, 1, 500), rng.normal(3, 1, 500), (0, 100), (0, 100), density_exclusion_corner=.2)


def test_tesselator_index_point_maps_are_inverse(rng):
    x, y, _ = two_clusters(rng)
    h = HistogramTesselator(x, y, (0, 100), (0, 200), Nx=50, Ny=50)
    P = np.array([[3., 7.], [49., 49.], [0., 0.]])
    assert np.allclose(h.point_to_ind(h.ind_to_point(P)), P)
    assert np.allclose(h.ind_to_point(np.array([[0., 0.]])), h.bins_center_corners[[0]])


def test_tesselator_nonsquare_grid(rng):
    """Nx and Ny are documented as separate grid sizes; a non-square grid should work."""
    x, y, _ = two_clusters(rng)
    h = HistogramTesselator(x, y, (0, 100), (0, 200), Nx=50, Ny=80)
    assert h.density.shape == (50, 80)


def test_tesselator_rectangle_from_main(rng):
    x, y = rng.normal(50, 5, 3000), rng.normal(40, 2, 3000)
    h = HistogramTesselator(x, y, (0, 100), (0, 100))
    (x0, x1), (y0, y1) = h.rectangle_from_main(3.)
    mad = lambda v: np.mean(np.abs(v - v.mean()))
    assert np.isclose((x0 + x1) / 2, np.median(x)) and np.isclose(x1 - x0, 6 * mad(x))
    assert np.isclose((y0 + y1) / 2, np.median(y)) and np.isclose(y1 - y0, 6 * mad(y))


def test_tesselator_subsample_affects_assignment_only(rng):
    x, y, _ = two_clusters(rng)
    h = HistogramTesselator(x, y, (0, 100), (0, 100), Nsubsample=500)
    full = HistogramTesselator(x, y, (0, 100), (0, 100))
    assert len(h.x) == 500 and h.voronoi_label.shape == (500,)
    assert np.allclose(h.density, full.density)


def test_scaled_voronoi_assigns_nearest_site_in_index_space(rng):
    x, y, _ = two_clusters(rng, n=3000)
    h = HistogramTesselator(x, y, (0, 100), (0, 1e5 / 2), bw_factor=1.)  # very different axis scales
    vor = h.voronoi
    X = np.vstack([x, y]).T[:200]
    d = ((h.point_to_ind(X)[:, None] - vor.points_index[None]) ** 2).sum(-1)
    assert np.array_equal(vor.assign_group(X), d.argmin(1))
    assert np.allclose(vor.points, h.ind_to_point(vor.points_index))
    ax = vor.plot(show_vertices=True)
    assert len(ax.collections) == 2


def test_tesselator_plot_runs(rng):
    x, y, _ = two_clusters(rng)
    ax = HistogramTesselator(x, y, (0, 100), (0, 100), bw_factor=1.).plot(max_points=100)
    assert ax.get_xlim() == (0, 100)


# ------------------------------------------------------------------------------------------------ DensityGate
def test_density_gate_hull_of_densest_events(rng):
    X = rng.normal(0, 1, (3000, 2))
    g = DensityGate(X, cutoff=.33)
    assert g.density_scaled.min() == 0 and g.density_scaled.max() == 1
    assert g.vertices_hull.shape[1] == 2 and np.array_equal(g.vertices_hull[0], g.vertices_hull[-1])
    inside = g.hull_gatemask(X)
    assert inside.dtype == bool
    # at least the densest 67 % are inside (hull vertices lie on the boundary); the least dense events are outside
    assert .6 < inside.mean() < .85
    assert inside[g.density_scaled > .4].mean() > .98
    assert inside[g.density_scaled < .02].mean() < .1
    assert DensityGate(X, cutoff=.8).hull_gatemask(X).mean() < inside.mean()


def test_density_gate_subsample_and_plot(rng):
    X = rng.normal(0, 1, (3000, 2))
    g = DensityGate(X, Nsubsample=1000)
    assert g.X.shape == (1000, 2) and g.density_scaled.shape == (1000,)
    ax = g.plot(xlim=(-4, 4), ylim=(-4, 4), max_points=200)
    assert ax.get_xlim() == (-4, 4)


# ------------------------------------------------------------------------------------------------ AutoSpillTesselationGating
def test_autospill_recovers_singlets(beads, gate):
    """C: the gate keeps mostly singlets, excludes doublets and debris."""
    x, y, label = beads
    m = gate.get_gatemask(x, y)
    assert m.dtype == bool and m.shape == x.shape
    assert m[label == 0].mean() > .6
    assert m[label == 1].mean() < .01
    assert m[label == 2].mean() < .05
    assert (label[m] == 0).mean() > .99


def test_autospill_deterministic_and_independent_of_global_seed(beads):
    """R1-D1: no random subsampling by default, so two fits give the same gate whatever the global numpy state."""
    x, y, _ = beads
    np.random.seed(1)
    a = AutoSpillTesselationGating(x, y, trim_beadmode=True)
    np.random.seed(2)
    b = AutoSpillTesselationGating(x, y, trim_beadmode=True)
    assert AutoSpillTesselationGating.subsample_kde_estimates is None
    assert np.array_equal(a.densitygate.vertices_hull, b.densitygate.vertices_hull)
    assert np.array_equal(a.get_gatemask(x, y), b.get_gatemask(x, y))


def test_autospill_no_runtime_warning(beads):
    """R2b-G2-13: the binned KDE is 0 far from the data; log(0) no longer warns."""
    x, y, _ = beads
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        AutoSpillTesselationGating(x, y, trim_beadmode=True)


def test_autospill_trim_beadmode_bounds(beads, gate):
    x, y, _ = beads
    assert np.allclose(gate.bounds_x, [.5 * np.median(x), 1.5 * np.median(x)])
    assert np.allclose(gate.bounds_y, [.5 * np.median(y), 1.5 * np.median(y)])
    # events at the bounds (clipped) are left out of the tessellations
    assert len(gate.x) < len(x) and gate.x.min() > gate.bounds_x[0] and gate.x.max() < gate.bounds_x[1]


def test_autospill_percentile_trimming(beads):
    x, y, _ = beads
    g = AutoSpillTesselationGating(x, y, trim_x_min=2, trim_x_max=98)
    assert np.allclose(g.bounds_x, np.percentile(x, [2, 98]))
    assert np.allclose(g.bounds_y, np.percentile(y, [1, 99]))


def test_autospill_kwargs_override_instance_only(beads):
    x, y, _ = beads
    g = AutoSpillTesselationGating(x, y, trim_beadmode=True, final_density_cutoff=.5, name_x="X")
    assert g.final_density_cutoff == .5 and g.name_x == "X"
    assert AutoSpillTesselationGating.final_density_cutoff == .33 and AutoSpillTesselationGating.name_x == "FSC-A"
    # documented: names are not checked, a misspelled one is set and has no effect
    t = AutoSpillTesselationGating(x, y, trim_beadmode=True, final_densty_cutoff=.9)
    assert t.final_density_cutoff == .33 and t.final_densty_cutoff == .9


def test_autospill_region_from_bound_tile(gate):
    rx, ry = gate.histogram_bound.rectangle_from_main(gate.bound_density_mad_factor)
    assert np.allclose(gate.region_x, rx) and np.allclose(gate.region_y, ry)
    assert np.all((gate.x_final > rx[0]) & (gate.x_final < rx[1]))


def test_autospill_final_cutoff_tightens_gate(beads, gate):
    x, y, _ = beads
    loose = AutoSpillTesselationGating(x, y, trim_beadmode=True, final_density_cutoff=.05, bound_density_mad_factor=4.)
    assert loose.get_gatemask(x, y).mean() > gate.get_gatemask(x, y).mean()


def test_autospill_plot(gate):
    fig, ax = gate.plot(title="beads", figsize=(8, 2))
    assert len(ax) == 4 and fig._suptitle.get_text() == "beads"
    assert "%" in ax[3].get_title()


def test_autospill_verbose_plots(beads):
    x, y, _ = beads
    with mock.patch.object(AutoSpillTesselationGating, "plot") as p:
        AutoSpillTesselationGating(x, y, trim_beadmode=True, verbose=True)
    p.assert_called_once()


# ------------------------------------------------------------------------------------------------ gate_scatter_bead_sample
def test_bead_gate_writes_entries_and_returns_copy(beads):
    x, y, label = beads
    a = bead_adata(x, y, extra={"FL1-A": np.ones_like(x)})
    out = gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"])
    keep = a.obs["keepmask_scatter"].to_numpy()
    assert keep.dtype == bool and a.uns["Nbeads_afterscattergate"] == keep.sum()
    hull = a.uns["density_gate_hull"]
    assert hull.shape[1] == 2 and np.array_equal(hull[0], hull[-1])
    assert out is not a and out.shape == (keep.sum(), 3) and out.obs["keepmask_scatter"].all()
    assert out.uns["Nbeads_afterscattergate"] == keep.sum()
    assert (label[keep] == 0).mean() > .99


def test_bead_gate_apply_false_returns_same_object(beads):
    x, y, _ = beads
    a = bead_adata(x, y)
    assert gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"], apply=False) is a
    assert "keepmask_scatter" in a.obs


def test_bead_gate_matches_autospill_with_documented_defaults(beads):
    x, y, _ = beads
    a = bead_adata(x, y)
    gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"], apply=False)
    asp = AutoSpillTesselationGating(x, y, trim_beadmode=True, final_density_cutoff=.2, bound_density_mad_factor=4.)
    assert np.array_equal(a.obs["keepmask_scatter"].to_numpy(), asp.get_gatemask(x, y))
    assert np.array_equal(a.uns["density_gate_hull"], asp.densitygate.vertices_hull)


def test_bead_gate_default_channels_from_cytoconfig(beads, config):
    x, y, _ = beads
    a = bead_adata(x, y)
    a.uns["cytoconfig"] = config
    seen = {}

    class Fake:
        def __init__(self, *X, **kw):
            seen["X"] = X; seen.update(kw)
            self.densitygate = type("D", (), {"vertices_hull": np.zeros((4, 2))})()

        def get_gatemask(self, *X):
            return np.ones(len(X[0]), bool)

    with mock.patch("cytovanni.gating.bead.AutoSpillTesselationGating", Fake):
        gate_scatter_bead_sample(a)
    assert seen["name_x"] == "FSC-A" and seen["name_y"] == "SSC-A" and np.array_equal(seen["X"][0], x)


@pytest.mark.parametrize("channels", [["FSC-A", "FL9-A"], ["FSC-A"], ["FSC-A", "SSC-A", "FSC-A"]])
def test_bead_gate_invalid_channels_raise(beads, channels):
    x, y, _ = beads
    with pytest.raises(ScatterGatingChannelException):
        gate_scatter_bead_sample(bead_adata(x[:100], y[:100]), gatechannels=channels)


def _capture_kwargs(**call):
    seen = {}

    class Fake:
        def __init__(self, *X, **kw):
            seen.update(kw)
            self.densitygate = type("D", (), {"vertices_hull": np.zeros((4, 2))})()

        def get_gatemask(self, *X):
            return np.ones(len(X[0]), bool)

    a = bead_adata(np.random.rand(50), np.random.rand(50))
    with mock.patch("cytovanni.gating.bead.AutoSpillTesselationGating", Fake):
        gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"], **call)
    return seen


def test_bead_gate_parameter_sets():
    """The defaults stated in the Notes for normal, robust and very_robust (very_robust only with robust)."""
    base = dict(final_density_cutoff=.2, verbose=False, name_x="FSC-A", name_y="SSC-A", trim_beadmode=True)
    assert _capture_kwargs() == {**base, "bound_density_mad_factor": 4.}
    assert _capture_kwargs(very_robust=True) == {**base, "bound_density_mad_factor": 4.}
    assert _capture_kwargs(robust=True) == {**base, "bound_density_mad_factor": 5., "trim_beadmode_min_clipfactor": .25,
                                            "trim_beadmode_max_clipfactor": 2., "region_density_bw_factor": 3}
    assert _capture_kwargs(robust=True, very_robust=True) == {
        **base, "bound_density_mad_factor": 10., "trim_beadmode_min_clipfactor": .25, "trim_beadmode_max_clipfactor": 2.,
        "region_density_bw_factor": 7, "bound_density_bw_factor": 6.}


def test_bead_gate_kwargs_override_defaults():
    """R2-K3: gating_kwargs naming a setting the function sets itself override it (no "multiple values" TypeError)."""
    seen = _capture_kwargs(robust=True, final_density_cutoff=.3, main_tile="density", trim_beadmode=False)
    assert seen["final_density_cutoff"] == .3 and seen["bound_density_mad_factor"] == 5.
    assert seen["main_tile"] == "density" and seen["trim_beadmode"] is False and seen["name_x"] == "FSC-A"


def test_bead_gate_plotpath_and_title(beads, tmp_path):
    x, y, _ = beads
    a = bead_adata(x[:5000], y[:5000])
    p = str(tmp_path / "gate.png")
    gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"], plotpath=p)
    assert os.path.getsize(p) > 0
    assert plt.get_fignums() == []  # closed after saving
    with mock.patch.object(AutoSpillTesselationGating, "plot", autospec=True) as pl, mock.patch.object(plt, "show"):
        gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"], plot=True, titlefct=lambda _: "fixed")
    assert pl.call_args.kwargs["title"] == "fixed"


def test_bead_gate_on_rainbow_fcs(rainbow_files, config):
    """C: on the simulated rainbow file the gate keeps the bead cloud and removes most of the 10 % debris."""
    from cytovanni.io import readfcs_rainbow_sample
    a = readfcs_rainbow_sample(rainbow_files["A"][0], config, len(synthetic.PEAKS), name="rb")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = gate_scatter_bead_sample(a)
    raw = a.to_df(layer="raw")
    debris = (raw["FSC-A"] < 6000).to_numpy()  # debris: FSC uniform 500-4000, beads ~1e4
    keep = a.obs["keepmask_scatter"].to_numpy()
    assert keep[debris].mean() < .02 and keep[~debris].mean() > .5
    assert out.n_obs == keep.sum() and out.uns["name_uid"] == a.uns["name_uid"]


@pytest.mark.parametrize("kw", [dict(robust=True), dict(robust=True, very_robust=True), dict(main_tile="density"),
                                dict(merge_tiles_min_saddle=.6)])
def test_bead_gate_variants_still_exclude_doublets(beads, kw):
    """R1-other: the robust settings, `main_tile` and `merge_tiles_min_saddle` (via gating_kwargs) give valid gates."""
    x, y, label = beads
    a = bead_adata(x, y)
    gate_scatter_bead_sample(a, gatechannels=["FSC-A", "SSC-A"], apply=False, **kw)
    keep = a.obs["keepmask_scatter"].to_numpy()
    assert keep[label == 0].mean() > .5 and (label[keep] == 0).mean() > .97
