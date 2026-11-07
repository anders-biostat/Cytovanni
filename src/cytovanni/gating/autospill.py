"""Automatic scatter gating of bead samples on singlets.

A Python reimplementation, with small modifications, of the gating strategy of AutoSpill (Roca et al. 2021,
https://github.com/carlosproca/autospill/blob/master/R/do_gate.r): two rounds of KDE peak finding and Voronoi
tessellation on a 2-D scatter plot (usually FSC-A vs SSC-A, raw scale) select the main bead population, and a convex
hull around its densest events is the final gate. Use it through `gate_scatter_bead_sample`.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from scipy.spatial import Voronoi, ConvexHull, QhullError
from matplotlib.collections import LineCollection
import flowutils
from sklearn.neighbors import KernelDensity
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

def _scotts_factor(X):
    """Scott's bandwidth factor ``n**(-1/(d+4))`` for data `X` of shape (events, d)."""
    return X.shape[0]**(-1./(X.shape[1]+4))

def _clip_to_percentiles(x, minperc, maxperc):
    """Clip values to the `minperc` and `maxperc` percentiles (0-100) of `x` (flattened)."""
    return np.clip(x, a_min=np.percentile(x, minperc), a_max=np.percentile(x, maxperc))

def _clip_to_medianfactor(x, fmin=.5, fmax=1.5):
    """Clip values to [`fmin` * median, `fmax` * median] of `x`."""
    return np.clip(x, a_min=fmin*np.median(x), a_max=fmax*np.median(x))

def _mad(data, axis=None):
    """Mean absolute deviation from the mean (not the median absolute deviation)."""
    return np.mean(np.abs(data - np.mean(data, axis)), axis)

def _plot_subsample(n, max_points):
    """Sorted indices of at most `max_points` of `n` events for plotting (fixed seed, so plots are reproducible)."""
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, max_points, replace=False))

def _style_axes(ax, name_x, name_y):
    """Set axis labels and tick label sizes of `ax`."""
    ax.set_xlabel(name_x, size=16)
    ax.set_ylabel(name_y, size=16)
    ax.tick_params(labelsize=12)

class _BinnedKDE(gaussian_kde):
    """2-D `scipy.stats.gaussian_kde`, evaluated approximately by linear binning, FFT convolution and interpolation.

    The data are binned linearly on an ``N_GRID`` x ``N_GRID`` grid (padded by ``PAD_SD`` kernel standard deviations),
    convolved with the full-covariance Gaussian kernel (truncated at ``PAD_SD`` standard deviations) and linearly
    interpolated at the evaluation points; negative interpolation artefacts are clipped to 0. Bandwidth, covariance and
    weights are exactly those of `gaussian_kde`, only the evaluation is approximate (cost linear in the number of
    events instead of events x points).
    """
    N_GRID = 512 # 256 can erase maxima the exact KDE finds; 512 matched exact gates in tests (density error ~3e-4 of max)
    PAD_SD = 4.

    def evaluate(self, points):
        """Density at `points` (array of shape (2, points)), approximated on the grid; 0 far from the data."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        sd = np.sqrt(np.diag(self.covariance))
        lo = np.minimum(self.dataset.min(1), points.min(1)) - self.PAD_SD * sd
        hi = np.maximum(self.dataset.max(1), points.max(1)) + self.PAD_SD * sd
        step = (hi - lo) / (self.N_GRID - 1)
        # linear binning of the (weighted) data
        u = (self.dataset - lo[:, None]) / step[:, None]
        i0 = np.clip(np.floor(u).astype(int), 0, self.N_GRID - 2)
        f = u - i0
        grid = np.zeros((self.N_GRID, self.N_GRID))
        for dx in (0, 1):
            for dy in (0, 1):
                w = self.weights * (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1])
                np.add.at(grid, (i0[0] + dx, i0[1] + dy), w)
        # Gaussian kernel image (full covariance), truncated at PAD_SD sd
        kx = np.arange(-int(np.ceil(self.PAD_SD * sd[0] / step[0])), int(np.ceil(self.PAD_SD * sd[0] / step[0])) + 1) * step[0]
        ky = np.arange(-int(np.ceil(self.PAD_SD * sd[1] / step[1])), int(np.ceil(self.PAD_SD * sd[1] / step[1])) + 1) * step[1]
        D = np.stack(np.meshgrid(kx, ky, indexing="ij"), -1)
        kern = np.exp(-.5 * np.einsum("...i,ij,...j->...", D, self.inv_cov, D)) / (2 * np.pi * np.sqrt(np.linalg.det(self.covariance)))
        dens = fftconvolve(grid, kern, mode="same")
        return np.clip(map_coordinates(dens, (points - lo[:, None]) / step[:, None], order=1, mode="nearest"), 0, None)
    __call__ = evaluate

class ScaledVoronoi():
    """Voronoi tessellation computed in grid index space, with coordinates mapped back to data values.

    The tessellation is done on grid indices so that both axes count equally, whatever the scale of the two channels;
    `points` and `vertices` hold the mapped (data-scale) coordinates, `points_index` the index-space sites.

    Parameters
    ----------
    voronoi : scipy.spatial.Voronoi
        Tessellation of the sites in grid index space.
    ind_to_point : callable
        Maps grid index coordinates (array of shape (n, 2)) to data values.
    point_to_ind : callable
        Inverse of `ind_to_point`: data values (array of shape (n, 2)) to grid index coordinates.
    """
    def __init__(self, voronoi, ind_to_point, point_to_ind):
        self.points = ind_to_point(voronoi.points)
        self.points_index = voronoi.points
        self.vertices = ind_to_point(voronoi.vertices)
        self.ridge_points = voronoi.ridge_points
        self.ridge_vertices = voronoi.ridge_vertices
        self.furthest_site = voronoi.furthest_site
        
        self.voronoi = voronoi
        self.ind_to_point = ind_to_point
        self.point_to_ind = point_to_ind
    
    def assign_group(self, X):
        """Assign points to Voronoi tiles.

        Parameters
        ----------
        X : numpy.ndarray
            Points of shape (events, 2) in data values.

        Returns
        -------
        numpy.ndarray
            Index of the nearest site (in grid index space), i.e. the tile, for every point; shape (events,).
        """
        group = KDTree(self.points_index).query(self.point_to_ind(X))[1]
        return group

    def plot(self, ax=None, show_points=True, show_vertices=False,
             line_colors="k", line_width=1., line_alpha=1., ray_length=100.):
        """Draw the tile borders in data coordinates (replacement for `scipy.spatial.voronoi_plot_2d`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Default: a new 6 x 6 figure.
        show_points : bool, default True
            Mark the Voronoi sites.
        show_vertices : bool, default False
            Mark the Voronoi vertices.
        line_colors : color, default "k"
            Colour of the tile borders.
        line_width : float, default 1.0
            Line width of the tile borders.
        line_alpha : float, default 1.0
            Opacity of the tile borders.
        ray_length : float, default 100.0
            Length of the infinite ridges (dashed), in multiples of the largest extent of the sites (index space).

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into.

        Notes
        -----
        The ridge geometry is built in index space, where the perpendicular bisector construction is valid, and only
        then mapped to data values. Passing the scaled object to `voronoi_plot_2d` would be wrong: it derives the ray
        directions by rotating the ridge tangent by 90 degrees, which the anisotropic scaling does not preserve.
        """
        vor, f = self.voronoi, self.ind_to_point
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(6, 6))

        if show_points:
            ax.plot(*f(vor.points).T, ".")
        if show_vertices:
            ax.plot(*f(vor.vertices).T, "o")

        center = vor.points.mean(axis=0)
        ptp = np.ptp(vor.points, axis=0).max()

        finite, infinite = [], []
        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                finite.append(f(vor.vertices[simplex]))
            else:
                i = simplex[simplex >= 0][0]
                t = vor.points[pointidx[1]] - vor.points[pointidx[0]]
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]])
                midpoint = vor.points[pointidx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if vor.furthest_site:
                    direction = -direction
                far = vor.vertices[i] + direction * ptp * ray_length
                infinite.append(f(np.vstack([vor.vertices[i], far])))

        ax.add_collection(LineCollection(finite, colors=line_colors,
                                         lw=line_width, alpha=line_alpha))
        ax.add_collection(LineCollection(infinite, colors=line_colors,
                                         lw=line_width, alpha=line_alpha,
                                         linestyle="dashed"))
        return ax

class HistogramTesselator():
    """KDE density on a grid, local maxima and Voronoi tessellation of a 2-D scatter plot (one step of the gating).

    The density of the events is estimated with a Gaussian KDE (bandwidth `bw_factor` x Scott's factor) on the centres
    of an `Nx` x `Ny` grid spanning `bounds_x` x `bounds_y`. Local maxima of the gridded density are the sites of a
    Voronoi tessellation in grid index space; every event is assigned to the tile of its nearest maximum, and one tile
    (or several, see `merge_tiles_min_saddle`) is marked as the main tile.

    Parameters
    ----------
    x, y : numpy.ndarray
        1-D arrays of the event values of the two channels (same scale as the bounds, raw in
        `AutoSpillTesselationGating`).
    bounds_x, bounds_y : tuple of float
        (min, max) of the grid along each axis.
    Nx, Ny : int, default 100
        Number of grid cells along x and y.
    bw_factor : float, default 3.0
        KDE bandwidth as a multiple of Scott's factor.
    neigh_size : int, default 3
        A grid point is a local maximum if its density equals the maximum within ``2 * neigh_size + 1`` grid points
        around it in both directions (truncated at the grid edge). Maxima below ``MIN_REL_MAXIMUM`` x the highest
        density are ignored.
    density_exclusion_corner : float, default 0.05
        Maxima whose grid index is within this fraction of the grid from the lower end in x or in y are discarded
        (low-scatter debris).
    name_x, name_y : str, default "FSC-A", "SSC-A"
        Axis labels for plots.
    Nsubsample : int, optional
        Keep a random subset of this many events for the tile assignment (`x`, `y` attributes); the histogram and the
        KDE still use all events. Default: all events.
    main_tile : {'count', 'density'}, default 'count'
        Main tile: the tile with the most events ('count', as in AutoSpill) or the tile of the highest density maximum
        ('density').
    merge_tiles_min_saddle : float, optional
        Also mark as main the tiles whose maximum is connected to the main maximum by a density ridge: the minimum
        density on the straight line (index space) between both maxima is at least this fraction of the density at the
        main maximum. Default: only the main tile.

    Attributes
    ----------
    density, logdensity : numpy.ndarray
        KDE density on the grid, shape (Nx, Ny); its log, clipped below at ``MIN_LOGDENSITY``.
    maxima, maxima_indices : numpy.ndarray
        Local maxima in data values and in grid indices, shape (maxima, 2). `maxima_indices` can contain up to three
        additional far-away dummy sites, added because the tessellation fails for two or for collinear sites.
    voronoi : ScaledVoronoi or None
        The tessellation, None if there is only one maximum.
    voronoi_label : numpy.ndarray
        Tile of every event, shape (events,).
    voronoi_label_ismain : numpy.ndarray
        Boolean mask of shape (events,): event lies in a main tile.
    saddle_ratio : dict of int to float
        For every maximum except the main one, the ridge ratio described under `merge_tiles_min_saddle`.

    Raises
    ------
    RuntimeError
        If no local maximum is found, or none is left after the exclusion of the lower edge.
    ValueError
        If `main_tile` is not 'count' or 'density'.
    """
    USE_APPROX_KDE_TREE = False # doesn't currently work!
    MIN_LOGDENSITY = -50
    MAXIMUM_TOLERANCE = 1e-7
    MIN_REL_MAXIMUM = 1e-6 # ignore maxima below this fraction of the peak density (flat empty regions)
    PLOT_LOGDENSITY_RANGE = 12. # log density colour range in plots
    
    def __init__(self, x, y, bounds_x, bounds_y, Nx=100, Ny=100, bw_factor=3., neigh_size=3, density_exclusion_corner=.05, name_x="FSC-A", name_y="SSC-A", Nsubsample=None,
                 main_tile="count", merge_tiles_min_saddle=None):
        self.x, self.y = x, y
        if Nsubsample is not None:
            randind = np.random.permutation(len(x))[:Nsubsample]
            self.x, self.y = self.x[randind], self.y[randind]
        self.bounds_x, self.bounds_y = bounds_x, bounds_y
        self.name_x, self.name_y = name_x, name_y
        
        # First simple histogram
        bins = [np.linspace(*bounds_x, Nx+1), np.linspace(*bounds_y, Ny+1)]
        self.histogram, self.bins_x, self.bins_y = np.histogram2d( x, y, bins=bins )
        self.bins_center_x = (self.bins_x[1:] + self.bins_x[:-1])/2
        self.bins_center_y = (self.bins_y[1:] + self.bins_y[:-1])/2
        self.bins_center = np.vstack([self.bins_center_x, self.bins_center_y]).T
        self.ind_to_point = lambda x: self.bins_center[[0]] + x/np.asarray([[Nx-1, Ny-1]]) * (self.bins_center[[-1]] - self.bins_center[[0]])
        self.point_to_ind = lambda X: (X - self.bins_center[[0]]) / (self.bins_center[[-1]] - self.bins_center[[0]]) * np.asarray([[Nx-1, Ny-1]])
        
        # kde density estimate
        if self.USE_APPROX_KDE_TREE:
            X = np.vstack([x, y]).T
            self.density_kde = KernelDensity(bandwidth=bw_factor * _scotts_factor(X), atol=1e-6, leaf_size=200, algorithm="ball_tree").fit(X)
            ind_X, ind_Y = np.mgrid[0:Nx, 0:Ny]
            X = np.vstack([self.bins_center_x[ind_X].flatten(), self.bins_center_y[ind_Y].flatten()]).T
            self.logdensity = np.clip(self.density_kde.score_samples(X).reshape((Nx, Ny)), a_min=self.MIN_LOGDENSITY, a_max=None)
            self.density = np.exp(self.logdensity)
        else:
            self.density_kde = _BinnedKDE(np.vstack([x, y]), lambda kde: bw_factor * gaussian_kde.scotts_factor(kde))
            ind_X, ind_Y = np.mgrid[0:Nx, 0:Ny]
            self.density = self.density_kde(np.vstack([self.bins_center_x[ind_X].flatten(), self.bins_center_y[ind_Y].flatten()])).reshape((Nx, Ny))
            # the binned KDE is exactly 0 far from the data (clipped at 0): log(0) = -inf is clipped right away
            with np.errstate(divide="ignore"):
                self.logdensity = np.clip(np.log(self.density), a_min=self.MIN_LOGDENSITY, a_max=None)
            #self.density = np.exp(self.density)
        # max density within neighbourhood size
        ind_ng_X, ind_ng_Y = np.mgrid[-neigh_size:neigh_size+1, -neigh_size:neigh_size+1]
        #mask = ~((ind_ng_X==0) | (ind_ng_Y==0))
        #ind_ng_X, ind_ng_Y = ind_ng_X[mask], ind_ng_Y[mask]
        density_max_neighbors = self.density[np.clip((ind_X.flatten()[:,None] + ind_ng_X.flatten()[None]), 0, Nx-1),
                                             np.clip((ind_Y.flatten()[:,None] + ind_ng_Y.flatten()[None]), 0, Ny-1)
                                            ].max(1).reshape(self.density.shape)
        # find maxima positions
        self.maxima_indices = np.argwhere((self.density==density_max_neighbors) & (self.density > self.MIN_REL_MAXIMUM * self.density.max()))
        if self.maxima_indices.shape[0]==0:
            raise RuntimeError("AutoSpillTesselationGating, Histogram: Could not find any maxima after kde smoothing!")
        
        # exclude maxima in lower left corner, usually debris
        self.maxima_indices = self.maxima_indices[~np.any(self.maxima_indices <= np.asarray([[Nx-1, Ny-1]]) * density_exclusion_corner, axis=1)]
        if self.maxima_indices.shape[0]==0:
            raise RuntimeError("AutoSpillTesselationGating, Histogram: Could not find any maxima after excluding lower left corner!")
        self.maxima = self.ind_to_point(self.maxima_indices)
        #self.maxima = np.vstack([self.bins_center_x[self.maxima_indices[:,0]], self.bins_center_y[self.maxima_indices[:,1]]]).T
        
        if self.maxima_indices.shape[0]==1:
            self.voronoi = None
            self.voronoi_label = np.zeros_like(self.x)
            self.voronoi_label_ismain = np.full_like(self.x, True, dtype=bool)
            self.saddle_ratio = {}
        else:
            if self.maxima_indices.shape[0]==2:
                # can't deal with only two points for some reason, add dummy far away
                self.maxima_indices = np.vstack([self.maxima_indices,[[-Nx,-Ny],[2*Nx,-Ny],[-Nx,2*Ny]]])
            try:
                voronoi = Voronoi(self.maxima_indices)
            except QhullError:
                # If points are coplanar qhull fails, try adding dummies far away
                self.maxima_indices = np.vstack([self.maxima_indices,[[-Nx, -Ny], [-Nx, -Ny-1], [-Nx-1, -Ny-1]]])
                voronoi = Voronoi(self.maxima_indices)
            self.voronoi = ScaledVoronoi(voronoi, self.ind_to_point, self.point_to_ind)
            self.voronoi_label = self.voronoi.assign_group(np.vstack([self.x,self.y]).T)
            # main tile: most events ("count", original behaviour) or highest density maximum ("density");
            # optionally merge tiles connected to the main maximum by a density ridge (see _saddle_ratio)
            maxima_density = self.density[tuple(self.maxima_indices[:len(self.maxima)].T)]
            if main_tile=="count":
                u, c = np.unique(self.voronoi_label, return_counts=True)
                main = [u[np.argmax(c)]]
            elif main_tile=="density":
                main = [np.argmax(maxima_density)]
            else:
                raise ValueError(f"main_tile must be 'count' or 'density', got {main_tile}")
            self.saddle_ratio = {j: self._saddle_ratio(main[0], j) for j in range(len(self.maxima)) if j != main[0]}
            if merge_tiles_min_saddle is not None:
                main = main + [j for j, r in self.saddle_ratio.items() if r >= merge_tiles_min_saddle]
            self.voronoi_label_ismain = np.isin(self.voronoi_label, main)
    
    def _saddle_ratio(self, i, j, n=100):
        """Minimum density on the straight line (grid index space) between maxima i (main) and j, relative to maximum i.

        High: j sits on a ridge connected to the main peak (e.g. sub-bands of one bead population); low: separated by a
        valley, or j is a small bump far from the main peak.
        """
        a, b = self.maxima_indices[i], self.maxima_indices[j]
        path = np.rint(a[None] + np.linspace(0, 1, n)[:,None] * (b - a)[None]).astype(int)
        return self.density[path[:,0], path[:,1]].min() / self.density[tuple(a)]
    
    def plot(self, ax=None, xlim=None, ylim=None, max_points=20000):
        """Plot the log density, the events coloured by tile (main tiles black), the maxima and the tile borders.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Default: a new 6 x 6 figure.
        xlim, ylim : tuple of float, optional
            Axis limits. Default: the grid bounds.
        max_points : int or None, default 20000
            At most this many events are drawn (fixed random subset); None draws all.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into.
        """
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(6,6))

        # colour range capped below the maximum: the binned KDE is exactly 0 far from all data (log clipped to MIN_LOGDENSITY)
        ax.imshow(self.logdensity[:,::-1].T, extent=(*self.bounds_x, *self.bounds_y), aspect="auto",
                  vmin=max(self.logdensity.min(), self.logdensity.max() - self.PLOT_LOGDENSITY_RANGE))
        ind = _plot_subsample(len(self.x), max_points)
        colors = plt.get_cmap("tab10")(np.asarray(self.voronoi_label)[ind].astype(int) % 10)
        colors[self.voronoi_label_ismain[ind]] = (0, 0, 0, 1)
        ax.scatter(self.x[ind], self.y[ind], s=1, c=colors, rasterized=True)
        ax.scatter(*self.maxima.T, color="white", edgecolor="black", s=40, zorder=3)
        if self.voronoi is not None:
            self.voronoi.plot(ax=ax, show_points=False, show_vertices=False)
        ax.set_xlim(self.bounds_x if xlim is None else xlim)
        ax.set_ylim(self.bounds_y if ylim is None else ylim)
        _style_axes(ax, self.name_x, self.name_y)

        return ax
    
    def rectangle_from_main(self, mad_factor):
        """Rectangle around the events of the main tiles: median +- `mad_factor` x mean absolute deviation per axis.

        Parameters
        ----------
        mad_factor : float
            Half-width of the rectangle in units of the mean absolute deviation (from the mean) of each axis.

        Returns
        -------
        xlim, ylim : tuple of float
            (min, max) of the rectangle along x and along y, in data values.
        """
        X = np.vstack([self.x,self.y]).T[self.voronoi_label_ismain]
        center = np.median(X, axis=0)
        mad = _mad(X, axis=0)
        xlim = (center[0] - mad_factor*mad[0], center[0] + mad_factor*mad[0])
        ylim = (center[1] - mad_factor*mad[1], center[1] + mad_factor*mad[1])
        return xlim, ylim

class DensityGate():
    """Final gate: convex hull of the densest events of a 2-D population.

    The events are ranked by their KDE density (bandwidth `factor_kde` x Scott's factor), the ranks scaled to [0, 1];
    the gate is the convex hull of the events with scaled rank above `cutoff`, i.e. of the densest ``1 - cutoff``
    fraction of the events.

    Parameters
    ----------
    X : numpy.ndarray
        Events of shape (events, 2), data values.
    cutoff : float, default 0.33
        Fraction of the least dense events left out of the hull.
    factor_kde : float, default 1.0
        KDE bandwidth as a multiple of Scott's factor.
    Nsubsample : int, optional
        Use a random subset of this many events. Default: all events.

    Attributes
    ----------
    density_scaled : numpy.ndarray
        Density rank of every event scaled to [0, 1], shape (events,).
    vertices_hull : numpy.ndarray
        Vertices of the convex hull, shape (vertices + 1, 2), the first vertex repeated at the end.
    """
    USE_APPROX_KDE_TREE=False # not implemented yet
    
    def __init__(self, X, cutoff=.33, factor_kde=1., Nsubsample=None):
        self.X = X
        if Nsubsample is not None:
            randind = np.random.permutation(self.X.shape[0])[:Nsubsample]
            self.X = self.X[randind]
        density_kde = _BinnedKDE(self.X.T, lambda kde: factor_kde * gaussian_kde.scotts_factor(kde))
        density = density_kde(self.X.T)
        self.density_scaled = np.argsort(np.argsort(density))/(len(density)-1)
        
        X_ = self.X[self.density_scaled>cutoff]
        self.convex_hull = ConvexHull(X_)
        vertices_hull = X_[self.convex_hull.vertices]
        self.vertices_hull = np.vstack([vertices_hull, vertices_hull[[0]]])
    
    def hull_gatemask(self, X):
        """Boolean mask of shape (events,): which of the points `X` (shape (events, 2)) lie inside the hull."""
        return flowutils.gating.points_in_polygon(self.vertices_hull, X)
    
    def plot(self, ax=None, xlim=None, ylim=None, max_points=20000):
        """Plot the events coloured by scaled density rank, and the hull.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Default: a new 6 x 6 figure.
        xlim, ylim : tuple of float, optional
            Axis limits. Default: matplotlib's automatic limits.
        max_points : int or None, default 20000
            At most this many events are drawn (fixed random subset); None draws all.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into.
        """
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(6,6))

        ind = _plot_subsample(self.X.shape[0], max_points)
        ax.scatter(self.X[ind,0], self.X[ind,1], c=self.density_scaled[ind], s=5, cmap="viridis", edgecolor="w", linewidth=.3, rasterized=True)
        ax.plot(*self.vertices_hull.T, color="black")
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        
        return ax

class AutoSpillTesselationGating():
    """Automatic singlet gate for bead samples, a reimplementation of the tessellation gating of AutoSpill.

    Ported, with small modifications, from https://github.com/carlosproca/autospill/blob/master/R/do_gate.r. The gate
    is computed when the object is created, in four steps:

    1. Trimming: each axis is clipped to the percentiles `trim_x_min`/`trim_x_max` (`trim_y_*`), or, with
       `trim_beadmode`, to [`trim_beadmode_min_clipfactor`, `trim_beadmode_max_clipfactor`] x its median. The clip
       range gives the bounds; events at the bounds (the clipped ones) are left out of the following steps.
    2. Bound tessellation (`histogram_bound`, a `HistogramTesselator` over the bounds, ``bound_density_*``
       parameters). The main tile defines a rectangle, median +- `bound_density_mad_factor` x mean absolute deviation
       of its events on each axis (`region_x`, `region_y`).
    3. Region tessellation (`histogram_region`, ``region_density_*`` parameters) of the events inside the rectangle;
       its main tile(s) are the candidate singlets (`x_final`, `y_final`).
    4. Density gate (`densitygate`, a `DensityGate`): the convex hull of the densest ``1 - final_density_cutoff``
       fraction of the candidate singlets is the final gate, see `get_gatemask`.

    All tessellations are done in the index space of the grid, not in data values, so that both axes count equally
    even though FSC and SSC values of beads can differ by orders of magnitude.

    Parameters
    ----------
    x, y : numpy.ndarray
        1-D arrays of the event values of the two scatter channels (raw scale), e.g. FSC-A and SSC-A.
    verbose : bool, default False
        Show the summary plot (`plot`) after the fit.
    **kwargs
        Override any of the class attributes below for this instance (set with ``setattr`` before the fit; names are
        not checked, so a misspelled name has no effect).

    Other Parameters
    ----------------
    trim_x_min, trim_x_max, trim_y_min, trim_y_max : float, default 1, 99
        Percentiles (0-100) the axes are clipped to, if not `trim_beadmode`.
    trim_beadmode : bool, default False
        Clip to multiples of the median instead of to percentiles (`gate_scatter_bead_sample` sets True).
    trim_beadmode_min_clipfactor, trim_beadmode_max_clipfactor : float, default 0.5, 1.5
        Clip range as factors of the median, if `trim_beadmode`.
    name_x, name_y : str, default "FSC-A", "SSC-A"
        Axis labels for plots.
    subsample_kde_estimates : int, optional
        `Nsubsample` of the tessellations and the density gate. Default None: all events.
    bound_density_bw_factor : float, default 3.0
        KDE bandwidth factor of the bound tessellation (see `HistogramTesselator`, `bw_factor`).
    bound_density_grid_N : int, default 100
        Grid size (per axis) of the bound tessellation.
    bound_density_neigh_size : int, default 3
        Neighbourhood size for local maxima of the bound tessellation.
    bound_density_exclusion_corner : float, default 0.05
        Lower-edge exclusion of maxima in the bound tessellation (see `density_exclusion_corner`).
    bound_density_mad_factor : float, default 3.0
        Half-width of the region rectangle in mean absolute deviations of the main tile.
    region_density_bw_factor : float, default 2.0
        KDE bandwidth factor of the region tessellation.
    region_density_grid_N : int, default 100
        Grid size (per axis) of the region tessellation.
    region_density_neigh_size : int, default 2
        Neighbourhood size for local maxima of the region tessellation.
    region_density_exclusion_corner : float, default 0.0
        Lower-edge exclusion of maxima in the region tessellation.
    final_density_bw_factor : float, default 1.0
        KDE bandwidth factor of the density gate.
    final_density_cutoff : float, default 0.33
        Fraction of the least dense candidate singlets left out of the hull.
    main_tile : {'count', 'density'}, default 'count'
        How the main tile is chosen in both tessellations (see `HistogramTesselator`).
    merge_tiles_min_saddle : float, optional
        Region tessellation only: also keep tiles connected to the main maximum by a density ridge of at least this
        fraction of the main maximum (see `HistogramTesselator`). Default None: only the main tile.
    plot_view_percentiles : tuple of float, default (5, 95)
        The plot view covers these percentiles of all events, the clip bounds and the region.
    plot_view_margin : float, default 0.1
        Margin around the plot view, as a fraction of its width.
    plot_max_points : int, default 10000
        Maximum number of events drawn per scatter plot.

    Raises
    ------
    RuntimeError
        If a tessellation finds no density maximum (from `HistogramTesselator`).

    See Also
    --------
    gate_scatter_bead_sample : applies this gate to a bead AnnData, with tuned defaults.

    Examples
    --------
    >>> asp = AutoSpillTesselationGating(fsc_a, ssc_a, trim_beadmode=True, final_density_cutoff=.2)
    >>> keep = asp.get_gatemask(fsc_a, ssc_a)
    """
    
    # General trimming percentiles
    trim_x_min = 1
    trim_x_max = 99
    trim_y_min = 1
    trim_y_max = 99
    
    trim_beadmode = False
    trim_beadmode_min_clipfactor = .5
    trim_beadmode_max_clipfactor = 1.5
    
    name_x = "FSC-A"
    name_y = "SSC-A"
    
    subsample_kde_estimates = None # no subsampling needed with the binned KDE (cost O(n)), keeps the gate deterministic
    
    # Parameters for full tesselation
    bound_density_bw_factor = 3. # 3.
    bound_density_grid_N = 100
    bound_density_neigh_size = 3 # 3
    bound_density_exclusion_corner = 0.05
    bound_density_mad_factor = 3. # 3.
    
    # Parameters for limited tesselation
    region_density_bw_factor = 2.
    region_density_grid_N = 100
    region_density_neigh_size = 2
    region_density_exclusion_corner = 0.
    
    # Parameters for final gate
    final_density_bw_factor = 1.
    final_density_cutoff = .33 # .33
    
    # Main tile choice: "count" (most events, original) or "density" (highest density maximum), in both tesselations;
    # merge_tiles_min_saddle (region tesselation only): also keep tiles whose maximum is connected to the main maximum by a
    # density ridge, i.e. min density on the line between both maxima >= merge_tiles_min_saddle * the main maximum
    main_tile = "count"
    merge_tiles_min_saddle = None

    # Plotting: view = union of these percentiles of all events, clip bounds and region (+ margin); max points drawn per scatter
    plot_view_percentiles = (5, 95)
    plot_view_margin = .1
    plot_max_points = 10000
    
    def __init__(self, x, y, verbose=False, **kwargs):
        self.verbose = verbose
        self.x_full = x
        self.y_full = y
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # Clip data to percentiles
        if self.trim_beadmode:
            self.x = _clip_to_medianfactor(self.x_full, self.trim_beadmode_min_clipfactor, self.trim_beadmode_max_clipfactor)
            self.y = _clip_to_medianfactor(self.y_full, self.trim_beadmode_min_clipfactor, self.trim_beadmode_max_clipfactor)
        else:
            self.x = _clip_to_percentiles(self.x_full, self.trim_x_min, self.trim_x_max)
            self.y = _clip_to_percentiles(self.y_full, self.trim_y_min, self.trim_y_max)
        
        # drop data that is out of bounds
        self.bounds_x = np.asarray([self.x.min(), self.x.max()])
        self.bounds_y = np.asarray([self.y.min(), self.y.max()])
        mask = (self.x!=self.x.min()) & (self.x!=self.x.max()) & (self.y!=self.y.min()) & (self.y!=self.y.max())
        self.x, self.y = self.x[mask], self.y[mask]
        
        # get initial histogram
        self.histogram_bound = HistogramTesselator(self.x, self.y, self.bounds_x, self.bounds_y, self.bound_density_grid_N, self.bound_density_grid_N,
                     bw_factor=self.bound_density_bw_factor, neigh_size=self.bound_density_neigh_size, density_exclusion_corner=self.bound_density_exclusion_corner,
                     name_x=self.name_x, name_y=self.name_y, Nsubsample=self.subsample_kde_estimates, main_tile=self.main_tile)
        # get region for second step
        self.region_x, self.region_y = self.histogram_bound.rectangle_from_main(self.bound_density_mad_factor)
        
        # get region histogram
        mask = (self.x>self.region_x[0]) & (self.x<self.region_x[1]) & (self.y>self.region_y[0]) & (self.y<self.region_y[1])
        self.histogram_region = HistogramTesselator(self.x[mask], self.y[mask], self.region_x, self.region_y, self.region_density_grid_N, self.region_density_grid_N,
                     bw_factor=self.region_density_bw_factor, neigh_size=self.region_density_neigh_size, density_exclusion_corner=self.region_density_exclusion_corner,
                     name_x=self.name_x, name_y=self.name_y, Nsubsample=self.subsample_kde_estimates, main_tile=self.main_tile,
                     merge_tiles_min_saddle=self.merge_tiles_min_saddle)
            
        # data in final tile
        self.x_final = self.histogram_region.x[self.histogram_region.voronoi_label_ismain]
        self.y_final = self.histogram_region.y[self.histogram_region.voronoi_label_ismain]
        
        # get final density gate
        self.densitygate = DensityGate(np.vstack([self.x_final, self.y_final]).T, cutoff=self.final_density_cutoff, factor_kde=self.final_density_bw_factor, Nsubsample=self.subsample_kde_estimates)
        
        if self.verbose:
            self.plot()
    
    def _plot_view(self):
        """Plot limits: union of the `plot_view_percentiles` range of all events, bounds and region, plus margin."""
        lims = []
        for full, bounds, region in [(self.x_full, self.bounds_x, self.region_x), (self.y_full, self.bounds_y, self.region_y)]:
            lo = min(np.percentile(full, self.plot_view_percentiles[0]), bounds[0], region[0])
            hi = max(np.percentile(full, self.plot_view_percentiles[1]), bounds[1], region[1])
            pad = self.plot_view_margin * (hi - lo)
            lims.append((lo - pad, hi + pad))
        return lims

    def plot(self, title="", figsize=(32,6)):
        """Summary plot of the four gating steps in one row.

        Panels: all events with the bound tessellation, clip bounds (dashed) and region (gray); the region
        tessellation (main tiles black); the candidate singlets with the density gate; the final gate on all events
        (red: outside) with the fraction of events inside.

        Parameters
        ----------
        title : str, default ""
            Figure title.
        figsize : tuple of float, default (32, 6)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : numpy.ndarray of matplotlib.axes.Axes
            The four axes.
        """
        fig, ax = plt.subplots(1,4,figsize=figsize)
        xlim, ylim = self._plot_view()
        ind = _plot_subsample(len(self.x_full), self.plot_max_points)
        xf, yf = self.x_full[ind], self.y_full[ind]

        # 1: all events, density and tiles within the clip bounds (dashed), region of step 2 (gray)
        outside = (xf<=self.bounds_x[0]) | (xf>=self.bounds_x[1]) | (yf<=self.bounds_y[0]) | (yf>=self.bounds_y[1])
        ax[0].scatter(xf[outside], yf[outside], s=1, color="0.7", rasterized=True)
        self.histogram_bound.plot(ax=ax[0], xlim=xlim, ylim=ylim, max_points=self.plot_max_points)
        ax[0].add_patch(patches.Rectangle((self.bounds_x[0], self.bounds_y[0]), self.bounds_x[1]-self.bounds_x[0], self.bounds_y[1]-self.bounds_y[0],
                                       linewidth=1.5, linestyle="--", edgecolor='black', facecolor='none'))
        ax[0].add_patch(patches.Rectangle((self.region_x[0], self.region_y[0]), self.region_x[1]-self.region_x[0], self.region_y[1]-self.region_y[0],
                                       linewidth=2, edgecolor='gray', facecolor='none'))
        ax[0].set_title("all events (gray: outside clip bounds, dashed)", size=12)

        # 2: tesselation within the region
        self.histogram_region.plot(ax=ax[1], max_points=self.plot_max_points)
        ax[1].set_title("region: tiles, main tile black", size=12)

        # 3: final tile with density gate, other events of the region in gray
        hr = self.histogram_region
        indr = _plot_subsample(len(hr.x), self.plot_max_points)
        ax[2].scatter(hr.x[indr], hr.y[indr], s=1, color="0.7", rasterized=True)
        self.densitygate.plot(ax=ax[2], xlim=self.region_x, ylim=self.region_y, max_points=self.plot_max_points)
        _style_axes(ax[2], self.name_x, self.name_y)
        ax[2].set_title("main tile: density and gate", size=12)

        # 4: gate on all events
        mask = self.get_gatemask(xf, yf)
        ax[3].scatter(xf[~mask], yf[~mask], s=1, color="red", rasterized=True)
        ax[3].scatter(xf[mask], yf[mask], s=1, color="black", rasterized=True)
        ax[3].plot(*self.densitygate.vertices_hull.T, color="white", lw=2.5)
        ax[3].plot(*self.densitygate.vertices_hull.T, color="tab:blue", lw=1.2)
        ax[3].set_xlim(xlim); ax[3].set_ylim(ylim)
        _style_axes(ax[3], self.name_x, self.name_y)
        remainder = self.get_gatemask(self.x_full, self.y_full).mean()
        ax[3].set_title(f"{remainder*100:.0f}% of all events within gate", size=12)

        if title:
            fig.suptitle(title, size=24)

        return fig, ax
    
    def get_gatemask(self, x, y):
        """Apply the final gate to events.

        Parameters
        ----------
        x, y : numpy.ndarray
            1-D arrays of the two channels, same scale as the data the gate was fitted on.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True inside the convex hull of the density gate.
        """
        return self.densitygate.hull_gatemask(np.vstack([x, y]).T)

