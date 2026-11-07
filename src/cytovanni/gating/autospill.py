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
    return X.shape[0]**(-1./(X.shape[1]+4))

def _clip_to_percentiles(x, minperc, maxperc):
    """ Clip values to min and max percentiles along axis.
    """
    return np.clip(x, a_min=np.percentile(x, minperc), a_max=np.percentile(x, maxperc))

def _clip_to_medianfactor(x, fmin=.5, fmax=1.5):
    """ Clip values to min and max percentiles along axis.
    """
    return np.clip(x, a_min=fmin*np.median(x), a_max=fmax*np.median(x))

def _mad(data, axis=None):
    """ Mean absolute deviation.
    """
    return np.mean(np.abs(data - np.mean(data, axis)), axis)

def _plot_subsample(n, max_points):
    """ Indices of at most max_points of n events for plotting (fixed seed, so plots are reproducible).
    """
    if max_points is None or n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, max_points, replace=False))

def _style_axes(ax, name_x, name_y):
    ax.set_xlabel(name_x, size=16)
    ax.set_ylabel(name_y, size=16)
    ax.tick_params(labelsize=12)

class _BinnedKDE(gaussian_kde):
    """gaussian_kde (2D) evaluated via linear binning on a grid + FFT convolution with the full-covariance Gaussian,
    then linear interpolation at the points. Bandwidth/covariance exactly as gaussian_kde."""
    N_GRID = 512 # 256 can erase maxima the exact KDE finds; 512 matched exact gates in tests (density error ~3e-4 of max)
    PAD_SD = 4.

    def evaluate(self, points):
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
    """ Voronoi, but scaled to translate between index in grid and actual values.
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
        group = KDTree(self.points_index).query(self.point_to_ind(X))[1]
        return group

    def plot(self, ax=None, show_points=True, show_vertices=False,
             line_colors="k", line_width=1., line_alpha=1., ray_length=100.):
        """ Replacement for voronoi_plot_2d.
            Ridge geometry is built in index space, where the perpendicular
            bisector construction is valid, and only then mapped through
            ind_to_point. Passing the scaled object to voronoi_plot_2d is
            wrong because it re-derives ray directions by rotating the
            tangent 90 deg, which the anisotropic scaling does not preserve.
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
    USE_APPROX_KDE_TREE = False # doesn't currently work!
    MIN_LOGDENSITY = -50
    MAXIMUM_TOLERANCE = 1e-7
    MIN_REL_MAXIMUM = 1e-6 # ignore maxima below this fraction of the peak density (flat empty regions)
    PLOT_LOGDENSITY_RANGE = 12. # log density colour range in plots
    
    """ Single histogram, peak finding and tesselation.
        
        Replace kde estimation by much faster FFT version!
    """
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
        """ Minimum density on the straight line (grid index space) between maxima i (main) and j, relative to the density
            at maximum i. High: j sits on a ridge connected to the main peak (e.g. sub-bands of one bead population);
            low: separated by a valley, or j is a small bump far from the main peak.
        """
        a, b = self.maxima_indices[i], self.maxima_indices[j]
        path = np.rint(a[None] + np.linspace(0, 1, n)[:,None] * (b - a)[None]).astype(int)
        return self.density[path[:,0], path[:,1]].min() / self.density[tuple(a)]
    
    def plot(self, ax=None, xlim=None, ylim=None, max_points=20000):
        """ Log density, events coloured by Voronoi tile (main tile black), maxima and tile borders.
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
        X = np.vstack([self.x,self.y]).T[self.voronoi_label_ismain]
        center = np.median(X, axis=0)
        mad = _mad(X, axis=0)
        xlim = (center[0] - mad_factor*mad[0], center[0] + mad_factor*mad[0])
        ylim = (center[1] - mad_factor*mad[1], center[1] + mad_factor*mad[1])
        return xlim, ylim

class DensityGate():
    USE_APPROX_KDE_TREE=False # not implemented yet
    
    """ Final gate with kde density cutoff.
    """
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
        return flowutils.gating.points_in_polygon(self.vertices_hull, X)
    
    def plot(self, ax=None, xlim=None, ylim=None, max_points=20000):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(6,6))

        ind = _plot_subsample(self.X.shape[0], max_points)
        ax.scatter(self.X[ind,0], self.X[ind,1], c=self.density_scaled[ind], s=5, cmap="viridis", edgecolor="w", linewidth=.3, rasterized=True)
        ax.plot(*self.vertices_hull.T, color="black")
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        
        return ax

class AutoSpillTesselationGating():
    """ Tesselation gating that is used by AutoSpill, ported to Python from:
        https://github.com/carlosproca/autospill/blob/master/R/do_gate.r
        
        All the Voronoi tesselations are done on the index space of the grid,
        not the actual values. Beads may have very different FSC/SSC values,
        otherwise will yield strange tiles.
        
        Initial trimming happens on the recorded data, I think AutoSpill uses something slightly different?
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
        """ Plot limits: union of the plot_view_percentiles range of all events, the clip bounds and the region, plus a margin.
        """
        lims = []
        for full, bounds, region in [(self.x_full, self.bounds_x, self.region_x), (self.y_full, self.bounds_y, self.region_y)]:
            lo = min(np.percentile(full, self.plot_view_percentiles[0]), bounds[0], region[0])
            hi = max(np.percentile(full, self.plot_view_percentiles[1]), bounds[1], region[1])
            pad = self.plot_view_margin * (hi - lo)
            lims.append((lo - pad, hi + pad))
        return lims

    def plot(self, title="", figsize=(32,6)):
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
        """ Get gate mask for final gate.
        """
        return self.densitygate.hull_gatemask(np.vstack([x, y]).T)

