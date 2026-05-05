import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, QhullError
import flowutils
from sklearn.neighbors import KernelDensity

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

class HistogramTesselator():
    USE_APPROX_KDE_TREE = False # doesn't currently work!
    MIN_LOGDENSITY = -50
    MAXIMUM_TOLERANCE = 1e-7
    
    """ Single histogram, peak finding and tesselation.
        
        Replace kde estimation by much faster FFT version!
    """
    def __init__(self, x, y, bounds_x, bounds_y, Nx=100, Ny=100, bw_factor=3., neigh_size=3, density_exclusion_corner=.05, name_x="FSC-A", name_y="SSC-A", Nsubsample=None):
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
        self.point_to_ind = lambda X: np.vstack([np.searchsorted(self.bins_x, X[:,0]), np.searchsorted(self.bins_y, X[:,1])]).T
        
        # kde density estimate
        if self.USE_APPROX_KDE_TREE:
            X = np.vstack([x, y]).T
            self.density_kde = KernelDensity(bandwidth=bw_factor * _scotts_factor(X), atol=1e-6, leaf_size=200, algorithm="ball_tree").fit(X)
            ind_X, ind_Y = np.mgrid[0:Nx, 0:Ny]
            X = np.vstack([self.bins_center_x[ind_X].flatten(), self.bins_center_y[ind_Y].flatten()]).T
            self.logdensity = np.clip(self.density_kde.score_samples(X).reshape((Nx, Ny)), a_min=self.MIN_LOGDENSITY, a_max=None)
            self.density = np.exp(self.logdensity)
        else:
            self.density_kde = gaussian_kde(np.vstack([x, y]), lambda kde: bw_factor * gaussian_kde.scotts_factor(kde))
            ind_X, ind_Y = np.mgrid[0:Nx, 0:Ny]
            self.density = self.density_kde(np.vstack([self.bins_center_x[ind_X].flatten(), self.bins_center_y[ind_Y].flatten()])).reshape((Nx, Ny))
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
        self.maxima_indices = np.argwhere(self.density==density_max_neighbors)
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
        else:
            if self.maxima_indices.shape[0]==2:
                # can't deal with only two points for some reason, add dummy far away
                self.maxima_indices = np.vstack([self.maxima_indices,[[-Nx, -Ny], [-Nx, -Ny-1], [-Nx-1, -Ny-1]]])
            try:
                voronoi = Voronoi(self.maxima_indices)
            except QhullError:
                # If points are coplanar qhull fails, try adding dummies far away
                self.maxima_indices = np.vstack([self.maxima_indices,[[-Nx, -Ny], [-Nx, -Ny-1], [-Nx-1, -Ny-1]]])
                voronoi = Voronoi(self.maxima_indices)
            self.voronoi = ScaledVoronoi(voronoi, self.ind_to_point, self.point_to_ind)
            self.voronoi_label = self.voronoi.assign_group(np.vstack([self.x,self.y]).T)
            u, c = np.unique(self.voronoi_label, return_counts=True)
            self.voronoi_label_ismain = self.voronoi_label==u[np.argmax(c)]
    
    def plot(self, ax=None, xlim=None, ylim=None):
        """ Voronoi plot at the boundaries sometimes wrong?!
        """
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(6,6))

        ax.imshow(self.logdensity[:,::-1].T, extent=(*self.bounds_x, *self.bounds_y), aspect="auto")
        mask = self.voronoi_label_ismain
        ax.scatter(self.x[~mask], self.y[~mask], s=1, color="red")
        ax.scatter(self.x[mask], self.y[mask], s=1, color="black")
        ax.scatter(*self.maxima.T, color="white")
        if self.voronoi is not None:
            voronoi_plot_2d(self.voronoi, ax=ax, show_vertices=False)
        ax.set_xlim(self.bounds_x if xlim is None else xlim)
        ax.set_ylim(self.bounds_y if ylim is None else ylim)
        ax.set_xlabel(self.name_x, size=20)
        ax.set_ylabel(self.name_y, size=20)
        
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
        density_kde = gaussian_kde(self.X.T, lambda kde: factor_kde * gaussian_kde.scotts_factor(kde))
        density = density_kde(self.X.T)
        self.density_scaled = np.argsort(np.argsort(density))/(len(density)-1)
        
        X_ = self.X[self.density_scaled>cutoff]
        self.convex_hull = ConvexHull(X_)
        vertices_hull = X_[self.convex_hull.vertices]
        self.vertices_hull = np.vstack([vertices_hull, vertices_hull[[0]]])
    
    def hull_gatemask(self, X):
        return flowutils.gating.points_in_polygon(self.vertices_hull, X)
    
    def plot(self, ax=None, xlim=None, ylim=None):
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=(6,6))
        
        sns.scatterplot(x=self.X[:,0], y=self.X[:,1], hue=self.density_scaled, s=5, palette="viridis", ax=ax, legend=False)
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
    
    subsample_kde_estimates = 20000
    
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
                     name_x=self.name_x, name_y=self.name_y, Nsubsample=self.subsample_kde_estimates)
        # get region for second step
        self.region_x, self.region_y = self.histogram_bound.rectangle_from_main(self.bound_density_mad_factor)
        
        # get region histogram
        mask = (self.x>self.region_x[0]) & (self.x<self.region_x[1]) & (self.y>self.region_y[0]) & (self.y<self.region_y[1])
        self.histogram_region = HistogramTesselator(self.x[mask], self.y[mask], self.region_x, self.region_y, self.region_density_grid_N, self.region_density_grid_N,
                     bw_factor=self.region_density_bw_factor, neigh_size=self.region_density_neigh_size, density_exclusion_corner=self.region_density_exclusion_corner,
                     name_x=self.name_x, name_y=self.name_y, Nsubsample=self.subsample_kde_estimates)
            
        # data in final tile
        self.x_final = self.histogram_region.x[self.histogram_region.voronoi_label_ismain]
        self.y_final = self.histogram_region.y[self.histogram_region.voronoi_label_ismain]
        
        # get final density gate
        self.densitygate = DensityGate(np.vstack([self.x_final, self.y_final]).T, cutoff=self.final_density_cutoff, factor_kde=self.final_density_bw_factor, Nsubsample=self.subsample_kde_estimates)
        
        if self.verbose:
            self.plot()
    
    def plot(self, title="", figsize=(24,6)):
        fig, ax = plt.subplots(1,3,figsize=figsize)
        
        self.histogram_bound.plot(ax=ax[0])
        ax[0].add_patch(patches.Rectangle((self.region_x[0], self.region_y[0]), self.region_x[1]-self.region_x[0], self.region_y[1]-self.region_y[0],
                                       linewidth=2, edgecolor='gray', facecolor='none'))

        self.histogram_region.plot(ax=ax[1]) #(xlim=self.bounds_x, ylim=self.bounds_y)

        self.densitygate.plot(ax=ax[2], xlim=self.region_x, ylim=self.region_y)
        ax[2].set_xlabel(self.name_x, size=20)
        ax[2].set_ylabel(self.name_y, size=20)
        
        if title:
            fig.suptitle(title, size=30)
        
        remainder = self.get_gatemask(self.x_full, self.y_full).mean()
        ax[2].set_title(f"{remainder*100:.0f}% of all Events within Gate", size=15)
        
        return fig, ax
    
    def get_gatemask(self, x, y):
        """ Get gate mask for final gate.
        """
        return self.densitygate.hull_gatemask(np.vstack([x, y]).T)

