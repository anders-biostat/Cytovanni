"""Polygon gate on two channels of an AnnData or DataFrame."""
import flowutils
import matplotlib.pyplot as plt
import numpy as np

from ..utils import silent_log

class PolygonGate():
    """Gate that keeps the events inside a polygon in the (`x`, `y`) plane.

    Similar to the PolygonGate of flowkit, but works on the AnnData layout of this package.

    Parameters
    ----------
    x, y : str
        Channel names of the two axes: ``adata.var`` index / DataFrame columns, or column names of
        ``adata.obsm[x_obsm]`` / ``adata.obsm[y_obsm]`` if those are given.
    vertices : numpy.ndarray
        Polygon vertices of shape (vertices, 2), columns x and y, in the units of the data gated (e.g. raw or
        calibrated intensities). The polygon is closed automatically. Must be an array (not a list) for `plot`.
    name : str, default ""
        Gate name, used as plot label.
    color : color, optional
        Matplotlib colour of the polygon in plots.
    layer : str, default "raw"
        Layer of the AnnData the gate is applied to, for the axes not taken from ``obsm``.
    onlog : bool, default False
        Test the containment on the natural log of vertices and data, i.e. the edges are straight lines in a log-log
        plot; `plot` then sets log axes. Events with non-positive values are not inside the polygon.
    x_obsm, y_obsm : str, optional
        Take the x (y) values from the DataFrame ``adata.obsm[x_obsm][x]`` (e.g. unmixed data) instead of from
        `layer`. Default: from `layer`. Only used by `apply`, `get_points` and `plot_adata`, not by `apply_df`.

    Examples
    --------
    >>> gate = PolygonGate("FSC-A", "SSC-A", np.asarray([[6e4, 1e4], [6e4, 8e4], [1.6e5, 2.2e5], [3e5, 5e4]]),
    ...                    name="cells", layer="calibrated", color="black")
    >>> adata = adata[gate.apply(adata)]
    """
    def __init__(self, x, y, vertices, name="", color=None, layer="raw", onlog=False, x_obsm=None, y_obsm=None):
        self.x, self.y = x, y
        self.vertices = vertices
        self.name = name
        self.color = color
        self.layer = layer
        self.onlog = onlog
        self.x_obsm = x_obsm
        self.y_obsm = y_obsm
        self.any_obsm = not ((self.x_obsm is None) and (self.y_obsm is None))
    
    def __repr__(self):
        repstr = f"Polygon gate '{self.name}', on channels {self.x} and {self.y}."
        return repstr
    
    def apply_points(self, points):
        """Boolean mask of shape (events,): which of `points` (shape (events, 2), columns x, y) are inside."""
        if self.onlog:
            return flowutils.gating.points_in_polygon(silent_log(self.vertices), silent_log(points))
        else:
            return flowutils.gating.points_in_polygon(self.vertices, points)

    def get_points(self, ad):
        """Values of the two gate axes for the events of `ad`.

        Parameters
        ----------
        ad : anndata.AnnData
            Data with ``ad.layers[layer]`` and, if set, ``ad.obsm[x_obsm]`` / ``ad.obsm[y_obsm]``.

        Returns
        -------
        numpy.ndarray
            Array of shape (events, 2), columns x and y.
        """
        if self.any_obsm:
            x = ad[:,self.x].layers[self.layer][:,0] if self.x_obsm is None else ad.obsm[self.x_obsm][self.x]
            y = ad[:,self.y].layers[self.layer][:,0] if self.y_obsm is None else ad.obsm[self.y_obsm][self.y]
            points = np.vstack([x, y]).T
        else:
            points = ad[:,[self.x, self.y]].layers[self.layer]
        return points
    
    def apply(self, ad, addkey=""):
        """Apply the gate to an AnnData.

        Parameters
        ----------
        ad : anndata.AnnData
            Data as described in `get_points`.
        addkey : str, default ""
            If given, also store the mask in ``ad.obs[addkey]`` (in place).

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events inside.
        """
        points = self.get_points(ad)
        mask = self.apply_points(points)
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        """Apply the gate to a DataFrame with columns `x` and `y`, e.g. the raw .fcs data.

        `layer` and the ``obsm`` settings are not used.

        Parameters
        ----------
        df : pandas.DataFrame
            Events with the columns `x` and `y`.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events inside.
        """
        points = df[[self.x, self.y]].to_numpy()
        mask = self.apply_points(points)
        return mask
    
    def plot(self, ax, addlegend=True):
        """Draw the polygon into existing axes (log axes if `onlog`).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw into.
        addlegend : bool, default True
            Label the polygon with `name` for a legend.
        """
        pts = np.vstack([self.vertices,self.vertices[[0]]])
        ax.plot(*pts.T, color=self.color, label=self.name if addlegend else "")
        if self.onlog:
            ax.set_xscale("log")
            ax.set_yscale("log")
    
    def plot_adata(self, adata, ax=None, **kwargs):
        """Scatter plot of the two gate axes of `adata` (see `get_points`) with the polygon.

        Parameters
        ----------
        adata : anndata.AnnData
            Data to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Default: a new figure.
        **kwargs
            Passed to ``ax.scatter``, e.g. ``s=.1``.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x, y = self.get_points(adata).T
        ax.scatter(x, y, **kwargs)
        self.plot(ax)
        ax.set_xlabel(self.x, size=20)
        ax.set_ylabel(self.y, size=20)
        ax.set_title(f"Gate '{self.name}'", size=25)
