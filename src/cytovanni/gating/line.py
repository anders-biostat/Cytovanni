"""Line gate: keep the events on one side of a straight line in a 2-D plot."""
import numpy as np
import matplotlib.pyplot as plt

def _distance_from_line(p1, p2, p):
    """Signed distance of 2-D points `p` (shape (N, 2)) from the line through `p1`, `p2` (positive: right of p1->p2)."""
    d = p2 - p1
    v = p - p1[None]
    # 2D cross product (np.cross on 2D vectors is deprecated in numpy 2)
    return -(d[0]*v[:,1] - d[1]*v[:,0])/np.linalg.norm(d)

def _closest_on_line(p1, p2, p):
    """Closest points on the line through `p1` and `p2` to the points `p` (shape (N, 2)); 2-D only."""
    d = p2[None]-p1[None]
    assert d.shape[-1]==2
    det = (d**2).sum(-1, keepdims=True)
    a = (d * (p-p1[None])).sum(-1, keepdims=True) / det
    pint = p1[None] + a * d
    return pint

class LineGate():
    """Gate that keeps the events below (or above) the line through `p1` and `p2` in the (`x`, `y`) plane.

    Parameters
    ----------
    x, y : str
        Channel names (``adata.var`` index / DataFrame columns) of the two axes.
    p1, p2 : tuple of float
        Two points (x, y) on the line, in the units of `layer`, in any order. The line must not be vertical, since
        "below" is defined by y.
    below : bool, default True
        True keeps the events with lower y than the line at their x, False those with higher y. Events exactly on
        the line are removed.
    name : str, default ""
        Gate name, used as plot label.
    color : color, optional
        Matplotlib colour of the line in plots.
    layer : str, default "raw"
        Layer of the AnnData the gate is applied to (`apply`, `get_distance`, `set_distance_scaling_factors`,
        `plot_adata`).
    scale_x, scale_y : float, default 1.0
        Positive factors applied to the two axes before the distance to the line is computed (see
        `set_distance_scaling_factors`); they do not change which side is kept.

    Raises
    ------
    ValueError
        If ``p1[0] == p2[0]`` (vertical line).

    Examples
    --------
    >>> gate = LineGate("FSC-A", "SSC-A", (5e4, 2e5), (2e5, 1e5), below=True, name="no debris")
    >>> mask = gate.apply(adata, addkey="gate_line")
    """
    def __init__(self, x, y, p1, p2, below=True, name="", color=None, layer="raw", scale_x=1., scale_y=1.):
        self.x, self.y = x, y
        self.scale_x, self.scale_y = scale_x, scale_y
        p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
        if p1[0] == p2[0]:
            raise ValueError(f"LineGate '{name}': p1 and p2 have the same x ({p1[0]}), a vertical line has no below/above!")
        # orient the line from lower to higher x, so that a positive distance always means lower y
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        self.p1, self.p2 = p1, p2
        self.below = below
        self.name = name
        self.color = color
        self.layer = layer
    
    def __repr__(self):
        repstr = f"Line gate '{self.name}', on channels {self.x} and {self.y}."
        return repstr
    
    def set_distance_scaling_factors(self, ad):
        """Set `scale_x` and `scale_y` to 1 / mean of the two channels in ``ad.layers[layer]`` (in place).

        Afterwards `get_distance` returns distances in units of the channel means. Only the distances change, not
        which events are kept.

        Parameters
        ----------
        ad : anndata.AnnData
            Data to take the means from.
        """
        self.scale_x = 1 / np.mean(ad[:,self.x].layers[self.layer][:,0])
        self.scale_y = 1 / np.mean(ad[:,self.y].layers[self.layer][:,0])
    
    def get_distance_points(self, points):
        """Signed, scaled distance of points from the line.

        Parameters
        ----------
        points : array-like
            Points of shape (events, 2), columns `x` and `y`, in the units of `layer`.

        Returns
        -------
        numpy.ndarray
            Distance of shape (events,) after scaling the axes by `scale_x`, `scale_y`; positive means kept (lower y
            than the line if `below`, higher y otherwise).
        """
        scaling = np.asarray([self.scale_x, self.scale_y])
        dist = _distance_from_line(scaling*self.p1, scaling*self.p2, scaling[None]*np.asarray(points, dtype=float))
        return dist if self.below else -dist

    def get_distance(self, ad):
        """Signed, scaled distance of the events of `ad` (``ad.layers[layer]``) from the line.

        Parameters
        ----------
        ad : anndata.AnnData
            Data with the channels `x`, `y` in ``ad.var`` and the layer `layer`.

        Returns
        -------
        numpy.ndarray
            Distance of shape (events,); positive means kept.
        """
        return self.get_distance_points(ad[:,[self.x, self.y]].layers[self.layer].copy())
    
    def apply(self, ad, addkey=""):
        """Apply the gate to an AnnData.

        Parameters
        ----------
        ad : anndata.AnnData
            Data with the channels `x`, `y` in ``ad.var`` and the layer `layer`.
        addkey : str, default ""
            If given, also store the mask in ``ad.obs[addkey]`` (in place).

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        dist = self.get_distance(ad)
        mask = dist>0
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        """Apply the gate to a DataFrame with columns `x` and `y` (e.g. the raw .fcs data; `layer` is not used).

        Parameters
        ----------
        df : pandas.DataFrame
            Events with the columns `x` and `y`.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        return self.get_distance_points(df[[self.x, self.y]].to_numpy())>0
    
    def plot(self, ax, addlegend=True, mark_outside=True):
        """Draw the line into existing axes, keeping their limits.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes with the data already drawn (the line spans their current x range).
        addlegend : bool, default True
            Label the line with `name` for a legend.
        mark_outside : bool, default True
            Shade the side of the line that is removed (red).
        """
        args = np.polyfit([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], 1)
        line = np.poly1d(args)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        x = np.linspace(min(xlim), max(xlim))
        ax.plot(x, line(x), color=self.color, label=self.name if addlegend else "")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if mark_outside:
            if self.below:
                ax.fill_between(x, line(x), ylim[1], interpolate=True, color='red', alpha=.1)
            else:
                ax.fill_between(x, ylim[0], line(x), interpolate=True, color='red', alpha=.1)
    
    def plot_adata(self, adata, ax=None, **kwargs):
        """Scatter plot of the two channels of `adata` (``adata.layers[layer]``) with the gate.

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
        
        x, y = adata[:,self.x].layers[self.layer][:,0], adata[:,self.y].layers[self.layer][:,0]
        ax.scatter(x, y, **kwargs)
        self.plot(ax)
        ax.set_xlabel(self.x, size=20)
        ax.set_ylabel(self.y, size=20)
        ax.set_title(f"Gate '{self.name}'", size=25)

