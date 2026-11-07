import numpy as np
import matplotlib.pyplot as plt

def _distance_from_line(p1, p2, p):
    """ Get signed distance of 2D points p (shape (N, 2)) from the line defined by points p1 and p2.
    """
    d = p2 - p1
    v = p - p1[None]
    # 2D cross product (np.cross on 2D vectors is deprecated in numpy 2)
    return -(d[0]*v[:,1] - d[1]*v[:,0])/np.linalg.norm(d)

def _closest_on_line(p1, p2, p):
    """ Get closest point on line defined by p1 and p2 to points p.
        Only works for 2D!
    """
    d = p2[None]-p1[None]
    assert d.shape[-1]==2
    det = (d**2).sum(-1, keepdims=True)
    a = (d * (p-p1[None])).sum(-1, keepdims=True) / det
    pint = p1[None] + a * d
    return pint

class LineGate():
    """ Gate data above/below a line.
    """
    def __init__(self, x, y, p1, p2, below=True, name="", color=None, layer="raw", scale_x=1., scale_y=1.):
        """ Gate keeping the events below (or above) the line through p1 and p2 in the (x, y) plane.

            :param x, y: str. Channel names (adata.var index / DataFrame columns) of the two axes.

            :param p1, p2: 2-tuples (x, y). Two distinct points on the line, in any order. The line must not be
                vertical (p1[0] == p2[0] raises ValueError), since "below" is defined by y.

            :param below: bool. True keeps the events with lower y than the line at their x, False those with higher y.

            :param name: str. Gate name, used as plot label.

            :param color: matplotlib color of the line in plots.

            :param layer: str. adata layer the gate is applied to.

            :param scale_x, scale_y: positive floats. Axis scaling used for the distance to the line
                (see set_distance_scaling_factors); they do not change which side is kept.
        """
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
        """ Set scaling factors from mean across ad.
        """
        self.scale_x = 1 / np.mean(ad[:,self.x].layers[self.layer][:,0])
        self.scale_y = 1 / np.mean(ad[:,self.y].layers[self.layer][:,0])
    
    def get_distance_points(self, points):
        """ Signed, scaled distance of points (shape (N, 2), columns x and y) from the line; positive means kept
            (lower y than the line if below, higher y otherwise). """
        scaling = np.asarray([self.scale_x, self.scale_y])
        dist = _distance_from_line(scaling*self.p1, scaling*self.p2, scaling[None]*np.asarray(points, dtype=float))
        return dist if self.below else -dist

    def get_distance(self, ad):
        return self.get_distance_points(ad[:,[self.x, self.y]].layers[self.layer].copy())
    
    def apply(self, ad, addkey=""):
        dist = self.get_distance(ad)
        mask = dist>0
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        """ Apply the gate to a DataFrame with columns x and y (e.g. the raw .fcs data), returns a boolean array. """
        return self.get_distance_points(df[[self.x, self.y]].to_numpy())>0
    
    def plot(self, ax, addlegend=True, mark_outside=True):
        """ Draw the line into ax (keeping its limits); with mark_outside, shade the side that is removed. """
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
        if ax is None:
            fig, ax = plt.subplots()
        
        x, y = adata[:,self.x].layers[self.layer][:,0], adata[:,self.y].layers[self.layer][:,0]
        ax.scatter(x, y, **kwargs)
        self.plot(ax)
        ax.set_xlabel(self.x, size=20)
        ax.set_ylabel(self.y, size=20)
        ax.set_title(f"Gate '{self.name}'", size=25)

