import flowutils
import matplotlib.pyplot as plt
import numpy as np

from ..utils import silent_log

class PolygonGate():
    """ Gate data within a polygon.
        Similar syntax as PolygonGate from flowkit, but works with our AnnData format.
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
        if self.onlog:
            return flowutils.gating.points_in_polygon(silent_log(self.vertices), silent_log(points))
        else:
            return flowutils.gating.points_in_polygon(self.vertices, points)

    def get_points(self, ad):
        if self.any_obsm:
            x = ad[:,self.x].layers[self.layer][:,0] if self.x_obsm is None else ad.obsm[self.x_obsm][self.x]
            y = ad[:,self.y].layers[self.layer][:,0] if self.y_obsm is None else ad.obsm[self.y_obsm][self.y]
            points = np.vstack([x, y]).T
        else:
            points = ad[:,[self.x, self.y]].layers[self.layer]
        return points
    
    def apply(self, ad, addkey=""):
        points = self.get_points(ad)
        mask = self.apply_points(points)
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        points = df[[self.x, self.y]].to_numpy()
        mask = self.apply_points(points)
        return mask
    
    def plot(self, ax, addlegend=True):
        pts = np.vstack([self.vertices,self.vertices[[0]]])
        ax.plot(*pts.T, color=self.color, label=self.name if addlegend else "")
        if self.onlog:
            ax.set_xscale("log")
            ax.set_yscale("log")
    
    def plot_adata(self, adata, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        
        x, y = self.get_points(adata).T
        ax.scatter(x, y, **kwargs)
        self.plot(ax)
        ax.set_xlabel(self.x, size=20)
        ax.set_ylabel(self.y, size=20)
        ax.set_title(f"Gate '{self.name}'", size=25)
