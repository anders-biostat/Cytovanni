import flowutils
import matplotlib.pyplot as plt
import numpy as np

class CutoffUnmxGate():
    """ Gate on unmixed data above/below cuoff.
    """
    def __init__(self, x, xcutoff, above=True, key="unmx"):
        self.x, self.xcutoff = x, xcutoff
        self.above = above
        self.key = key
    
    def __repr__(self):
        repstr = f"CutoffUnmxGate on '{self.key}', gates marker {self.x} to {'above' if self.above else 'below'} {self.xcutoff}."
        return repstr
    
    def apply(self, ad):
        x = ad.obsm[self.key][self.x]
        if self.above:
            return x>=self.xcutoff
        else:
            return x<=self.xcutoff
    
    def plot_adata(self, adata, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        
        x = adata.obsm[self.key][self.x]
        ax.hist(x, bins=100, **kwargs)
        ax.axvline(self.xcutoff, color="black")
        ax.set_xlabel(self.x, size=20)
        ax.set_title(f"CutoffUnmxGate on\n'{self.key}'", size=25)

class UnmxMinGate():
    """ Gate AnnData to minimal marker abundance across whole panel.
        Excludes events that are below cutoff in every marker.
    """
    def __init__(self, key_unmx, cutoff):
        self.key_unmx = key_unmx
        self.cutoff = cutoff
    
    def __repr__(self):
        repstr = f"MinMax gate above {self.cutoff} in unmixed layer {self.key_unmx}."
        return repstr
    
    def apply(self, ad, addkey=""):
        vals = ad.obsm[self.key_unmx]
        mask = np.any(vals>self.cutoff, axis=1)
        if addkey:
            ad.obs[addkey] = mask
        return mask
    
    def plot_adata(self, adata, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        
        x = adata.obsm[self.key_unmx].max(1)
        ax.hist(x, bins=100, **kwargs)
        ax.axvline(self.cutoff, color="black")
        ax.set_xlabel(f"Max across '{self.key_unmx}'", size=20)
        ax.set_title(f"UnmxMinGate on\n'{self.key_unmx}'", size=25)
