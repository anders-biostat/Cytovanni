"""Gates on unmixed data (DataFrames in ``adata.obsm``)."""
import flowutils
import matplotlib.pyplot as plt
import numpy as np

class CutoffUnmxGate():
    """Gate that keeps the events above (or below) a cutoff in one marker of the unmixed data.

    Parameters
    ----------
    x : str
        Marker: column of the unmixed DataFrame ``adata.obsm[key]``, e.g. ``"BV786 | CD14"``.
    xcutoff : float
        Cutoff, on the scale of ``adata.obsm[key]`` (e.g. arcsinh-transformed for a ``*_arcsinh`` key). The bound is
        inclusive.
    above : bool, default True
        True keeps the events with value >= `xcutoff`, False those with value <= `xcutoff`.
    key : str, default "unmx"
        Key of the unmixed data in ``adata.obsm`` (a DataFrame with the markers as columns, as written by
        `cytovanni.utils.add_adata_unmixed`).

    Examples
    --------
    >>> gate = CutoffUnmxGate("BV786 | CD14", 2, key="unmx_arcsinh")
    >>> adata_CD14 = adata[gate.apply(adata)]
    """
    def __init__(self, x, xcutoff, above=True, key="unmx"):
        self.x, self.xcutoff = x, xcutoff
        self.above = above
        self.key = key
    
    def __repr__(self):
        repstr = f"CutoffUnmxGate on '{self.key}', gates marker {self.x} to {'above' if self.above else 'below'} {self.xcutoff}."
        return repstr
    
    def apply(self, ad, addkey=""):
        """Apply the gate to an AnnData.

        Parameters
        ----------
        ad : anndata.AnnData
            Data with the unmixed DataFrame ``ad.obsm[key]``.
        addkey : str, default ""
            If given, also store the mask in ``ad.obs[addkey]`` (in place).

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        mask = self.apply_df(ad.obsm[self.key])
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        """Apply the gate to a DataFrame of unmixed data (what ``ad.obsm[key]`` holds, markers as columns).

        Parameters
        ----------
        df : pandas.DataFrame
            Unmixed events, markers as columns (must contain `x`).

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        x = df[self.x].to_numpy()
        return x>=self.xcutoff if self.above else x<=self.xcutoff
    
    def plot_adata(self, adata, ax=None, **kwargs):
        """Histogram (100 bins) of the marker in ``adata.obsm[key]`` with the cutoff.

        Parameters
        ----------
        adata : anndata.AnnData
            Data to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Default: a new figure.
        **kwargs
            Passed to ``ax.hist``.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x = adata.obsm[self.key][self.x]
        ax.hist(x, bins=100, **kwargs)
        ax.axvline(self.xcutoff, color="black")
        ax.set_xlabel(self.x, size=20)
        ax.set_title(f"CutoffUnmxGate on\n'{self.key}'", size=25)

class UnmxMinGate():
    """Gate that removes the events that are not above a cutoff in any marker of the unmixed data.

    Keeps an event if at least one marker is > `cutoff` (strict), e.g. to remove unstained events or debris.

    Parameters
    ----------
    key_unmx : str
        Key of the unmixed data in ``adata.obsm`` (a DataFrame with the markers as columns).
    cutoff : float
        Cutoff, on the scale of ``adata.obsm[key_unmx]`` (e.g. arcsinh-transformed for a ``*_arcsinh`` key).

    Examples
    --------
    >>> gate = UnmxMinGate("unmx_arcsinh", 1.)
    """
    def __init__(self, key_unmx, cutoff):
        self.key_unmx = key_unmx
        self.cutoff = cutoff
    
    def __repr__(self):
        repstr = f"MinMax gate above {self.cutoff} in unmixed layer {self.key_unmx}."
        return repstr
    
    def apply(self, ad, addkey=""):
        """Apply the gate to an AnnData.

        Parameters
        ----------
        ad : anndata.AnnData
            Data with the unmixed DataFrame ``ad.obsm[key_unmx]``.
        addkey : str, default ""
            If given, also store the mask in ``ad.obs[addkey]`` (in place).

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        mask = self.apply_df(ad.obsm[self.key_unmx])
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        """Apply the gate to a DataFrame (or array) of unmixed data (what ``ad.obsm[key_unmx]`` holds).

        Parameters
        ----------
        df : pandas.DataFrame
            Unmixed events, markers as columns.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        return np.any(np.asarray(df)>self.cutoff, axis=1)
    
    def plot_adata(self, adata, ax=None, **kwargs):
        """Histogram (100 bins) of the maximum over all markers of ``adata.obsm[key_unmx]``, with the cutoff.

        Parameters
        ----------
        adata : anndata.AnnData
            Data to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. Default: a new figure.
        **kwargs
            Passed to ``ax.hist``.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        x = adata.obsm[self.key_unmx].max(1)
        ax.hist(x, bins=100, **kwargs)
        ax.axvline(self.cutoff, color="black")
        ax.set_xlabel(f"Max across '{self.key_unmx}'", size=20)
        ax.set_title(f"UnmxMinGate on\n'{self.key_unmx}'", size=25)
