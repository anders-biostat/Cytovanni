"""Range gates: keep events whose values lie within [v_min, v_max] in every given channel."""
import numpy as np

class MinMaxGate():
    """Gate that keeps the events within [`v_min`, `v_max`] in all given channels (bounds inclusive).

    Typically used on the raw data to remove events outside the linear range of the detectors, e.g. as `gates` of
    `cytovanni.io.readfcs_sample`.

    Parameters
    ----------
    channels : list of str
        Channels (``adata.var`` index / DataFrame columns) that all have to be within the range.
    v_min, v_max : float
        Lower and upper bound, in the units of `layer`.
    layer : str, default "raw"
        Layer of the AnnData the gate is applied to (`apply` only).

    Examples
    --------
    >>> gate = MinMaxGate(["FSC-A", "FSC-H", "SSC-A", "SSC-H"], 0, 4e5, layer="raw")
    >>> adata = readfcs_sample(path, cytoconfig, gates=[gate])
    """
    def __init__(self, channels, v_min, v_max, layer="raw"):
        self.channels = list(channels)
        self.v_min, self.v_max = v_min, v_max
        self.layer = layer
    
    def __repr__(self):
        repstr = f"MinMax gate within [{self.v_min}, {self.v_max}], gate on channels {self.channels}."
        return repstr
    
    def apply(self, ad, addkey=""):
        """Apply the gate to an AnnData.

        Parameters
        ----------
        ad : anndata.AnnData
            Data with the channels in ``ad.var`` and the layer `layer`.
        addkey : str, default ""
            If given, also store the mask in ``ad.obs[addkey]`` (in place).

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        vals = ad[:,self.channels].layers[self.layer]
        mask = np.all((vals>=self.v_min) & (vals <=self.v_max), axis=1)
        if addkey:
            ad.obs[addkey] = mask
        return mask

    def apply_df(self, df):
        """Apply the gate to a DataFrame with the channels as columns (e.g. the raw .fcs data; `layer` is not used).

        Parameters
        ----------
        df : pandas.DataFrame
            Events with the gate channels as columns.

        Returns
        -------
        numpy.ndarray
            Boolean mask of shape (events,), True for the events kept.
        """
        vals = df[self.channels].to_numpy()
        mask = np.all((vals>=self.v_min) & (vals <=self.v_max), axis=1)
        return mask

class FluorescenceMinMaxGate(MinMaxGate):
    """`MinMaxGate` on all fluorescence channels of a cytometer configuration.

    Parameters
    ----------
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Its ``channels_fluorescence`` are gated (not the optional -H/-W channels) and its ``max_linear_range`` sets
        the default upper bound.
    v_min : float, default -1000
        Lower bound, in the units of `layer` (raw intensities by default).
    max_tolerance : float, default 1.5
        Default upper bound: ``cytoconfig.max_linear_range * max_tolerance``. Ignored if `v_max` is given.
    layer : str, default "raw"
        Layer of the AnnData the gate is applied to.
    v_max : float, optional
        Explicit upper bound. Default: from `max_tolerance`.

    Examples
    --------
    >>> gate = FluorescenceMinMaxGate(cytoconfig, -1e3, 1.5, layer="raw")
    """
    def __init__(self, cytoconfig, v_min=-1e3, max_tolerance=1.5, layer="raw", v_max=None):
        super().__init__(cytoconfig.channels_fluorescence, v_min, cytoconfig.max_linear_range*max_tolerance if v_max is None else v_max, layer=layer)
    
