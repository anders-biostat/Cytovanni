"""Singlet gating of bead samples (rainbow beads, single-stain controls)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .autospill import AutoSpillTesselationGating
from ..exceptions import ScatterGatingChannelException

def gate_scatter_bead_sample(adata, gatechannels=None, plot=False, plotpath="", apply=True, titlefct=lambda ad: ad.uns["name_uid"], robust=False, very_robust=False, **gating_kwargs):
    """Gate a bead sample on singlets in a scatter plot, with the AutoSpill tessellation gating.

    Runs `AutoSpillTesselationGating` on the raw values of the two scatter channels, with median-based trimming
    (``trim_beadmode=True``) and the defaults below, and stores the result in `adata`. Used for rainbow beads and for
    single-stain controls.

    Parameters
    ----------
    adata : anndata.AnnData
        Bead sample as read by `cytovanni.io.readfcs_sample` / `readfcs_rainbow_sample`; the gate uses
        ``adata.layers['raw']``. With the default `gatechannels` and `titlefct`, ``adata.uns['cytoconfig']`` and
        ``adata.uns['name_uid']`` are needed.
    gatechannels : list of str, optional
        The two channels to gate on, as (x, y).
        Default: ``adata.uns['cytoconfig'].get_default_scatter_gate_channels()``.
    plot : bool, default False
        Show the summary plot of the gating (`AutoSpillTesselationGating.plot`).
    plotpath : str, default ""
        If given, save the summary plot there (dpi 120); the figure is closed afterwards unless `plot`.
    apply : bool, default True
        Return the gated events (a copy) instead of `adata` itself.
    titlefct : callable, default ``lambda ad: ad.uns["name_uid"]``
        Function of `adata` that returns the plot title; pass e.g. ``lambda ad: "name"`` for a fixed title.
    robust : bool, default False
        More tolerant settings for beads with patterns such as two close peaks instead of one (wider trimming range
        0.25-2 x median, more KDE smoothing in the region tessellation, larger region). Noticeably worse at excluding
        large doublet peaks, so use with caution.
    very_robust : bool, default False
        Even more tolerant settings (larger region and more smoothing in both tessellations). Only has an effect
        together with `robust`.
    **gating_kwargs
        Further `AutoSpillTesselationGating` attributes, e.g. ``main_tile="density"`` or
        ``merge_tiles_min_saddle=.6``. They override the defaults set here, including those of `robust` /
        `very_robust` (e.g. `final_density_cutoff`).

    Returns
    -------
    anndata.AnnData
        If `apply`, a copy of `adata` restricted to the events inside the gate; otherwise `adata` itself.

    Raises
    ------
    ScatterGatingChannelException
        If a gate channel is not in ``adata.var`` or not exactly two channels are given.

    Notes
    -----
    `adata` is changed in place in both cases: ``adata.obs['keepmask_scatter']`` (bool, inside the gate),
    ``adata.uns['Nbeads_afterscattergate']`` (number of events inside) and ``adata.uns['density_gate_hull']``
    (hull vertices of shape (vertices + 1, 2), raw values of the two channels). The gated copy carries the same
    entries.

    Default settings: ``final_density_cutoff=0.2`` and ``bound_density_mad_factor=4`` (`robust`: 5, also
    ``trim_beadmode_min_clipfactor=0.25``, ``trim_beadmode_max_clipfactor=2``, ``region_density_bw_factor=3``;
    `very_robust`: 10, as robust but ``region_density_bw_factor=7``, ``bound_density_bw_factor=6``); all others are
    the class defaults of `AutoSpillTesselationGating`.

    Examples
    --------
    >>> adata = gate_scatter_bead_sample(adata, gatechannels=["FSC-A", "SSC-A"], plotpath="gate.png")
    """
    if gatechannels is None:
        gatechannels = adata.uns["cytoconfig"].get_default_scatter_gate_channels()
    if not np.all(np.isin(gatechannels, adata.var.index.to_numpy())):
        raise ScatterGatingChannelException(f"Chosen scatter channels to gate on {gatechannels} are not all available! Choose two of available channels:\n{adata.var.index.to_numpy()}")
    if len(gatechannels)!=2:
        raise ScatterGatingChannelException(f"Chosen scatter channels to gate on {gatechannels} are invalid, has to be exactly two!")
    
    X = adata[:,gatechannels].layers["raw"].T
    if robust and very_robust:
        params = dict(final_density_cutoff=.2, bound_density_mad_factor=10.,
                      trim_beadmode_min_clipfactor=.25, trim_beadmode_max_clipfactor=2.,
                      region_density_bw_factor=7, bound_density_bw_factor=6.)
    elif robust:
        params = dict(final_density_cutoff=.2, bound_density_mad_factor=5.,
                      trim_beadmode_min_clipfactor=.25, trim_beadmode_max_clipfactor=2.,
                      region_density_bw_factor=3)
    else:
        params = dict(final_density_cutoff=.2, bound_density_mad_factor=4.)
    params.update(verbose=False, name_x=gatechannels[0], name_y=gatechannels[1], trim_beadmode=True)
    params.update(gating_kwargs) # explicitly passed settings override the defaults above
    asp = AutoSpillTesselationGating(*X, **params)
    if plot or plotpath:
        asp.plot(title=titlefct(adata))
        if plotpath:
            plt.savefig(plotpath, dpi=120)
        if plot: plt.show()
        else:    plt.close()
    adata.obs["keepmask_scatter"] = asp.get_gatemask(*X)
    adata.uns["Nbeads_afterscattergate"] = adata.obs["keepmask_scatter"].sum()
    adata.uns["density_gate_hull"] = asp.densitygate.vertices_hull
    return adata[adata.obs["keepmask_scatter"]].copy() if apply else adata
