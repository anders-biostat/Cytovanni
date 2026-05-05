import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .autospill import AutoSpillTesselationGating
from ..exceptions import ScatterGatingChannelException

def gate_scatter_bead_sample(adata, gatechannels=None, plot=False, plotpath="", apply=True, titlefct=lambda ad: ad.uns["name_uid"], robust=False, very_robust=False):
    """ Gate bead sample, using the same approach as AutoSpill.
        
        :param adata: AnnData. The sample.
        
        :param gatechannels: list or None. The two channels to use for the scatter gating, if None defaults to output of get_default_scatter_gate_channels() from cytometer configuration.
        
        :param plot: bool. Whether to show the corresponding summary plot.
        
        :param plotpath: str. Path to save the summary plot to.
        
        :param apply: bool. Whether to apply the gating directly, or only store in 'keepmask_RB_scatter'.
        
        :param titlefct: lambda. Function to get plot title from adata, set arbitrary name by passing (lamba x: "name").
        
        :param robust: bool. Some beads have patterns like two close peaks instead of only one etc., the robust version changes some parameters to accomodate that. However, be cautious as this is noticeably worse at excluding large doublet peaks.

        :param very_robust: bool. Even more tolerant gating, needs robust to also be set.
    """
    if gatechannels is None:
        gatechannels = adata.uns["cytoconfig"].get_default_scatter_gate_channels()
    if not np.all(np.in1d(gatechannels, adata.var.index.to_numpy())):
        raise ScatterGatingChannelException(f"Chosen scatter channels to gate on {gatechannels} are not all available! Choose two of available channels:\n{adata.var.index.to_numpy()}")
    if len(gatechannels)!=2:
        raise ScatterGatingChannelException(f"Chosen scatter channels to gate on {gatechannels} are invalid, has to be exactly two!")
    
    X = adata[:,gatechannels].layers["raw"].T
    if robust:
        if very_robust:
            asp = AutoSpillTesselationGating(*X, verbose=False, final_density_cutoff=.2, bound_density_mad_factor=10.,
                                             trim_beadmode_min_clipfactor=.25, trim_beadmode_max_clipfactor=2.,
                                             region_density_bw_factor=7, bound_density_bw_factor=6.,
                                             name_x=gatechannels[0], name_y=gatechannels[1], trim_beadmode=True)
        else:
            asp = AutoSpillTesselationGating(*X, verbose=False, final_density_cutoff=.2, bound_density_mad_factor=5.,
                                             trim_beadmode_min_clipfactor=.25, trim_beadmode_max_clipfactor=2.,
                                             region_density_bw_factor=3,
                                             name_x=gatechannels[0], name_y=gatechannels[1], trim_beadmode=True)
    else:
        asp = AutoSpillTesselationGating(*X, verbose=False, final_density_cutoff=.2, bound_density_mad_factor=4.,
                                         name_x=gatechannels[0], name_y=gatechannels[1], trim_beadmode=True)
    if plot or plotpath:
        asp.plot(title=titlefct(adata))
        if plotpath:
            plt.savefig(plotpath, dpi=200)
        if plot: plt.show()
        else:    plt.close()
    adata.obs["keepmask_scatter"] = asp.get_gatemask(*X)
    adata.uns["Nbeads_afterscattergate"] = adata.obs["keepmask_scatter"].sum()
    adata.uns["density_gate_hull"] = asp.densitygate.vertices_hull
    return adata[adata.obs["keepmask_scatter"]].copy() if apply else adata
