import anndata
import pandas as pd
import numpy as np

from ..utils import CytometerConfiguration
from .utils import readfcs_sample


def readfcs_rainbow_sample(rbpath, cytoconfig, N_rainbow_peak, name="", rainbow_type="default", rainbow_batch="default", metadct={}, warn_spillover=False):
    """ Read a single rainbow bead sample.
        Based on FlowKit so we don't need to maintain reading code.
        
        :param rbpath: str. Path to .fcs with rainbow bead sample.
        
        :param cytoconfig: CytometerConfiguration. Cytometer configuration for sample.
        
        :param N_rainbow_peak: int. Number of peaks in the rainbow bead sample. Usually between 5 and 8.
        
        :param name: str. Optional name of the sample.
        
        :param rainbow_type: str. Optional type/lot of rainbow bead, if different ones were used.
        
        :param rainbow_batch: str. Optional name of the rainbow bead batch.
        
        :param metadct: dict. Optional additional metadata to add to .uns.
        
        :param warn_spillover: bool. If True, throws a warning if rainbow bead sample contains a non-trivial spillover matrix.
    """
    adata = readfcs_sample(rbpath, cytoconfig, metadct=metadct, warn_spillover=warn_spillover,
                warn_spillover_message="Rainbow bead sample contains a non-trivial spillover matrix! Make sure you are not accidentally compensating rainbow beads.")
    
    adata.uns["name"] = name
    adata.uns["name_uid"] = f"{adata.uns['name']} ({adata.uns['uid']})"
    
    adata.uns["N_rainbow_peak"] = N_rainbow_peak
    adata.uns["rainbow_type"] = rainbow_type
    adata.uns["rainbow_batch"] = rainbow_batch
    
    return adata


def write_rainbow_h5ad(adata_, filepath):
    """ Write rainbow bead AnnData to .h5ad.
        Export cytometry configuration to make it writable.
    """
    adata = adata_.copy()
    adata.uns["cytoconfig"] = adata.uns["cytoconfig"].to_dictionary()
    adata.write(filepath)

def read_rainbow_h5ad(filepath):
    """ Read rainbow bead AnnData from .h5ad.
        Process cytometry configuration.
    """
    adata = anndata.read_h5ad(filepath)
    cytoconfig = CytometerConfiguration([])
    cytoconfig.import_exported(adata.uns["cytoconfig"])
    adata.uns["cytoconfig"] = cytoconfig
    return adata
