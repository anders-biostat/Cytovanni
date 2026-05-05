import anndata
from ..utils import CytometerConfiguration

def write_bead_h5ad(adata_, filepath):
    """ Write rainbow bead AnnData to .h5ad.
        Export cytometry configuration to make it writable.
    """
    adata = adata_.copy()
    adata.uns["cytoconfig"] = adata.uns["cytoconfig"].to_dictionary()
    adata.write(filepath)

def read_bead_h5ad(filepath):
    """ Read rainbow bead AnnData from .h5ad.
        Process cytometry configuration.
    """
    adata = anndata.read_h5ad(filepath)
    cytoconfig = CytometerConfiguration([])
    cytoconfig.import_exported(adata.uns["cytoconfig"])
    adata.uns["cytoconfig"] = cytoconfig
    return adata