"""Shared implementation of the .h5ad writers and readers that keep the cytometer configuration."""
import anndata
from ..utils import CytometerConfiguration


def write_h5ad_with_cytoconfig(adata_, filepath):
    """Write a copy of `adata_` with ``uns['cytoconfig']`` exported to a dictionary (see `write_bead_h5ad`)."""
    adata = adata_.copy()
    adata.uns["cytoconfig"] = adata.uns["cytoconfig"].to_dictionary()
    adata.write(filepath)


def read_h5ad_with_cytoconfig(filepath):
    """Read an .h5ad file and restore ``uns['cytoconfig']`` (see `read_bead_h5ad`)."""
    adata = anndata.read_h5ad(filepath)
    adata.uns["cytoconfig"] = CytometerConfiguration.from_exported(adata.uns["cytoconfig"])
    return adata
