"""Write and read bead AnnData objects (e.g. single-stain controls) as .h5ad, including the cytometer configuration."""
import anndata
from ..utils import CytometerConfiguration

def write_bead_h5ad(adata_, filepath):
    """Write a bead AnnData (e.g. a single-stain control) to .h5ad.

    The `cytovanni.utils.CytometerConfiguration` in ``adata_.uns['cytoconfig']`` is exported to a dictionary
    (``to_dictionary``) so that the file can be written; `adata_` itself is not changed (a copy is written).

    Parameters
    ----------
    adata_ : anndata.AnnData
        Bead sample with ``uns['cytoconfig']``, e.g. as read by `cytovanni.io.readfcs_sample`.
    filepath : str
        Path of the .h5ad file (the folder must exist).

    See Also
    --------
    read_bead_h5ad : the inverse.
    """
    adata = adata_.copy()
    adata.uns["cytoconfig"] = adata.uns["cytoconfig"].to_dictionary()
    adata.write(filepath)

def read_bead_h5ad(filepath):
    """Read a bead AnnData written by `write_bead_h5ad`.

    Parameters
    ----------
    filepath : str
        Path of the .h5ad file.

    Returns
    -------
    anndata.AnnData
        The sample, with ``uns['cytoconfig']`` restored as a `cytovanni.utils.CytometerConfiguration` (of the exported
        subclass).

    Raises
    ------
    CytometerConfigurationException
        If the exported configuration class or its settings-hash function is not known (see
        `cytovanni.utils.register_hash_function`).
    """
    adata = anndata.read_h5ad(filepath)
    adata.uns["cytoconfig"] = CytometerConfiguration.from_exported(adata.uns["cytoconfig"])
    return adata