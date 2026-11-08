"""Read rainbow bead samples from .fcs, and write/read them as .h5ad including the cytometer configuration."""
import anndata
import pandas as pd
import numpy as np

from ..utils import CytometerConfiguration
from .utils import readfcs_sample
from .h5ad import write_h5ad_with_cytoconfig, read_h5ad_with_cytoconfig


def readfcs_rainbow_sample(rbpath, cytoconfig, N_rainbow_peak, name="", rainbow_type="default", rainbow_batch="default", metadct=None, warn_spillover=False):
    """Read a single rainbow bead sample from an .fcs file into an AnnData object.

    Reads the file with `readfcs_sample` and adds the rainbow metadata that `cytovanni.rainbow.RainbowCalibrator`
    needs.

    Parameters
    ----------
    rbpath : str
        Path to the .fcs file of the rainbow bead measurement.
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Cytometer configuration of the measurement.
    N_rainbow_peak : int
        Number of peaks (bead populations) of the rainbow beads, usually between 5 and 8. Stored in
        ``uns['N_rainbow_peak']`` for the GMM fit (`cytovanni.rainbow.fit_rainbow_GMM`).
    name : str, default ""
        Name of the sample, stored in ``uns['name']`` and used in ``uns['name_uid']``.
    rainbow_type : str, default "default"
        Type of the rainbow beads (bead type and lot), if different ones were used; the bead peaks are only assumed
        to be equivalent within a type.
    rainbow_batch : str, default "default"
        Name of the rainbow bead batch, i.e. the group of measurements that share one calibration (one set of
        cytometer settings).
    metadct : dict, optional
        Additional metadata, added to ``uns``.
    warn_spillover : bool, default False
        Warn (`RainbowSpilloverWarning`) if the file contains a non-trivial spillover matrix; rainbow beads should
        not be compensated.

    Returns
    -------
    anndata.AnnData
        As returned by `readfcs_sample` (``layers['raw']``, ``var``, ``uns`` entries listed there), plus ``uns``
        entries 'name', 'name_uid' (``"<name> (<uid>)"``), 'N_rainbow_peak', 'rainbow_type' and 'rainbow_batch'.

    Raises
    ------
    ChannelMismatchException
        If channels of `cytoconfig` are missing in the file (see `readfcs_sample`).

    Examples
    --------
    >>> adata = readfcs_rainbow_sample(path, cytoconfig, N_rainbow_peak=8, name="day1",
    ...                                rainbow_type="lot_A", rainbow_batch="day1")
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
    """Write a rainbow bead AnnData to .h5ad.

    The `cytovanni.utils.CytometerConfiguration` in ``adata_.uns['cytoconfig']`` is exported to a dictionary
    (``to_dictionary``) so that the file can be written; `adata_` itself is not changed (a copy is written).

    Parameters
    ----------
    adata_ : anndata.AnnData
        Rainbow bead sample with ``uns['cytoconfig']``, e.g. as read by `readfcs_rainbow_sample`.
    filepath : str
        Path of the .h5ad file (the folder must exist).

    See Also
    --------
    read_rainbow_h5ad : the inverse.
    """
    write_h5ad_with_cytoconfig(adata_, filepath)

def read_rainbow_h5ad(filepath):
    """Read a rainbow bead AnnData written by `write_rainbow_h5ad`.

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
    return read_h5ad_with_cytoconfig(filepath)
