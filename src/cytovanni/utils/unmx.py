"""Linear unmixing of mixed (per-channel) intensities with pandas DataFrames, and the arcsinh transform.

Spectra here have the orientation (stain, channel): one row per stain, one column per detector channel. Inverted
spectra have the orientation (channel, stain). All functions match channels by name, so the column order of the data
does not matter.
"""
import numpy as np
import pandas as pd
import torch

from ..exceptions import SpilloverInversionException


def apply_arcsinh(x, cofactor=1500):
    """Apply the arcsinh transform ``arcsinh(x / cofactor)``.

    Parameters
    ----------
    x : numpy.ndarray or pandas.DataFrame
        Data on a linear scale (raw or calibrated).
    cofactor : float, default 1500
        Cofactor, in the units of `x`; roughly the intensity below which the transform becomes linear.

    Returns
    -------
    numpy.ndarray or pandas.DataFrame
        Transformed data, same type and shape as `x`.
    """
    return np.arcsinh(x / cofactor)


def invert_spectra(spectra, use_pinv=False):
    """Invert single-stain spectra for unmixing.

    For as many channels as stains (conventional flow) the exact inverse is used, for more channels than stains
    (spectral flow) the right pseudo-inverse ``S.T @ inv(S @ S.T)``, i.e. ordinary least squares unmixing.

    Parameters
    ----------
    spectra : pandas.DataFrame
        Single-stain spectra of shape (stain, channel): index stains, columns detector channels.
    use_pinv : bool, default False
        Use ``numpy.linalg.pinv`` (SVD based) instead of the explicit formulas above. It does not raise for fewer
        channels than stains or singular spectra (minimum-norm solution), so check the spectra in that case.

    Returns
    -------
    pandas.DataFrame
        Inverted spectra of shape (channel, stain): index the channels, columns the stains of `spectra`.

    Raises
    ------
    ValueError
        If `spectra` is not a DataFrame.
    SpilloverInversionException
        If not `use_pinv` and there are fewer channels than stains.

    See Also
    --------
    apply_unmixing_inv : apply the result to data.
    apply_unmixing_lstsq : unmixing by a least-squares solve without explicit inversion.
    """
    if not isinstance(spectra, pd.DataFrame):
        raise ValueError(f"Pass spectra as pandas DataFrame! Passed type is {type(spectra)}.")
    np_spectra = spectra.to_numpy()
    if use_pinv:
        inv_spectra = np.linalg.pinv(np_spectra)
    else:
        if np_spectra.shape[-1]==np_spectra.shape[-2]: # normal flow, square spillover
            inv_spectra = np.linalg.inv(np_spectra)
        elif np_spectra.shape[-1]>np_spectra.shape[-2]: # spectral flow, more channels than stains
            # get inverse of S as S.T (S S.T)^(-1)
            inv_spectra = np_spectra.T @ np.linalg.inv(np_spectra @ np_spectra.T)
        else: # not invertible
            raise SpilloverInversionException(
                f"Spillover matrix with {np_spectra.shape[-2]} stains and {np_spectra.shape[-1]} channels is not invertible! Needs more channels than stains.")
    return pd.DataFrame(inv_spectra, index=spectra.columns, columns=spectra.index)


def apply_unmixing_inv(inv_spectra, x):
    """Unmix data with inverted spectra: ``x[channels] @ inv_spectra``.

    Both inputs must be DataFrames so that the channels are matched by name.

    Parameters
    ----------
    inv_spectra : pandas.DataFrame
        Inverted spectra of shape (channel, stain), as returned by `invert_spectra`.
    x : pandas.DataFrame
        Mixed data of shape (events, channel) on a linear scale (raw or calibrated); must contain every channel of
        `inv_spectra` (extra columns are ignored).

    Returns
    -------
    pandas.DataFrame
        Unmixed data of shape (events, stain), same scale as `x`; index of `x`, columns the stains.

    Raises
    ------
    ValueError
        If `inv_spectra` or `x` is not a DataFrame.
    """
    if not isinstance(inv_spectra, pd.DataFrame):
        raise ValueError(f"Pass inverted spectra as pandas DataFrame! Passed type is {type(inv_spectra)}.")
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Pass fluorescence data as pandas DataFrame! Passed type is {type(x)}.")
    return x[inv_spectra.index] @ inv_spectra

def apply_unmixing_lstsq(spectra, x):
    """Unmix data by solving the least-squares problem ``x = u @ spectra`` for the stain intensities ``u``.

    Gives the same result as `invert_spectra` + `apply_unmixing_inv` for well-conditioned spectra, but uses
    ``numpy.linalg.lstsq`` instead of an explicit inverse, which is numerically more stable. Both inputs must be
    DataFrames so that the channels are matched by name.

    Parameters
    ----------
    spectra : pandas.DataFrame
        Single-stain spectra of shape (stain, channel): index stains, columns detector channels.
    x : pandas.DataFrame
        Mixed data of shape (events, channel) on a linear scale (raw or calibrated); must contain every channel of
        `spectra` (extra columns are ignored).

    Returns
    -------
    pandas.DataFrame
        Unmixed data of shape (events, stain), same scale as `x`; index of `x`, columns the stains.

    Raises
    ------
    ValueError
        If `spectra` or `x` is not a DataFrame.
    """
    if not isinstance(spectra, pd.DataFrame):
        raise ValueError(f"Pass spectra as pandas DataFrame! Passed type is {type(spectra)}.")
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Pass fluorescence data as pandas DataFrame! Passed type is {type(x)}.")
    unmx = np.linalg.lstsq(spectra.T, x[spectra.columns].T, rcond=None)[0].T
    return pd.DataFrame(unmx, index=x.index, columns=spectra.index)


def add_adata_unmixed(adata, spectra, addkey="unmx", layer="calibrated", add_arcsinh=True, arcsinh_cofactor=1500):
    """Unmix the events of `adata` with `spectra` and store the result in ``adata.obsm``.

    Uses `invert_spectra` and `apply_unmixing_inv`. Modifies `adata` in place: writes ``adata.obsm[addkey]`` (and
    ``adata.obsm[addkey + '_arcsinh']`` if `add_arcsinh`), overwriting existing entries. Returns None.

    Parameters
    ----------
    adata : anndata.AnnData
        Events; ``adata.var_names`` must contain all channels of `spectra`.
    spectra : pandas.DataFrame
        Single-stain spectra of shape (stain, channel): index stains, columns detector channels; on the same scale as
        `layer` (calibrated spectra for calibrated data).
    addkey : str, default "unmx"
        Key in ``adata.obsm`` for the unmixed data, a DataFrame of shape (events, stain) on the scale of `layer`.
    layer : str, default "calibrated"
        Layer with the mixed per-channel intensities (``"raw"`` or ``"calibrated"``).
    add_arcsinh : bool, default True
        Also store ``arcsinh(unmixed / arcsinh_cofactor)`` in ``adata.obsm[addkey + '_arcsinh']``.
    arcsinh_cofactor : float, default 1500
        Cofactor of the arcsinh transform, in the units of `layer`.

    Raises
    ------
    ValueError
        From `invert_spectra` / `apply_unmixing_inv` if `spectra` is not a DataFrame.
    SpilloverInversionException
        If there are fewer channels than stains.
    """
    x = adata[:,spectra.columns].to_df(layer=layer)
    adata.obsm[addkey] = apply_unmixing_inv(invert_spectra(spectra), x)
    if add_arcsinh:
        adata.obsm[addkey+"_arcsinh"] = apply_arcsinh(adata.obsm[addkey], cofactor=arcsinh_cofactor)


def unmixed_cluster_means(unmx, cluster):
    """Mean of every marker per cluster.

    Parameters
    ----------
    unmx : pandas.DataFrame
        Data of shape (events, stain), e.g. unmixed intensities (any scale; the mean is taken on that scale).
    cluster : array-like
        Cluster label of every event, length events, in the row order of `unmx`.

    Returns
    -------
    pandas.DataFrame
        Mean per cluster, shape (cluster, stain): index the sorted unique cluster labels, columns of `unmx`.
    """
    u_clusters = np.unique(cluster)
    means = pd.DataFrame([unmx.iloc[np.asarray(cluster)==u].mean() for u in u_clusters], index=u_clusters)
    return means
