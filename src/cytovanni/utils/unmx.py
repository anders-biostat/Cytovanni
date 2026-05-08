import numpy as np
import pandas as pd
import torch

from ..exceptions import SpilloverInversionException


def apply_arcsinh(x, cofactor=1500):
    """ Apply ArcSinh with cofactor to data x.
    """
    return np.arcsinh(x / cofactor)


def invert_spectra(spectra, use_pinv=False):
    """ Invert spectra DataFrame of shape (stain, channel)
        returns (channel, stain)
        
        :param spectra: pd.DataFrame. Single stain spectra, index is stain, column cytometer channel.
        
        :param use_pinv: bool. Whether to use np.linalg.pinv or invert by hand.
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
    """ Apply unmixing to x, shape (event, channel), using the inverted spectra, shape (channel, stain) directly.
        returns (event, stain)
        
        Enforces using DataFrames to make sure all channels line up properly.
        
        :param inv_spectra: pd.DataFrame. Inverted spectra as produced by invert_spectra.
        
        :param x: pd.DataFrame. DataFrame where the columns match the channel names in the spectra, and the index labels different events.
    """
    if not isinstance(inv_spectra, pd.DataFrame):
        raise ValueError(f"Pass spectra as pandas DataFrame! Passed type is {type(spectra)}.")
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Pass fluorescence data as pandas DataFrame! Passed type is {type(x)}.")
    return x[inv_spectra.index] @ inv_spectra

def apply_unmixing_lstsq(spectra, x):
    """ Apply unmixing to x, shape (event, channel), using the spectra, shape (stain, channel) and solving a least squares problem.
        More stable than explicit inversion.
        returns (event, stain)
        
        Enforces using DataFrames to make sure all channels line up properly.
        
        :param spectra: pd.DataFrame. Single stain spectra, index is stain, column cytometer channel.
        
        :param x: pd.DataFrame. DataFrame where the columns match the channel names in the spectra, and the index labels different events.
    """
    if not isinstance(spectra, pd.DataFrame):
        raise ValueError(f"Pass spectra as pandas DataFrame! Passed type is {type(spectra)}.")
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"Pass fluorescence data as pandas DataFrame! Passed type is {type(x)}.")
    unmx = np.linalg.lstsq(spectra.T, x[spectra.columns].T, rcond=None)[0].T
    return pd.DataFrame(unmx, index=x.index, columns=spectra.index)


def add_adata_unmixed(adata, spectra, addkey="unmx", layer="raw", add_arcsinh=True, arcsinh_cofactor=1500):
    """ Add unmixed data to adata.obsm, using single stains in spectra.
        Mixed fluorescence intensity is taken from layer.
        
        :param adata: AnnData. Contains the event data.
        
        :param spectra: pd.DataFrame. Single stain spectra, index is stain, column cytometer channel.
        
        :param addkey: str. Unmixed data is added to adata.obsm[addkey].
        
        :param layer: str. Raw data is taken from adata.layer[layer].
        
        :param add_arcsinh: bool. If True, also add arcsinh transformed data to adata.obsm['addkey'_arcsinh].
        
        :param arcsinh_cofactor: float. Cofactor for ArcSinh.
    """
    x = adata[:,spectra.columns].to_df(layer="calibrated")
    adata.obsm[addkey] = apply_unmixing_inv(invert_spectra(spectra), x)
    if add_arcsinh:
        adata.obsm[addkey+"_arcsinh"] = apply_arcsinh(adata.obsm[addkey], cofactor=arcsinh_cofactor)


def unmixed_cluster_means(unmx, cluster):
    """ Get mean of unmixed data across clusters.
        
        :param unmx: pd.DataFrame. Unmixed data, index events, columns markers.
        
        :param cluster: iterable. Cluster assignment
    """
    u_clusters = np.unique(cluster)
    means = pd.DataFrame([unmx.iloc[np.asarray(cluster)==u].mean() for u in u_clusters], index=u_clusters)
    return means
