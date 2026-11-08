"""Helpers for rainbow calibration results: apply factors to spectra, split log factors into laser and channel parts."""
import numpy as np
import pandas as pd
import warnings

from ..exceptions import CalibrationModuleWarning, CalibrationModuleException, SpilloverWarning

def scale_spectra_wfactors(spectra, factors, spectral=False):
    """Multiply spectra by per-channel calibration factors and normalise every spectrum again.

    Parameters
    ----------
    spectra : pandas.DataFrame
        Single-stain spectra of shape (channel, dye) (index channels, columns dyes), on the scale the factors apply to
        (e.g. raw).
    factors : pandas.Series
        Factor per channel; must contain every channel of `spectra` (extra entries are ignored), e.g. from
        `RainbowCalibrator.get_calibration_factor`. Use ``1 / factors`` to go from calibrated back to raw.
    spectral : bool, default False
        Normalisation after scaling. True (spectral flow): the largest entry of every spectrum is 1 (NaN ignored).
        False (conventional flow): the entry that is exactly 1 in the input stays 1; spectra with no or several
        entries equal to 1 (or all NaN) fall back to the spectral normalisation, with a `SpilloverWarning`.

    Returns
    -------
    pandas.DataFrame
        Scaled spectra, shape (channel, dye), float.

    Raises
    ------
    CalibrationModuleException
        If `factors` lacks a channel of `spectra` or contains NaN for one.

    Examples
    --------
    >>> calibrated = scale_spectra_wfactors(spectra_raw, calibrator.get_calibration_factor(rainbow_batch="b1"),
    ...                                     spectral=True)
    """
    spectra = spectra.astype(float)
    factors = factors.astype(float)
    if len(set(spectra.index) & set(factors.index)) != spectra.shape[0]:
        raise CalibrationModuleException(f"Cannot apply calibration to spectra, missing factors for channels {list(set(spectra.index) - set(factors.index))}.")

    apply_factors = factors.loc[spectra.index]
    if np.isnan(apply_factors).sum()>0:
        raise CalibrationModuleException(f"Cannot apply calibration to spectra, the calibration factors contain nan.")
    
    #negativecount = (spectra<0).sum()
    #if negativecount.sum()>0:
    #    warnings.warn(f"Spectra of {negativecount[negativecount>0].index.tolist()} contain negative entries!", SpilloverWarning)
    
    spectra_scaled = spectra * apply_factors.to_numpy()[:,None]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        if spectral:
            spectra_scaled = spectra_scaled / np.nanmax(spectra_scaled, axis=0)[None]
        else:
            normpositioncount = (spectra==1.).sum() + np.all(np.isnan(spectra), axis=0).astype(int)
            if normpositioncount.max()>1:
                warnings.warn(f"Spectra of {normpositioncount[normpositioncount>1].index.tolist()} have ambiguous normalisation, falling back to spectral convention!", SpilloverWarning)
            if normpositioncount.min()<1:
                warnings.warn(f"Spectra of {normpositioncount[normpositioncount<1].index.tolist()} have invalid normalisation, falling back to spectral convention!", SpilloverWarning)

            for col in spectra_scaled.columns:
                if ((spectra[col]==1.).sum()>1) or ((spectra[col]==1.).sum()<1):
                    spectra_scaled[col] = spectra_scaled[col]/np.nanmax(spectra_scaled[col])
                if (spectra[col]==1.).sum()==1:
                    spectra_scaled[col] = spectra_scaled[col]/spectra_scaled[col].to_numpy()[np.argmax(spectra[col]==1.)]
    
    return spectra_scaled

def extract_logfactor_laserfactor(logfactors, fixed_components=True):
    """Split log calibration factors into a common factor per batch and per-channel residuals.

    Decomposes ``logfactors[batch, channel] = common[batch] * component[channel] + residuals[batch, channel]``. Meant
    for the channels of one laser: a change of the laser scales all of them approximately equally, so the common
    factor approximately captures the laser and the residuals the individual detectors.

    Parameters
    ----------
    logfactors : pandas.DataFrame
        Log calibration factors, shape (batch, channel), e.g. ``numpy.log(calibrator.get_calibration_factors_all())``
        restricted to the channels of one laser. Batches with a NaN in any channel are left out (NaN in the output).
    fixed_components : bool, default True
        If True, the component is 1 for all channels, so the common factor is the mean over channels of every batch
        (least squares). If False, the component is the first principal component over batches, rescaled to mean 1
        over the channels (which also fixes its sign), and the common factor is the mean over channels of the PCA
        reconstruction (offset included); for a constant component both are identical.

    Returns
    -------
    common_logfactor : pandas.Series
        Common log factor per batch (index of `logfactors`).
    residuals : pandas.DataFrame
        Residuals, same shape and labels as `logfactors`.

    Raises
    ------
    CalibrationModuleException
        If not `fixed_components` and the first principal component has no common direction (mean over channels
        near 0).

    See Also
    --------
    extract_logfactor_laserfactor_bylaser : the same for every laser separately.
    """
    mask_nan = logfactors.isna().any(axis=1)

    if fixed_components:
        x = logfactors[~mask_nan].to_numpy()
        Y, res, _, _ = np.linalg.lstsq(np.ones((x.shape[1], 1)), x.T, rcond=-1)
        common_logfactor_ = pd.Series(Y[0], index=logfactors.index[~mask_nan])
        residuals_ = logfactors[~mask_nan] - Y.T
    else:
        from sklearn.decomposition import PCA
        pca = PCA(1)
        Y = pca.fit_transform(logfactors[~mask_nan])
        component_mean = pca.components_.mean()
        if np.abs(component_mean) < 1e-3*np.abs(pca.components_).max():
            raise CalibrationModuleException("First principal component of the logfactors has no common direction across channels, cannot extract a common factor!")
        component = pca.components_[0] / component_mean
        common = pca.mean_.mean() + Y[:,0] * component_mean
        common_logfactor_ = pd.Series(common, index=logfactors.index[~mask_nan])
        residuals_ = logfactors[~mask_nan] - common[:,None] * component[None]

    common_logfactor = pd.Series(np.full((logfactors.shape[0],), np.nan), index=logfactors.index)
    common_logfactor.loc[common_logfactor_.index] = common_logfactor_

    residuals = pd.DataFrame(np.full(logfactors.shape, np.nan), index=logfactors.index, columns=logfactors.columns)
    residuals.loc[residuals_.index] = residuals_
    
    return common_logfactor, residuals

def extract_logfactor_laserfactor_bylaser(logfactors, key_laser, fixed_components=True):
    """Run `extract_logfactor_laserfactor` separately for the channels of every laser.

    Parameters
    ----------
    logfactors : pandas.DataFrame
        Log calibration factors, shape (batch, channel).
    key_laser : numpy.ndarray or pandas.Series
        Laser of every column of `logfactors`, in column order (e.g.
        ``cytoconfig.channel_laser_key.loc[logfactors.columns]``).
    fixed_components : bool, default True
        See `extract_logfactor_laserfactor`.

    Returns
    -------
    residuals : pandas.DataFrame
        Residuals, shape (batch, channel), columns grouped by laser (sorted laser names). Channels that are NaN in all
        batches stay NaN.
    laserfactors : pandas.DataFrame
        Common log factor per batch and laser, shape (batch, laser); NaN for a laser without any non-NaN channel.

    Notes
    -----
    The order of the outputs (residuals first) is the reverse of `extract_logfactor_laserfactor`.
    """
    lasers = np.unique(key_laser)
    def safe_extract(logfactors):
        keys_missing = logfactors.columns[logfactors.isna().all(axis=0)]
        keys_present = logfactors.columns[~np.isin(logfactors.columns, keys_missing)]
        common_logfactor, residuals = extract_logfactor_laserfactor(logfactors[keys_present], fixed_components=fixed_components)
        residuals[keys_missing] = np.nan
        if len(keys_present)==0:
            common_logfactor.iloc[:] = np.nan
        return common_logfactor, residuals
    extracted = [safe_extract(logfactors.loc[:,key_laser==l]) for l in lasers]
    residuals = pd.concat([e[1] for e in extracted], axis=1)
    laserfactors = pd.concat([e[0] for e in extracted], axis=1)
    laserfactors.columns = lasers
    return residuals, laserfactors

def reorder_peaks(adata, new_order):
    """Change the peak numbering of a rainbow bead measurement after `fit_rainbow_GMM`, in place.

    Parameters
    ----------
    adata : anndata.AnnData
        Measurement after `fit_rainbow_GMM`. Changed in place: ``obs['GMM_label']`` and the rows of ``uns``
        'GMM_means', 'GMM_asinh_means', 'GMM_asinh_covariances' and 'GMM_asinh_stds' are reordered.
    new_order : pandas.Series
        Index the old label, value the new label (a permutation of 0..N-1), e.g.
        ``pd.Series(np.arange(5)[::-1], index=np.arange(5))`` reverses 5 peaks.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `new_order` does not have as many entries as there are labels in ``obs['GMM_label']``.
    """
    if not len(adata.obs["GMM_label"].unique())==len(new_order):
        raise ValueError("Make sure the new order has as many classes as the old one!")
    
    adata.obs["GMM_label"] = new_order.loc[adata.obs["GMM_label"]].to_numpy()

    idx = new_order.sort_values().index.to_numpy()
    adata.uns["GMM_asinh_covariances"] = adata.uns["GMM_asinh_covariances"][idx]
    adata.uns["GMM_asinh_means"] = adata.uns["GMM_asinh_means"][idx]
    adata.uns["GMM_asinh_stds"] = adata.uns["GMM_asinh_stds"][idx]
    adata.uns["GMM_means"] = adata.uns["GMM_means"][idx]

    