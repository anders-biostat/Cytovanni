import numpy as np
import pandas as pd
import warnings

from ..exceptions import CalibrationModuleWarning, CalibrationModuleException

def scale_spectra_wfactors(spectra, factors, spectral=False):
    """ Scale spectra with calibration factors.
        If spectral flow, norm largest entry of every spectrum to one.
        For conventional flow, try keeping the entry that is one normed to one,
        use spectral convention as fallback if multiple entries are exactly one.

        :param spectra: pd.DataFrame. Single stain spectra, index channel, column dye.

        :param factors: pd.Series. Scaling factor for every channel with corresponding index.
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
    """ Take fitted logfactors for every channel, pd.DataFrame of shape (batch, channel).
        
        If not fixed_components, fits a common factor principal component;
        the common factor is normalized such that the slope of common factor
        and the individual channel factors should be approximately one.
        
        If fixed_components, fits a least-squares factor where the component
        is simply set to one for all channels.
        
        Returns common factor and residuals
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
        common_logfactor_ = pd.Series((Y * np.abs(pca.components_).mean())[:,0], index=logfactors.index[~mask_nan])
        residuals_ = logfactors[~mask_nan] - Y @ pca.components_

    common_logfactor = pd.Series(np.full((logfactors.shape[0],), np.nan), index=logfactors.index)
    common_logfactor.loc[common_logfactor_.index] = common_logfactor_

    residuals = pd.DataFrame(np.full(logfactors.shape, np.nan), index=logfactors.index, columns=logfactors.columns)
    residuals.loc[residuals_.index] = residuals_
    
    return common_logfactor, residuals

def extract_logfactor_laserfactor_bylaser(logfactors, key_laser, fixed_components=True):
    """ Same as extract_logfactor_laserfactor, but run separately for every laser according to key_laser.
        key_laser should correspond to the columns of logfactors.
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
    """ Reorder peaks in rainbow bead measurement adata.

        new_order should be the old order as index, new order as value,
        e.g. pd.Series(np.arange(5)[::-1], index=np.arange(5)) flips the order.
        Applies this not only to GMM_label, but also the various entries in adata.uns
    """
    if not len(adata.obs["GMM_label"].unique())==len(new_order):
        raise ValueError("Make sure the new order has as many classes as the old one!")
    
    adata.obs["GMM_label"] = new_order.loc[adata.obs["GMM_label"]].to_numpy()

    idx = new_order.sort_values().index.to_numpy()
    adata.uns["GMM_asinh_covariances"] = adata.uns["GMM_asinh_covariances"][idx]
    adata.uns["GMM_asinh_means"] = adata.uns["GMM_asinh_means"][idx]
    adata.uns["GMM_asinh_stds"] = adata.uns["GMM_asinh_stds"][idx]
    adata.uns["GMM_means"] = adata.uns["GMM_means"][idx]

    