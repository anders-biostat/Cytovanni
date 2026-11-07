"""Single-stain spectra of one measurement (`SpilloverMeasurement`) and of many measurements (`SpilloverCollection`).

Spectra of a measurement are pandas.DataFrame of shape (channel, dye): index the fluorescence channels, columns the
dyes. The spectra collected per dye across measurements are of shape (measurement, channel).
"""
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from ..exceptions import SpilloverWarning, SpilloverException, NegativeSpilloverWarning
from ..io import read_bead_h5ad
from ..rainbow import scale_spectra_wfactors
from ..utils import get_cmap, cmap_to_legendhandles
from .pca import SingleSpectrumPCAFitter

def _find_smallest_h5ad(folder):
    """Path of the smallest .h5ad file in `folder` (a quick file to read the measurement metadata from)."""
    files = list(filter(lambda x: x.endswith(".h5ad"), os.listdir(folder)))
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(folder, f)))
    return os.path.join(folder, files[0])

def _ensure_limit_channels(matrix, channels):
    """Restrict the rows of `matrix` to `channels`, in that order; missing channels are filled with NaN."""
    matrix = matrix.loc[matrix.index.intersection(channels)]
    for row in pd.Index(channels).difference(matrix.index):
        matrix.loc[row] = np.nan
    matrix = matrix.loc[channels]
    return matrix
def _read_matrix_custom(filepath, channels):
    """Read a spectra .csv of `collect_single_fits` as (channel, dye), rows limited to `channels`, dyes sorted."""
    matrix = pd.read_table(filepath, sep=",", index_col=0)
    matrix = _ensure_limit_channels(matrix, channels)
    matrix = matrix[sorted(matrix.columns)]
    return matrix
def _read_matrix_autospill(filepath, channels):
    """Read an AutoSpill spillover .csv as (channel, dye): transposed, ``-A`` removed from the dyes, dyes sorted."""
    matrix = pd.read_table(filepath, sep=",", index_col=0).T
    matrix.columns = [c[:-2] if c.endswith("-A") else c for c in matrix.columns]
    matrix = _ensure_limit_channels(matrix, channels)
    matrix = matrix[sorted(matrix.columns)]
    return matrix

class SpilloverMeasurement():
    """Single-stain spectra of one measurement, raw and calibrated to common units.

    Holds one or several spectra matrices of the same single stains (one per fit method), the metadata of the dyes,
    and their calibrated versions. Usually created with one of the constructors `from_custom` (the package's own
    single-stain fits), `from_AutoSpill` or `from_csv`; the constructor itself only sets up an empty object.

    Parameters
    ----------
    rbint : cytovanni.rainbow.RainbowCalibrator, optional
        Rainbow calibration; its fluorescence channels are used, and it provides the calibration factor. Required if
        `rainbow_batch` or `rainbow_batch_adata` is given.
    is_spectral : bool, default False
        Normalisation convention for the calibrated spectra: spectral flow (largest entry 1) or conventional flow
        (the entry that is 1, i.e. the assigned channel, stays 1); see `cytovanni.rainbow.scale_spectra_wfactors`.
    rainbow_batch : str, optional
        Rainbow batch of the single-stain recording, for the calibration factor.
    rainbow_batch_adata : anndata.AnnData, optional
        Measurement from which the rainbow batch is inferred instead (see
        `cytovanni.rainbow.RainbowCalibrator.get_calibration_factor`).
    channels : list of str, optional
        Fluorescence channels; required if `rbint` is not given.

    Raises
    ------
    ValueError
        If neither `rbint` nor `channels` is given.

    Attributes
    ----------
    channels : list of str
        Fluorescence channels (rows of the spectra).
    spectra : dict of str to pandas.DataFrame
        Spectra per fit method, each of shape (channel, dye), on the raw scale of this measurement. Keys:
        'linear', 'linearos', 'posneg', 'torchos' (`from_custom`), 'autospill', 'autospill_000', 'autospill_posneg'
        (AutoSpill results) or 'spectra' (`from_csv`).
    spectra_calibrated : dict of str to pandas.DataFrame
        The same spectra calibrated with ``calibration_factor`` and normalised again (empty for `from_csv`).
    calibration_factor : pandas.Series
        Rainbow calibration factor per channel (only set if a rainbow batch or adata is given).
    panel_meta : pandas.DataFrame
        Metadata per dye, index the dye names (columns of the spectra), e.g. ``channel``, ``antigen``,
        ``set_hash`` and further columns added by the user (``fluorophore``, ``is_bad``, ``positive_peak``, ...).
    """
    def __init__(self, rbint=None, is_spectral=False, rainbow_batch=None, rainbow_batch_adata=None, channels=None):
        self.rbint = rbint
        if (rbint is not None) + (channels is not None) ==0:
            raise ValueError("Needs either rbint or channels to be set!")
        self.channels = rbint.rfim.channels if self.rbint is not None else channels
        self.spectra = {}
        self.spectra_calibrated = {}
        self.is_spectral = is_spectral
        
        # calibration factors
        if (rainbow_batch is not None) + (rainbow_batch_adata is not None) >0:
            self.calibration_factor = rbint.get_calibration_factor(rainbow_batch=rainbow_batch, adata=rainbow_batch_adata)
    
    
    @classmethod
    def from_custom(cls, rbint, custom_data_folder, rainbow_batch=None, abort_inconsistent=True, asp_folder=None, is_spectral=False, clip_negative=True):
        """Load the spectra of the package's own single-stain fits of one measurement.

        Expects `custom_data_folder` as produced by `collect_bead_controls_regularflow` (or
        `collect_bead_controls_spectralflow`), the single-stain fitters and `collect_single_fits`:

        - ``panel_meta.csv``: metadata with columns ``dye`` and ``set_hash`` (plus ``datafile``, which is dropped);
        - ``spectra_linear.csv``, ``spectra_linearos.csv``, ``spectra_posneg.csv``, ``spectra_torchos.csv``: spectra of
          the corresponding fit methods, shape (channel, dye); every file present is loaded;
        - ``data/*.h5ad``: the gated single stains; the smallest is read to infer the rainbow batch if
          `rainbow_batch` is not given.

        Checks that all single stains were recorded at the same cytometer settings (``set_hash``), checks that every dye
        has metadata, optionally clips negative entries, and adds the calibrated spectra.

        Parameters
        ----------
        rbint : cytovanni.rainbow.RainbowCalibrator
            Rainbow calibration that covers the single-stain recording.
        custom_data_folder : str
            Folder of the measurement (see above).
        rainbow_batch : str, optional
            Rainbow batch of the single-stain recording. Default: inferred from the smallest .h5ad file in
            ``custom_data_folder/data`` (``uns['rainbow_batch']``, else the closest rainbow batch by date, time,
            cytometer and settings).
        abort_inconsistent : bool, default True
            Raise if the single stains were recorded at different settings; otherwise only warn.
        asp_folder : str, optional
            Folder with AutoSpill results to load in addition (see `try_load_AutoSpill_spectra`).
        is_spectral : bool, default False
            Normalisation convention for the calibrated spectra (spectral or conventional flow).
        clip_negative : bool, default True
            Set negative spectrum entries to 0 (before calibration).

        Returns
        -------
        SpilloverMeasurement

        Raises
        ------
        SpilloverException
            If the settings are inconsistent and `abort_inconsistent`, or a dye in the spectra has no metadata.

        Warns
        -----
        SpilloverWarning
            If the settings are inconsistent and not `abort_inconsistent`, or no spectra file is found.
        NegativeSpilloverWarning
            If negative entries are clipped.
        """
        adata = read_bead_h5ad(_find_smallest_h5ad(os.path.join(custom_data_folder, "data"))) if (rainbow_batch is None) else None
        spm = cls(rbint, is_spectral=is_spectral, rainbow_batch=rainbow_batch, rainbow_batch_adata=adata)
        spm.custom_data_folder = custom_data_folder
        
        spm.panel_meta = pd.read_table(os.path.join(spm.custom_data_folder, "panel_meta.csv"), sep=",", index_col=0)
        spm.panel_meta = spm.panel_meta.drop(columns=["datafile"])
        spm.panel_meta.index = spm.panel_meta["dye"].tolist()

        # ensure consistent instrument settings
        set_hash = spm.panel_meta["set_hash"]
        if len(np.unique(set_hash[set_hash==set_hash]))>1:
            if abort_inconsistent:
                raise SpilloverException(f"Found inconsistent settings in single stains in {custom_data_folder}!")
            else:
                warnings.warn(f"Found inconsistent settings in single stains in {custom_data_folder}!", SpilloverWarning)
        
        # read spillovers
        spm.try_load_custom_spectra()
        if asp_folder is not None:
            spm.asp_folder = asp_folder
            spm.try_load_AutoSpill_spectra()
        spm.check_missing_metadata()
        if clip_negative: spm.clip_negative_spectra()
        spm.add_calibrated_spectra()
        
        return spm
    
    @classmethod
    def from_AutoSpill(cls, rbint, asp_folder, rainbow_batch, is_spectral=False, clip_negative=True):
        """Load the spectra fitted by AutoSpill for one measurement.

        Reads the metadata from ``{asp_folder}/fcs_control.csv`` (as made by `make_AutoSpill_fcs_config`; dye names
        without ``-A``) and the spillover tables in ``{asp_folder}/table_spillover/`` (see
        `try_load_AutoSpill_spectra`), checks that every dye has metadata, optionally clips negative entries, and adds
        the calibrated spectra. Channels that were padded for AutoSpill (antigen ``'_dummy_pad'``) are loaded like the
        others.

        Parameters
        ----------
        rbint : cytovanni.rainbow.RainbowCalibrator
            Rainbow calibration that covers the single-stain recording.
        asp_folder : str
            Folder where AutoSpill stored its results.
        rainbow_batch : str
            Rainbow batch of the single-stain recording.
        is_spectral : bool, default False
            Normalisation convention for the calibrated spectra (spectral or conventional flow).
        clip_negative : bool, default True
            Set negative spectrum entries to 0 (before calibration).

        Returns
        -------
        SpilloverMeasurement

        Raises
        ------
        SpilloverException
            If a dye in the spectra has no metadata.
        """
        spm = cls(rbint, is_spectral=is_spectral, rainbow_batch=rainbow_batch)
        spm.asp_folder = asp_folder
        
        spm.panel_meta = pd.read_table(os.path.join(asp_folder, "fcs_control.csv"), sep=",", index_col=0)
        spm.panel_meta = spm.panel_meta.drop(columns=["filename","wavelength"])
        spm.panel_meta.index = spm.panel_meta["dye"].apply(lambda x: x[:-2] if x.endswith("-A") else x).tolist()
        
        # read spillovers
        spm.try_load_AutoSpill_spectra()
        spm.check_missing_metadata()
        if clip_negative: spm.clip_negative_spectra()
        spm.add_calibrated_spectra()
                
        return spm

    @classmethod
    def from_csv(cls, file_spectra, file_meta, is_spectral=False, clip_negative=True):
        """Load already fitted spectra and their metadata from two .csv files, without calibration.

        The spectra are stored under the key ``'spectra'``; their rows define the channels. No calibrated spectra are
        added and the metadata is not checked against the spectra.

        Parameters
        ----------
        file_spectra : str
            Path to the .csv with the spectra, shape (channel, dye): index the channels, columns the dyes.
        file_meta : str
            Path to the .csv with the dye metadata, index the dye names (as the columns of the spectra).
        is_spectral : bool, default False
            Normalisation convention (spectral or conventional flow); stored, only used for calibration.
        clip_negative : bool, default True
            Set negative spectrum entries to 0.

        Returns
        -------
        SpilloverMeasurement

        Examples
        --------
        >>> spm = SpilloverMeasurement.from_csv("run1/spectra_calibrated.csv", "run1/panel_meta.csv")
        """
        spectra = pd.read_table(file_spectra, sep=",", index_col=0)
    
        spm = cls(is_spectral=is_spectral, channels=spectra.index)
        
        spm.panel_meta = pd.read_table(file_meta, sep=",", index_col=0)
        spm.spectra["spectra"] = spectra
        if clip_negative: spm.clip_negative_spectra()
        
        return spm
    
    
    def try_load_AutoSpill_spectra(self, warn_empty=True):
        """Load the AutoSpill spillover tables found in ``{self.asp_folder}/table_spillover/`` into ``self.spectra``.

        Loads ``autospill_spillover.csv`` as 'autospill', ``autospill_spillover_000.csv`` as 'autospill_000' and
        ``posnegpop_spillover.csv`` as 'autospill_posneg', if present.

        Parameters
        ----------
        warn_empty : bool, default True
            Warn (`SpilloverWarning`) if none of the files is found.
        """
        try_loading = [("autospill_spillover.csv", "autospill"), ("autospill_spillover_000.csv", "autospill_000"), ("posnegpop_spillover.csv", "autospill_posneg")]
        found = 0
        if self.asp_folder is not None:
            for file, key in try_loading:
                filepath = os.path.join(self.asp_folder, "table_spillover", file)
                if os.path.exists(filepath):
                    found += 1
                    self.spectra[key] = _read_matrix_autospill(filepath, self.channels)
        if warn_empty and found==0:
            warnings.warn(f"Could not find any AutoSpill fits in folder {self.asp_folder}!", SpilloverWarning)
    
    def try_load_custom_spectra(self, warn_empty=True):
        """Load the spectra files of the package's fits found in ``self.custom_data_folder`` into ``self.spectra``.

        Loads ``spectra_linear.csv``, ``spectra_linearos.csv``, ``spectra_posneg.csv`` and ``spectra_torchos.csv`` as
        'linear', 'linearos', 'posneg' and 'torchos', if present.

        Parameters
        ----------
        warn_empty : bool, default True
            Warn (`SpilloverWarning`) if none of the files is found.
        """
        try_loading = [("spectra_linear.csv", "linear"), ("spectra_linearos.csv", "linearos"), ("spectra_posneg.csv", "posneg"), ("spectra_torchos.csv", "torchos")]
        found = 0
        for file, key in try_loading:
            filepath = os.path.join(self.custom_data_folder, file)
            if os.path.exists(filepath):
                found += 1
                self.spectra[key] = _read_matrix_custom(filepath, self.channels)
        if warn_empty and found==0:
            warnings.warn(f"Could not find any custom fits in folder {self.custom_data_folder}!", SpilloverWarning)
    
    def check_missing_metadata(self):
        """Raise `SpilloverException` if a dye in any of ``self.spectra`` has no row in ``self.panel_meta``."""
        present_dyes = set().union(*[set(s.columns) for s in self.spectra.values()])
        metadata_dyes = set(self.panel_meta.index)
        
        if len(present_dyes | metadata_dyes) > len(metadata_dyes):
            raise SpilloverException(f"Missing metadata for dyes {list(present_dyes - metadata_dyes)}!")
    
    def add_calibrated_spectra(self):
        """Calibrate all spectra with ``self.calibration_factor`` into ``self.spectra_calibrated``.

        Every channel is multiplied by its calibration factor and the spectra are normalised again
        (`cytovanni.rainbow.scale_spectra_wfactors`, convention from ``is_spectral``).
        """
        for key in self.spectra:
            self.spectra_calibrated[key] = scale_spectra_wfactors(self.spectra[key], self.calibration_factor, self.is_spectral)
    
    def clip_negative_spectra(self, warn=True):
        """Set negative entries of all raw and calibrated spectra to 0 (in place).

        Parameters
        ----------
        warn : bool, default True
            Warn (`NegativeSpilloverWarning`) with the number of clipped entries, if any.
        """
        clipped_entries = 0
        n_keys = 0
        
        for key in self.spectra:
            n_keys += 1
            clipped_entries += (self.spectra[key].to_numpy()<0).sum()
            self.spectra[key][self.spectra[key]<0] = 0.
            
        for key in self.spectra_calibrated:
            n_keys += 1
            clipped_entries += (self.spectra_calibrated[key].to_numpy()<0).sum()
            self.spectra_calibrated[key][self.spectra_calibrated[key]<0] = 0.
        
        if warn and clipped_entries>0:
            warnstr = f"Found {clipped_entries} negative entries in spectra across {n_keys} fits! Clipping to zero."
            warnings.warn(warnstr, NegativeSpilloverWarning)
    
    
    def mask_spectra_fromkey(self, key, mask_meta=[]):
        """Set the spectra of the dyes flagged in a metadata column to NaN (in place), e.g. single stains that look bad.

        Parameters
        ----------
        key : str
            Column of ``self.panel_meta``; dyes where it is True are masked in all raw and calibrated spectra. The
            column is converted with ``astype(bool)``, so missing values (NaN) count as True.
        mask_meta : list of str, default []
            Columns of ``self.panel_meta`` that are set to NaN for the flagged dyes as well, e.g. ``["positive_peak"]``.

        Examples
        --------
        >>> spm.mask_spectra_fromkey("is_bad", mask_meta=["positive_peak"])
        """
        for skey in self.spectra:
            mask = self.panel_meta.loc[self.spectra[skey].columns, key].to_numpy().astype(bool)
            self.spectra[skey][self.spectra[skey].columns[mask]] = np.nan
        for skey in self.spectra_calibrated:
            mask = self.panel_meta.loc[self.spectra_calibrated[skey].columns, key].to_numpy().astype(bool)
            self.spectra_calibrated[skey][self.spectra_calibrated[skey].columns[mask]] = np.nan
        
        for mkey in mask_meta:
            mask = self.panel_meta[key].to_numpy().astype(bool)
            self.panel_meta.loc[mask, mkey] = np.nan

def _collect_meta(spillovers, dye, stain, register_on):
    """Metadata rows of `dye` from every measurement where ``panel_meta.loc[dye, register_on] == stain``."""
    index = []
    metas = []
    for name, sp in spillovers.items():
        if (dye in sp.panel_meta.index) and (sp.panel_meta.loc[dye, register_on]==stain):
            index.append(name)
            metas.append(sp.panel_meta.loc[dye])
    return pd.DataFrame(metas, index=index)

def _collect_spectrum(spillovers, dye, stain, register_on, key, calibrated=False):
    """Spectra of `dye` for fit method `key` from every matching measurement, shape (measurement, channel)."""
    index = []
    spectra = []
    for name, sp in spillovers.items():
        # make sure metadata is present
        if dye in sp.panel_meta.index and sp.panel_meta.loc[dye, register_on]==stain:
            spect = (sp.spectra_calibrated if calibrated else sp.spectra)
            # make sure fit type is present, and spectrum for dye is available
            if key in spect and dye in spect[key]:
                index.append(name)
                spectra.append(spect[key][dye])
    return pd.DataFrame(spectra, index=index)

def _normalize_spectra_set(spectra, norm_channel=None):
    """Normalise every spectrum of one dye to 1 in `norm_channel`.

    Parameters
    ----------
    spectra : pandas.DataFrame
        Spectra of one dye, shape (measurement, channel).
    norm_channel : str, optional
        Channel set to 1. Default: the channel with the highest mean over the measurements.
    """
    if spectra.shape[0]==0:
        return spectra
    if norm_channel is None:
        norm_channel = spectra.columns[np.argmax(np.nanmean(spectra, axis=0))]
    spectra_normed = spectra / spectra[norm_channel].to_numpy()[:,None]
    return spectra_normed

class SingleStainSpilloverCollection():
    """Spectra of one dye collected from many measurements, with an optional PCA fit.

    Created by `SpilloverCollection`, one per value of ``panel_meta[register_on]`` ("stain").

    Parameters
    ----------
    spillovers : pandas.Series of SpilloverMeasurement
        The measurements, index the measurement names.
    dye : str
        Dye name (column of the measurement spectra) under which the stain is collected.
    stain : str
        Value of ``panel_meta[register_on]``; only measurements where the dye has this value are collected.
    register_on : str, default "dye"
        Metadata column that defines the stain.

    Attributes
    ----------
    meta : pandas.DataFrame
        Metadata of the dye per measurement, index the measurement names.
    spectra : dict of str to pandas.DataFrame
        Raw spectra per fit method, each of shape (measurement, channel).
    spectra_calibrated : dict of str to pandas.DataFrame
        Calibrated spectra per fit method, each of shape (measurement, channel).
    pcafitter : SingleSpectrumPCAFitter
        Set by `fit_PCA`.
    """
    def __init__(self, spillovers, dye, stain, register_on="dye"):
        self.spillovers = spillovers
        self.register_on = register_on
        self.dye = dye
        self.stain = stain
        
        self.keys = list(np.unique([k for sp in self.spillovers for k in sp.spectra.keys()]))
        self.keys_calibrated = list(np.unique([k for sp in self.spillovers for k in sp.spectra_calibrated.keys()]))
        
        self.meta = _collect_meta(self.spillovers, self.dye, self.stain, self.register_on)
        
        self.spectra = {key:_collect_spectrum(self.spillovers, self.dye, self.stain, self.register_on, key, calibrated=False)
                        for key in self.keys}
        self.spectra_calibrated = {key:_collect_spectrum(self.spillovers, self.dye, self.stain, self.register_on, key, calibrated=True)
                                   for key in self.keys}

    def normalize_spectra(self, norm_channel=None):
        """Normalise all spectra, every fit method and raw/calibrated separately.

        Parameters
        ----------
        norm_channel : str, optional
            Channel set to 1 in every spectrum. Default: the channel with the highest mean intensity (determined
            separately per fit method and for raw and calibrated spectra).
        """
        self.spectra = {k:_normalize_spectra_set(v, norm_channel=norm_channel)
                                       for k, v in self.spectra.items()}
        self.spectra_calibrated = {k:_normalize_spectra_set(v, norm_channel=norm_channel)
                                       for k, v in self.spectra_calibrated.items()}
    
    def fit_PCA(self, method, calibrated=True):
        """Fit a `SingleSpectrumPCAFitter` (at most 5 components) to the spectra of one fit method.

        Parameters
        ----------
        method : str
            Fit method (key of ``spectra``), e.g. 'linearos'.
        calibrated : bool, default True
            Use the calibrated spectra.
        """
        self.pcafitter = SingleSpectrumPCAFitter(self.spectra_calibrated[method] if calibrated else self.spectra[method], N=5, name=self.stain)
        self.pca_method = method
        self.pca_calibrated = calibrated
    
    def export_pca(self, pc):
        """Export the PCA fit for `PanelConfiguration.collect_PCA_data`.

        Components are named ``'{stain}_PC{i}'``, i counting from 0.

        Parameters
        ----------
        pc : PanelConfiguration
            Panel; its ``components`` select which components of this stain are used.

        Returns
        -------
        dict
            Keys ``embedding`` (DataFrame (measurement, component)), ``components`` (DataFrame (component, channel), L2
            norm 1), ``offset`` (mean spectrum, DataFrame (1, channel), index the stain), ``components_stain`` (Series:
            stain per component), ``components_use`` (list of the component names used by the panel),
            ``explained_variance`` (Series per component), ``explained_variance_channelratio`` (DataFrame (component,
            channel)) and ``spectra`` (the spectra used in the fit, DataFrame (measurement, channel)).
        """
        fitter = self.pcafitter
        spectra = pd.DataFrame(fitter.spectra, index=fitter.obs, columns=fitter.channels)
        component_names = [self.stain+f"_PC{i}" for i in range(fitter.embedding.shape[1])]
        embedding = pd.DataFrame(fitter.embedding, index=fitter.obs, columns=component_names)
        components = pd.DataFrame(fitter.components, columns=fitter.channels, index=component_names)
        offset = pd.DataFrame(fitter.spectra_means, columns=fitter.channels, index=[self.stain])
        components_use = [self.stain+f"_PC{i}" for i in pd.Series(pc.components, index=pc.stains).loc[self.stain]]
        components_stain = pd.Series(self.stain, index=component_names) #[self.stain for i in range(fitter.embedding.shape[1])]

        explained_variance = pd.Series(fitter.explained_variance, index=component_names)
        explained_variance_channelratio = pd.DataFrame(fitter.explained_variance_bychannel_ratio,
                                                       index=component_names, columns=fitter.channels)

        return {"embedding":embedding, "components":components, "offset":offset, "components_stain":components_stain, "components_use":components_use,
                "explained_variance":explained_variance, "explained_variance_channelratio":explained_variance_channelratio, "spectra":spectra}
    
    def plot_spill_correlations_pcafit(self, variance_cutoff=1e-6, Npc=None, figsize=(13,7)):
        """Plot the correlations of the spectra between channels, the variance per channel and its share per component.

        Left: correlation across measurements between the spectrum entries of every pair of channels (clustered);
        centre: variance per channel; right: share of the variance of every channel explained by each component and
        their sum. Uses the spectra of the PCA fit (`fit_PCA` first). Warns (`SpilloverWarning`) and plots nothing
        for a dummy PCA fit.

        Parameters
        ----------
        variance_cutoff : float, default 1e-6
            Only show channels with variance above this (at least the two with the largest variance).
        Npc : int, optional
            Show at most this many components. Default: all.
        figsize : tuple of float, default (13, 7)
            Figure size.
        """
        if self.pcafitter.is_dummy:
            warnings.warn(f"{self.stain} has trivial PCA fit, omitting plotting!", SpilloverWarning)
            return None
        spill_channel = self.spectra_calibrated[self.pca_method] if self.pca_calibrated else self.spectra[self.pca_method]
    
        variance_bychannel = self.pcafitter.variance_bychannel
        explained_variance_bychannel_ratio = self.pcafitter.explained_variance_bychannel_ratio.T
        if Npc is not None:
            explained_variance_bychannel_ratio = explained_variance_bychannel_ratio[:,:Npc]
    
        mask = variance_bychannel>variance_cutoff
        if mask.sum()<2:
            mask[np.argsort(variance_bychannel)[-2:]] = True
    
        spill_channel = spill_channel.iloc[:,mask]
        variance_bychannel = variance_bychannel[mask]
        explained_variance_bychannel_ratio = explained_variance_bychannel_ratio[mask]
    
        def correl(x, y):
            mask = (x==x)&(y==y)
            return np.corrcoef(x[mask], y[mask])[0,1]
        #correls = pd.DataFrame(np.abs(np.asarray([[correl(spill_channel[i], spill_channel[j]) for i in spill_channel.columns]  for j in spill_channel.columns])),
        #                       index=spill_channel.columns, columns=spill_channel.columns)
        correls = pd.DataFrame(np.asarray([[correl(spill_channel[i], spill_channel[j]) for i in spill_channel.columns]  for j in spill_channel.columns]),
                               index=spill_channel.columns, columns=spill_channel.columns)
    
        cg = sns.clustermap(correls, vmin=-1, vmax=1)
        reordered_ind = cg.dendrogram_row.reordered_ind
        plt.close()
    
        spill_channel = spill_channel[spill_channel.columns[reordered_ind]]
        correls = correls.loc[correls.index[reordered_ind], correls.columns[reordered_ind]]
        variance_bychannel = variance_bychannel[reordered_ind]
        explained_variance_bychannel_ratio = explained_variance_bychannel_ratio[reordered_ind]
    
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = plt.GridSpec(1,20, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0:12])
        ax1 = fig.add_subplot(gs[0, 12:17])
        ax2 = fig.add_subplot(gs[0, 17:20])
    
        ax0.imshow(correls, origin="lower", vmin=-1, vmax=1, cmap="RdBu", aspect="auto")
        ax0.set_xticks(np.arange(correls.shape[0]), labels=correls.index, rotation=45, ha="right", rotation_mode='anchor', size=15)
        ax0.set_yticks(np.arange(correls.shape[1]), labels=correls.columns, rotation=0, size=15)
        ax0.set_title(self.stain, size=45)
    
        ax1.scatter(variance_bychannel, np.arange(variance_bychannel.shape[0]), zorder=10, color="black")
    
        ax1.set_yticks([])
        ax1.set_ylim([-.5, variance_bychannel.shape[0]-.5])
        ax1.set_xscale("log")
        ax1.set_title("Variance", size=25)
        ax1.set_xticks(10**np.arange(-7,-1, dtype=float), minor=True)
        ax1.grid(axis="x", which="both")
        ax1.set_xlim([variance_cutoff,max(variance_bychannel.max(), 3e-2)*1.1])
    
        ax2.imshow(np.hstack([explained_variance_bychannel_ratio, explained_variance_bychannel_ratio.sum(1, keepdims=True)]),
                   origin="lower", vmin=0, vmax=1, cmap="Purples", aspect="auto")
        ax2.set_yticks([])
        N = explained_variance_bychannel_ratio.shape[1]
        ax2.set_xticks(np.arange(N+1), labels=[f"PC {i}" for i in range(N)]+[f"SUM"], rotation=90, size=15)


class SpilloverCollection():
    """Single-stain spectra of many measurements, collected per stain, with PCA fits of the spectra of every stain.

    A stain is a value of the metadata column `register_on` (e.g. the dye, or a ``fluorophore`` column that tells apart
    different products recorded in the same channel). Every stain is collected under the dye name (spectrum column)
    of its first occurrence: in each measurement, only the dye with that name is used, and only if its `register_on`
    value is that stain.

    Parameters
    ----------
    spillovers : pandas.Series of SpilloverMeasurement
        The measurements, index the measurement names (e.g. the single-stain runs).
    register_on : str, default "dye"
        Column of the ``panel_meta`` of the measurements that defines the stains. Within every measurement, each value
        may belong to at most one dye.

    Attributes
    ----------
    available_keys : pandas.Series
        Stains (values), sorted, with the dye name they are collected under as index.
    single_stains : dict of str to SingleStainSpilloverCollection
        Collection per stain; also accessible as ``collection[stain]``.

    Examples
    --------
    >>> spcollection = SpilloverCollection(pd.Series(spillovers, index=runs), register_on="dye")
    >>> spcollection.normalize_spectra(key_dye_channel)
    >>> spcollection.add_PCA_fit("linearos", calibrated=True)
    """
    def __init__(self, spillovers, register_on="dye"):
        self.spillovers = spillovers
        self.register_on = register_on
        
        self.available_keys = pd.concat([sp.panel_meta[self.register_on] for sp in self.spillovers]).drop_duplicates().sort_values()
        
        self.single_stains = {stain: SingleStainSpilloverCollection(self.spillovers, dye, stain, self.register_on)
                              for dye, stain in self.available_keys.items()}
    
    def __getitem__(self, key):
        return self.single_stains[key]

    def normalize_spectra(self, dye_channel_key=None):
        """Normalise all spectra of every stain (see `SingleStainSpilloverCollection.normalize_spectra`).

        Parameters
        ----------
        dye_channel_key : pandas.Series, optional
            Channel to set to 1 per stain: index the stains (values of `register_on`), values the channels. Stains
            missing from it get a warning and use the channel with the highest mean intensity, as all stains do if it is
            not given.
        """
        for key in self.available_keys:
            norm_channel = None
            if dye_channel_key is not None:
                try:
                    norm_channel = dye_channel_key.loc[key]
                except KeyError:
                    warnings.warn(f"Dye channel key was given, but could not find an entry for {key}!")
            self[key].normalize_spectra(norm_channel)
    
    def plot_peak_position(self):
        """Plot the position of the positive peak of every single stain, per stain (log scale).

        Requires the metadata column ``positive_peak`` (see `collect_peakpos_fromposneg`).
        """
        fig, ax = plt.subplots(1,1,figsize=(10,5/20*len(self.available_keys)))
        for key in self.available_keys:
            sp = self[key]
            if (~np.isnan(sp.meta["positive_peak"])).sum()>0:
                plt.scatter(sp.meta["positive_peak"], sp.meta[self.register_on], s=4)
        plt.xscale("log")
        plt.xlabel("Positive Peak Intensity")
    
    def plot_spectrum(self, stain, method, hue, calibrated=False):
        """Plot all spectra of one stain, one line per measurement, channels sorted by mean value (log scale).

        Parameters
        ----------
        stain : str
            Stain to plot (value of `register_on`).
        method : str
            Fit method of the spectra, e.g. 'linearos'.
        hue : str
            Categorical metadata column by which to colour the measurements, e.g. 'antigen'.
        calibrated : bool, default False
            Use the calibrated spectra.
        """
        sp = self[stain]
        spectra = sp.spectra_calibrated[method] if calibrated else sp.spectra[method]
        plothue = sp.meta[hue].astype(str)

        fig, ax = plt.subplots(1,1,figsize=(10,7))
        cmap = get_cmap(plothue)

        inds = np.asarray(np.argsort(spectra.mean(0)))
        spectra = spectra.iloc[:,inds]

        for name, row in spectra.iterrows():
            if (~np.isnan(row)).sum()>0:
                ax.scatter(row, row.index, color=cmap[plothue.loc[name]], s=20)
                ax.plot(row, row.index, color=cmap[plothue.loc[name]], linewidth=.5)
        for i in range(spectra.shape[1]+1):
            ax.axhline(i-.5, color="gray", linewidth=1, alpha=.4)
        ax.legend(handles=cmap_to_legendhandles(cmap), loc="lower right")
        ax.set_xscale("log")
        ax.set_xlim([5e-3,None])
        ax.set_title(stain, size=20)
    
    def plot_spectrum_correlation_single(self, stain, method, c1, c2, hue, calibrated=False):
        """Plot the spectrum entries of one stain in two channels against each other, one point per measurement.

        The title shows the Pearson correlation.

        Parameters
        ----------
        stain : str
            Stain to plot (value of `register_on`).
        method : str
            Fit method of the spectra, e.g. 'linearos'.
        c1 : str
            Channel on the x axis.
        c2 : str
            Channel on the y axis.
        hue : str
            Metadata column by which to colour the points.
        calibrated : bool, default False
            Use the calibrated spectra.
        """
        sp = self[stain]
        spectra = sp.spectra_calibrated[method] if calibrated else sp.spectra[method]
        plothue = sp.meta[hue]

        fig, ax = plt.subplots(1,1,figsize=(8, 5))
        
        x, y = spectra[c1], spectra[c2]
        mask = (x==x) & (y==y)
        sns.scatterplot(x=x[mask], y=y[mask], hue=plothue[mask])
        ax.set_title(stain+f", r={np.corrcoef(x[mask],y[mask])[0,1]:.2f}", size=20)
        ax.set_xlabel(c1, size=15)
        ax.set_ylabel(c2, size=15)
    
    def count_channels_above_cutoff(self, method="linearos", cutoff=.01, calibrated=False):
        """Count per stain the channels whose spillover exceeds `cutoff` in at least one measurement.

        Entries that are exactly 1 (the normalisation channel) are not counted.

        Parameters
        ----------
        method : str, default "linearos"
            Fit method of the spectra.
        cutoff : float, default 0.01
            Threshold on the spectrum entries (normalised spectra).
        calibrated : bool, default False
            Use the calibrated spectra.

        Returns
        -------
        pandas.Series
            Number of channels per stain.
        """
        def get_single(key):
            spectrum = self[key].spectra_calibrated[method] if calibrated else self[key].spectra[method]
            return (((spectrum>cutoff) & (spectrum!=1.)).sum()>0).sum()
        return pd.Series([get_single(key) for key in self.available_keys], index=self.available_keys)
    
    def count_channels_range_above_cutoff(self, method="linearos", cutoff=.01, calibrated=False):
        """Count per stain the channels whose spillover range (max - min over measurements) exceeds `cutoff`.

        Parameters
        ----------
        method : str, default "linearos"
            Fit method of the spectra.
        cutoff : float, default 0.01
            Threshold on the range of the spectrum entries (normalised spectra).
        calibrated : bool, default False
            Use the calibrated spectra.

        Returns
        -------
        pandas.Series
            Number of channels per stain.
        """
        def get_single(key):
            spectrum = self[key].spectra_calibrated[method] if calibrated else self[key].spectra[method]
            return ((spectrum.max()-spectrum.min())>cutoff).sum()
        return pd.Series([get_single(key) for key in self.available_keys], index=self.available_keys)

    def add_PCA_fit(self, method="linearos", calibrated=False):
        """Fit a PCA (at most 5 components) to the spectra of every stain, see `SingleStainSpilloverCollection.fit_PCA`.

        Normalise the spectra first (`normalize_spectra`).

        Parameters
        ----------
        method : str, default "linearos"
            Fit method of the spectra.
        calibrated : bool, default False
            Use the calibrated spectra.
        """
        for sp in self.single_stains.values():
            sp.fit_PCA(method, calibrated=calibrated)
        self.pca_method = method
        self.pca_calibrated = calibrated
    
    def plot_total_variance(self, channel_max=False):
        """Plot the total variance of the spectra per stain (log scale); requires `add_PCA_fit`.

        Parameters
        ----------
        channel_max : bool, default False
            Plot the largest variance of a single channel instead of the total over channels.
        """
        total_var = pd.Series(self.single_stains).apply(lambda x: x.pcafitter.max_channel_variance if channel_max else x.pcafitter.total_variance)
        
        fig, ax = plt.subplots(1,1,figsize=(5,.25*len(total_var)))
        sns.scatterplot(x=total_var, y=total_var.index)
        ax.set_xscale("log")
        ax.grid(axis="x")
        ax.set_xlabel("Total Variance", size=20)
        ax.set_ylabel("")
    
    def plot_explained_variance(self):
        """Plot the explained variance of every PCA component per stain (log scale); requires `add_PCA_fit`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        total_var = pd.Series(self.single_stains).apply(lambda x: x.pcafitter.total_variance)
        exp_var = [s.pcafitter.explained_variance for s in pd.Series(self.single_stains)]
        Npad = max([len(e) for e in exp_var])
        exp_var = np.asarray([list(e) + [0]*(Npad-len(e)) for e in exp_var])
        cmap = get_cmap(np.arange(exp_var.shape[1]))

        fig, ax = plt.subplots(1,1,figsize=(7,.25*len(total_var)))
        for i in range(exp_var.shape[1]):
            sns.scatterplot(x=exp_var[:,i], y=total_var.index, zorder=10, color=cmap[i])
        ax.set_xscale("log")
        ax.grid(which="minor", axis="x")
        ax.set_xlabel("Explained Variance per Component")
        ax.set_ylabel("")
        ax.legend(handles=cmap_to_legendhandles(cmap))

        return ax
    
    def plot_explained_variance_panel(self, pc):
        """Plot the explained variance of every PCA component for the stains of a panel, used components outlined.

        Like `plot_explained_variance`, restricted to the stains of `pc` and labelled with ``pc.stain_marker_name``;
        requires `add_PCA_fit`.

        Parameters
        ----------
        pc : PanelConfiguration
            Panel; its stains must be stains of the collection.

        Returns
        -------
        matplotlib.axes.Axes
        """
        total_var = pd.Series(self.single_stains).loc[pc.stains].apply(lambda x: x.pcafitter.total_variance)
        exp_var = [s.pcafitter.explained_variance for s in pd.Series(self.single_stains).loc[pc.stains]]
        Npad = max([len(e) for e in exp_var])
        exp_var = np.asarray([list(e) + [0]*(Npad-len(e)) for e in exp_var])
        cmap = get_cmap(np.arange(exp_var.shape[1]))

        fig, ax = plt.subplots(1,1,figsize=(7,.25*len(total_var)))
        for i in range(exp_var.shape[1]):
            sns.scatterplot(x=exp_var[:,i], y=pc.stain_marker_name, zorder=10, color=cmap[i])
            mask = np.array([np.any(np.isin(c, [i])) for c in pc.components])
            if mask.sum()>0:
                sns.scatterplot(x=exp_var[mask,i], y=pc.stain_marker_name[mask], color="None", edgecolors="black", linewidth=1.5, zorder=11)
            
        ax.set_xscale("log")
        ax.grid(which="minor", axis="x")
        ax.set_xlabel("Explained Variance per Component")
        ax.set_ylabel("")
        ax.set_title(f"Panel {pc.name}")
        ax.legend(handles=cmap_to_legendhandles(cmap))

        return ax

