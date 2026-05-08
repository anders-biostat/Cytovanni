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
    files = list(filter(lambda x: x.endswith(".h5ad"), os.listdir(folder)))
    files = sorted(files, key=lambda f: os.path.getsize(os.path.join(folder, f)))
    return os.path.join(folder, files[0])

def _ensure_limit_channels(matrix, channels):
    """ Ensure only entries in channels are kept, fills up missing with nan.
    """
    matrix = matrix.loc[matrix.index.intersection(channels)]
    for row in pd.Index(channels).difference(matrix.index):
        matrix.loc[row] = np.nan
    matrix = matrix.loc[channels]
    return matrix
def _read_matrix_custom(filepath, channels):
    """ Read result of custom single stain fit.
    """
    matrix = pd.read_table(filepath, sep=",", index_col=0)
    matrix = _ensure_limit_channels(matrix, channels)
    matrix = matrix[sorted(matrix.columns)]
    return matrix
def _read_matrix_autospill(filepath, channels):
    """ Read result of AutoSpill fit.
    """
    matrix = pd.read_table(filepath, sep=",", index_col=0).T
    matrix.columns = [c[:-2] if c.endswith("-A") else c for c in matrix.columns]
    matrix = _ensure_limit_channels(matrix, channels)
    matrix = matrix[sorted(matrix.columns)]
    return matrix

class SpilloverMeasurement():
    """ Spillover matrices for one single stain measurement.
        
        Document how it should look for custom from_ !
    """
    def __init__(self, rbint=None, is_spectral=False, rainbow_batch=None, rainbow_batch_adata=None, channels=None):
        """ 
            
            :param rbint: RainbowCalibrator.
            
            :param is_spectral: bool. Whether to use spectral convention to normalize single stain spectra.
            
            :param rainbow_batch: str. Optional, rainbow batch for calibration factor.
            
            :param rainbow_batch_adata: AnnData. Optional, adata from which to infer rainbow batch.

            :param channels: list. Optional, if rbint is not given should contain the fluorescence channels.
        """
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
        """ Load fitted spectra from custom spectra fit result.
            Ensures all single stains were recorded at the same settings.
            
            :param rbint: RainbowCalibrator.
            
            :param custom_data_folder: str. Folder where the custom fit was done.
            
            :param rainbow_batch: str. Optional, rainbow batch for single stain recording, if not given trys to infer it from smallest data .h5ad file.
            
            :param abort_inconsistent: str. Whether to abort if inconsistent settings are found, or just throw a warning.
            
            :param asp_folder: str. Optional, folder where AutoSpill stored its results.
            
            :param is_spectral: bool. Whether spectral flow conventions should be used when normalizing spectra.
            
            :param clip_negative: bool. Whether to clip negative spectra entries to zero.
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
                warnings.warn(f"Found inconsistent settings in single stains in {custom_data_folder}!", SpilloverException)
        
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
        """ Load fitted spectra from AutoSpill fit result.
            
            :param rbint: RainbowCalibrator.
            
            :param asp_folder: str. Folder where AutoSpill stored its results.
            
            :param rainbow_batch: str. Rainbow batch for single stain recording.
            
            :param is_spectral: bool. Whether spectral flow conventions should be used when normalizing spectra.
            
            :param clip_negative: bool. Whether to clip negative spectra entries to zero.
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
        """ Load fitted spectra from .csv files.
            Spectra should have index channels and columns dyes, while metadata should have index dyes.
            
            :param file_spectra: str. Path to .csv with the dye spectra.
            
            :param file_meta: str. Path to the .csv with the dye metadata.
            
            :param is_spectral: bool. Whether spectral flow conventions should be used when normalizing spectra.
            
            :param clip_negative: bool. Whether to clip negative spectra entries to zero.
        """
        spectra = pd.read_table(file_spectra, sep=",", index_col=0)
    
        spm = cls(is_spectral=is_spectral, channels=spectra.index)
        
        spm.panel_meta = pd.read_table(file_meta, sep=",", index_col=0)
        spm.spectra["spectra"] = spectra
        if clip_negative: spm.clip_negative_spectra()
        
        return spm
    
    
    def try_load_AutoSpill_spectra(self, warn_empty=True):
        """ Try loading AutoSpill spectra from self.asp_folder if present.
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
        """ Try loading custom spectra from self.custom_data_folder if present.
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
        """ Make sure all dyes in self.spectra have metadata.
        """
        present_dyes = set().union(*[set(s.columns) for s in self.spectra.values()])
        metadata_dyes = set(self.panel_meta.index)
        
        if len(present_dyes | metadata_dyes) > len(metadata_dyes):
            raise SpilloverException(f"Missing metadata for dyes {list(present_dyes - metadata_dyes)}!")
    
    def add_calibrated_spectra(self):
        """ Add batch calibrated version of all present spectra.
        """
        for key in self.spectra:
            self.spectra_calibrated[key] = scale_spectra_wfactors(self.spectra[key], self.calibration_factor, self.is_spectral)
    
    def clip_negative_spectra(self, warn=True):
        """ Clip all negative spectra entries to zero.
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
        """ Using key from self.panel_meta, mask all spectra where key is True with nan.
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
    index = []
    metas = []
    for name, sp in spillovers.items():
        if (dye in sp.panel_meta.index) and (sp.panel_meta.loc[dye, register_on]==stain):
            index.append(name)
            metas.append(sp.panel_meta.loc[dye])
    return pd.DataFrame(metas, index=index)

def _collect_spectrum(spillovers, dye, stain, register_on, key, calibrated=False):
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
    """ If given, normalizes all spectra of a single dye to one in norm_channel, otherwise in the channel that has the highest mean intensity.

        :param spectra: pd.DataFrame. set of spectra for one dye, index measurement, column channel.

        :param norm_channel: None, str. If given, channel to normalize to one.
    """
    if spectra.shape[0]==0:
        return spectra
    if norm_channel is None:
        norm_channel = spectra.columns[np.argmax(np.nanmean(spectra, axis=0))]
    spectra_normed = spectra / spectra[norm_channel].to_numpy()[:,None]
    return spectra_normed

class SingleStainSpilloverCollection():
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
        """ Normalize all spectra in self.
            Normalizes each extraction type, calibrated/raw etc. separately.
            Either norm_channel is set to one, or the channel with the highest average intensity.
        """
        self.spectra = {k:_normalize_spectra_set(v, norm_channel=norm_channel)
                                       for k, v in self.spectra.items()}
        self.spectra_calibrated = {k:_normalize_spectra_set(v, norm_channel=norm_channel)
                                       for k, v in self.spectra_calibrated.items()}
    
    def fit_PCA(self, method, calibrated=True):
        self.pcafitter = SingleSpectrumPCAFitter(self.spectra_calibrated[method] if calibrated else self.spectra[method], N=5, name=self.stain)
        self.pca_method = method
        self.pca_calibrated = calibrated
    
    def export_pca(self, pc):
        """ Export PCA data for panel pc.
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
        """ Plot correlations of spectra, along with share of explained variance for PCs.
            Only show channels with variance above 'variance_cutoff', and at most Npc PCs.
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
    def __init__(self, spillovers, register_on="dye"):
        """ 
            
            :param spillovers: pd.Series. Series of SpilloverMeasurement objects, with index the name of the measurement.
            
            :param register_on: str. SpilloverMeasurement spectra should have column names for spectra that fit to panel_meta, collect data separated by the corresponding entry in panel_meta[register_on]. Needs to be unique to at most one spectrum within every measurement.
        """
        self.spillovers = spillovers
        self.register_on = register_on
        
        self.available_keys = pd.concat([sp.panel_meta[self.register_on] for sp in self.spillovers]).drop_duplicates().sort_values()
        
        self.single_stains = {stain: SingleStainSpilloverCollection(self.spillovers, dye, stain, self.register_on)
                              for dye, stain in self.available_keys.items()}
    
    def __getitem__(self, key):
        return self.single_stains[key]

    def normalize_spectra(self, dye_channel_key=None):
        """ Normalize all contained spectra.
            Either takes the channel to be normed to one from dye_channel_key, or uses the channel with the highest average intensity for every dye.

            :param dye_channel_key: pd.Series. Optional, pd.Series with index dye name and entry channel which should be normalized to one for that dye.
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
        """ For all single stain fits, plot position of positive peak.
            Requires "positive_peak" to be present in metadata.
        """
        fig, ax = plt.subplots(1,1,figsize=(10,5/20*len(self.available_keys)))
        for key in self.available_keys:
            sp = self[key]
            if (~np.isnan(sp.meta["positive_peak"])).sum()>0:
                plt.scatter(sp.meta["positive_peak"], sp.meta[self.register_on], s=4)
        plt.xscale("log")
        plt.xlabel("Positive Peak Intensity")
    
    def plot_spectrum(self, stain, method, hue, calibrated=False):
        """ Plot all single stain spectra for the same stain.
            
            :param stain: str. Stain to plot, value of register_on.
            
            :param method: str. Fit method from which to get the spectra, e.g. 'posneg' etc.
            
            :param hue: str. Categorical metadata by which to color the plot.
            
            :param calibrated: bool. Whether to use the calibrated spectra.
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
        """ Plot all single stain spectra for the same stain, values in two channels against each other.
            
            :param stain: str. Stain to plot, value of register_on.
            
            :param method: str. Fit method from which to get the spectra, e.g. 'posneg' etc.
            
            :param c1: str. Channel on the x axis.
            
            :param x2: str. Channel on the y axis.
            
            :param hue: str. Categorical metadata by which to color the plot.
            
            :param calibrated: bool. Whether to use the calibrated spectra.
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
        """ For every dye, count the number of channels in which the spillover is larger than cutoff
            for at least one spectrum measurement.
        """
        def get_single(key):
            spectrum = self[key].spectra_calibrated[method] if calibrated else self[key].spectra[method]
            return (((spectrum>cutoff) & (spectrum!=1.)).sum()>0).sum()
        return pd.Series([get_single(key) for key in self.available_keys], index=self.available_keys)
    
    def count_channels_range_above_cutoff(self, method="linearos", cutoff=.01, calibrated=False):
        """ For every dye, count the number of channels in which the possible spillover range
            is larger than cutoff.
        """
        def get_single(key):
            spectrum = self[key].spectra_calibrated[method] if calibrated else self[key].spectra[method]
            return ((spectrum.max()-spectrum.min())>cutoff).sum()
        return pd.Series([get_single(key) for key in self.available_keys], index=self.available_keys)

    def add_PCA_fit(self, method="linearos", calibrated=False):
        """ Add PCA fit for every single stain.
        """
        for sp in self.single_stains.values():
            sp.fit_PCA(method, calibrated=calibrated)
        self.pca_method = method
        self.pca_calibrated = calibrated
    
    def plot_total_variance(self, channel_max=False):
        """ Plot total variance of spectra per dye.
        """
        total_var = pd.Series(self.single_stains).apply(lambda x: x.pcafitter.max_channel_variance if channel_max else x.pcafitter.total_variance)
        
        fig, ax = plt.subplots(1,1,figsize=(5,.25*len(total_var)))
        sns.scatterplot(x=total_var, y=total_var.index)
        ax.set_xscale("log")
        ax.grid(axis="x")
        ax.set_xlabel("Total Variance", size=20)
        ax.set_ylabel("")
    
    def plot_explained_variance(self):
        """ Plot explained variance per component and dye.
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
        """ Plot explained variance per component and dye.
            Limited to dyes used in the panel pc, and marking the used components.
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

