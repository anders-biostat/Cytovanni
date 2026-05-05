import numpy as np
import pandas as pd
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import flowkit as fk
import os

from ..utils import printwtime
from ..io import writefcs, readfcs_sample
from ..exceptions import IntegrationModuleWarning, IntegrationModuleException, MissingChannelException, MissingGPUWarning, IntegrationModuleUntrainedParameterWarning
from .rboe import RainbowBatchOrdinalEncoder
from .rbim_scatter import RainbowScatterIntegrationModule
from .rbim_data import RainbowFluorescenceGMMIntegrationDataset
from .rbim_fluorescence import RainbowFluorescenceGMMIntegrationModule, RainbowFluorescenceGMMIntegrationTrainer
from .utils import scale_spectra_wfactors

import scipy.stats as stats
def _plot_rfim_pdf(ax, i, mean, std, shares, islinear, vline=True):
    """ Plot PDF of the rainbow model in channel i onto axis ax.
    """
    xlim = ax.get_xlim()
    x = np.linspace(xlim[0], xlim[1], 1000)
    y_linear = (stats.norm.pdf(x[:,None], mean[None,:,i], std[None,:,i]) * shares[None])[:,islinear[:,i]].sum(-1)
    y_nonlinear = (stats.norm.pdf(x[:,None], mean[None,:,i], std[None,:,i]) * shares[None])[:,~islinear[:,i]].sum(-1)
    ax.plot(x, y_nonlinear, zorder=-2, color="dimgray")
    ax.plot(x, y_linear, zorder=-1, color="black")
    ax.set_xlim(xlim)
    if vline:
        for v in mean[:,i]:
            ax.axvline(v, color="gray", zorder=-3)

def _geometric_mean(x):
    return np.exp(np.log(x).mean())

class RainbowIntegrator():
    REQUIRED_UNS = ["cytometer", "rainbow_batch", "rainbow_type", "uid", "set_hash"]
    REQUIRED_GMM_UNS = ["GMM_means", "GMM_asinh_cofactor"]
    REQUIRED_GMM_OBS = ["GMM_label"]
    REQUIRED_LAYER = ["raw"]
    channel_missing_cutoff = None
    
    def __init__(self, adatas=None, include_shift=True):
        """ Initialise from list of adatas.
            
            Requires:
                'cytometer': name of the cytometer
                'rainbow_batch': batch name
                'rainbow_type': type of the rainbow bead (lot etc.)
                'uid': rainbow bead sample unique identifier
                'set_hash': hash of relevant cytometer settings (voltages etc.) to ensure no changes in batches
            to be present in adata.uns.
            If available also uses 'UTC' and 'date'.
            
            The fluorescence integration also expects adata.uns['GMM_means'], adata.uns['GMM_asinh_cofactor']
            and adata.obs['GMM_label'] as added by fit_rainbow_GMM.
            
            If adatas is None, has to be initialised manually later!
        """
        if adatas is not None:
            self._check_adatas(adatas)
            self.adatas = adatas

            # Initialise batch encoder
            df_meta = self._process_adatas_meta(adatas)
            self.rboe = RainbowBatchOrdinalEncoder.from_data(df_meta)

            # Initialize and fit scatter integration
            self.rsim = RainbowScatterIntegrationModule.from_data(self.rboe, self.adatas)

            # Initialize fluorescence integration
            dct_Npeak = RainbowFluorescenceGMMIntegrationDataset.get_type_Npeak(self.adatas)
            channels = self.adatas[0].uns["cytoconfig"].channels_fluorescence # have to be the same across all adatas anyway
            self._check_adatas_channels(self.adatas, channels) # make sure they are all present
            class_cofactor = {tp: _geometric_mean(df_meta.loc[df_meta["rainbow_type"]==tp, "GMM_asinh_cofactor"].to_numpy().astype(float))
                              for tp in df_meta["rainbow_type"].unique()} # geometric mean of cofactors across each type
            self.rfim = RainbowFluorescenceGMMIntegrationModule(self.rboe, dct_Npeak, channels, class_cofactor, include_shift=include_shift)
    
    
    @classmethod
    def _check_adatas(cls, adatas):
        """ Checks that all required entries are present in adatas.
        """
        abort = False
        
        for key in cls.REQUIRED_UNS:
            missing = sum([key not in ad.uns for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing .uns metadata key {key}!", IntegrationModuleWarning)
        
        for key in cls.REQUIRED_GMM_UNS:
            missing = sum([key not in ad.uns for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing .uns metadata key {key}! Are you sure they were set up with fit_rainbow_GMM?", IntegrationModuleWarning)
        
        for key in cls.REQUIRED_GMM_OBS:
            missing = sum([key not in ad.obs for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing .obs metadata key {key}! Are you sure they were set up with fit_rainbow_GMM?", IntegrationModuleWarning)
        
        for key in cls.REQUIRED_LAYER:
            missing = sum([key not in ad.layers for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing the 'raw' data in .layers!", IntegrationModuleWarning)
        
        if abort:
            raise IntegrationModuleException("Passed adatas are missing some initialisations!")
    
    @classmethod
    def _check_adatas_channels(cls, adatas, channels):
        """ Ensure that all fluorescence channels are present in all adatas.
        """
        missing = 0
        for ad in adatas:
            if not np.all(np.in1d(channels, ad.var.index)):
                missing += 1
        if missing>0:
            raise MissingChannelException(f"{missing} passed adatas are missing some of the required fluorescence channels {channels}!")
    
    @classmethod
    def _process_adatas_meta(self, adatas):
        """ Extract metadata from adatas.
        """
        df_meta = pd.DataFrame([[ad.uns["cytometer"] for ad in adatas],
                                [ad.uns["rainbow_batch"] for ad in adatas],
                                [ad.uns["rainbow_type"] for ad in adatas],
                                [ad.uns["uid"] for ad in adatas],
                                [ad.uns["set_hash"] for ad in adatas],
                                [ad.uns["UTC"] if "UTC" in ad.uns else np.nan for ad in adatas],
                                [ad.uns["date"] if "date" in ad.uns else np.nan for ad in adatas],
                                [ad.uns["GMM_asinh_cofactor"] for ad in adatas],],
                            index = ["cytometer", "rainbow_batch", "rainbow_type", "uid", "set_hash",
                     "UTC", "date", "GMM_asinh_cofactor"]).T
        return df_meta
    
    
    def fit_fluorescence(self, gpu=True, device="cuda:0", verbosity=2, freeze=True, channel_missing_cutoff=-np.inf):
        """ Fit fluorescence part of the model.
            
            :param gpu: bool. If True, tries to run fitting on GPU.
            
            :param device: bool. CUDA device if gpu, defaults to 'cuda:0'
            
            :param verbosity: int. 0 silent, 1 only progress bars, 2 also loss plots.
            
            :param freeze: bool. Whether all parameters should be frozen when the fit is done. Cannot be reversed, so should be turned off when the same batches should be trained again afterwards. But necessary to achieve the correct behaviour when extending the model.
            
            :param channel_missing_cutoff: float. Usually, all channels need to be present for the integration to work. But when looking at calibration data etc. there may be rainbow bead measurements where a laser was turned off etc. Setting this to a finite value will treat all channels where every peak (in a given sample) is below this value as not fittable to avoid trying to fit pure noise. This is done by simply treating all peaks in that channel as nonlinear in the same way as for peaks that are off-scale.
        """
        if not hasattr(self, "adatas"):
            raise IntegrationModuleException("RainbowIntegrator was loaded from saved and has no training data attached! If you want to extend the model use .extend() instead!")
        
        if gpu and not torch.cuda.is_available():
            warnings.warn("Tried using GPU but torch doesn't find any available devices, falling back to CPU at much lower training speed!\n Use 'import torch; torch.set_num_threads(N)' before fitting to set the number of CPU cores torch should use, defaults to using only one.", MissingGPUWarning)
            gpu = False
        if not gpu:
            device = "cpu"
        if self.channel_missing_cutoff is not None and self.channel_missing_cutoff!=channel_missing_cutoff:
            warnings.warn("Using a different 'channel_missing_cutoff' than was used for last training!", IntegrationModuleWarning)
        self.channel_missing_cutoff = channel_missing_cutoff
        
        if verbosity>=1: printwtime("Initializing Dataset")
        self.dataset = RainbowFluorescenceGMMIntegrationDataset(self.rboe, self.adatas, device=device, channel_missing_cutoff=self.channel_missing_cutoff)
        self.rfim.to(device)
        self.rfim_trainer = RainbowFluorescenceGMMIntegrationTrainer(self.rfim)
        
        if verbosity>=1: printwtime("Fit Initialization")
        self.rfim_trainer.fit_init(self.dataset, showprogress=verbosity>=1)
        if verbosity>=2: self.rfim_trainer.plot_fit_init()
        
        if verbosity>=1: printwtime("Full Model Fit")
        self.rfim_trainer.fit(self.dataset, showprogress=verbosity>=1)
        if freeze: self.rfim.freeze()
        if verbosity>=2: self.rfim_trainer.plot_fit()
    
    
    def extend_fit(self, adatas, **fitkwargs):
        """ Extend fitmodel to new samples in adata, freezes all previously present parameters.
            Also fits the fluorescence.
            Previously loaded data is discarded and not used for the extension fit!
            
        """
        self._check_adatas(adatas)
        self.adatas = adatas
        
        # Extend batch encoder
        df_meta = self._process_adatas_meta(self.adatas)
        extend_inds = self.rboe.extend(df_meta)
        
        # Extend and fit scatter integrator
        self.rsim.extend(self.rboe, extend_inds, self.adatas)
        
        # Extend fluorescence integrator, freeze previous parameters in case this was not done previously
        dct_Npeak = RainbowFluorescenceGMMIntegrationDataset.get_type_Npeak(self.adatas)
        channels = self.adatas[0].uns["cytoconfig"].channels_fluorescence # have to be the same across all adatas anyway
        self._check_adatas_channels(self.adatas, channels) # make sure they are all present
        class_cofactor = {tp: _geometric_mean(df_meta.loc[df_meta["rainbow_type"]==tp, "GMM_asinh_cofactor"].to_numpy().astype(float))
                          for tp in df_meta["rainbow_type"].unique()} # geometric mean of cofactors across each type
        self.rfim.freeze()
        self.rfim.extend(self.rboe, extend_inds, dct_Npeak=dct_Npeak, class_cofactor=class_cofactor)
        
        # Fit extended fluorescence integrator
        self.fit_fluorescence(**fitkwargs)
    
    
    def export(self):
        """ Export self to dict.
        """
        exported = {}
        exported["rboe"] = self.rboe.export()
        exported["rsim"] = self.rsim.export()
        exported["rfim"] = self.rfim.export()
        exported["channel_missing_cutoff"] = self.channel_missing_cutoff
        return exported
    
    def save(self, filepath, make_path_structure=False):
        """ Save self to filepath.
            
            :param filepath: str. Path where file should be saved, should end in '.pt'.
            
            :param make_path_structure: bool. Whether path structure in filepath should be created if it doesn't yet exist. Turned off by default to not silently ignore unintentional errors in filepath, but can be enabled.
        """
        exported = self.export()
        if make_path_structure:
            Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True) # ensure path exists
        torch.save(exported, filepath)
    
    @classmethod
    def from_saved(cls, filepath):
        """ Reconstruct saved integrator from saved at filepath.
        """
        exported = torch.load(filepath, map_location=torch.device("cpu"), weights_only=False)
        rbint = cls()
        rbint.rboe = RainbowBatchOrdinalEncoder.from_exported(exported["rboe"])
        rbint.rsim = RainbowScatterIntegrationModule.from_exported(exported["rsim"], rbint.rboe)
        rbint.rfim = RainbowFluorescenceGMMIntegrationModule.from_exported(exported["rfim"], rbint.rboe)
        rbint.channel_missing_cutoff = exported["channel_missing_cutoff"]
        return rbint
    
    
    def get_integration_factor(self, rainbow_batch=None, adata=None):
        """ Get integration factor from batch into common units.
            Integration is done by multiplying the raw data with the factors.
            Correspondingly, to take data from batch1 to batch2 instead of to the common units, multiply with factors1 / factors2.
            
            Includes integration factors for both height and area of scatter channels if applicable,
            only includes factors for area of fluorescence channels.

            :param rainbow_batch: str. Optional, batch for which to get the integration factor.

            :param adata: AnnData. Optional, AnnData from which to get the batch for the integration factor.
        """
        factor_scatter = self.rsim.get_integration_factor(rainbow_batch=rainbow_batch, adata=adata)
        factor_fluorescence = self.rfim.get_integration_factor(rainbow_batch=rainbow_batch, adata=adata, fix_untrained=True, warn_untrained=True, return_mask=False)
        factor = pd.concat([factor_scatter, factor_fluorescence])
        return factor
    
    def get_integration_factor_all_single(self, rainbow_batch):
        """ get_integration_factor, modified version for get_integration_factors_all.
        """
        factor_scatter = self.rsim.get_integration_factor(rainbow_batch=rainbow_batch)
        factor_fluorescence, mask_trained_fluorescence = self.rfim.get_integration_factor(rainbow_batch=rainbow_batch, fix_untrained=True, warn_untrained=False, return_mask=True)
        factor = pd.concat([factor_scatter, factor_fluorescence])
        return factor, mask_trained_fluorescence
    
    def get_integration_factors_all(self, warn_untrained=True):
        """ Run .get_integration_factor for all available rainbow batches, return as DataFrame.
        """
        outs = [self.get_integration_factor_all_single(rainbow_batch=label) for label in self.rboe.oes["rainbow_batch"].labels]
        factors = pd.DataFrame([o[0] for o in outs],
             index=self.rboe.oes["rainbow_batch"].labels)
        mask = pd.DataFrame([o[1] for o in outs])
        
        if warn_untrained and ((~mask).sum().sum()>0):
            warnstr = "For some batches, some scaling factors were not trained! They are set to 1."
            n_channel, Nc = ((~mask.to_numpy()).sum(0)>0).sum(), mask.shape[1]
            n_batch, Nb = ((~mask.to_numpy()).sum(1)>0).sum(), mask.shape[0]
            warnstr+= f"\n{n_channel} (out of {Nc}) channels are not trained in some batches, {n_batch} (out of {Nb}) batches have some untrained channels."
            warnings.warn(warnstr, IntegrationModuleUntrainedParameterWarning)
        
        return factors
    
    def get_rainbow_shifts_all(self):
        """ Get all rainbow integration shifts.
            Returns a dictionary for every rainbow type.
        """
        outdct = {}
        df = self.rboe.data_df[["uid", "rainbow_type"]]
        for tp in self.rboe.oes["rainbow_type"].labels:
            df_tp = df[df["rainbow_type"]==tp]
            shifts = pd.DataFrame([self.rfim.get_integration_rainbow_shift(uid=row["uid"], tp=row["rainbow_type"])
                                  for name, row in df_tp.iterrows()], index=df_tp["uid"].to_numpy())
            outdct[tp] = shifts
        
        return outdct
    
    def get_rainbow_peak_parameters(self, adata, channels):
        """ Get parameters of the rainbow GMM fit for adata and channels.
            
            Returned parameters are exactly like they are used in the model,
            i.e. they describe the raw measured data after adding the trained shift.
            
            Return means and covariances of Gaussians on arcsinh scale, the class shares,
            wether the peak is in the linear range, as well as the arcsinh cofactor that was used.
        """
        dataset = RainbowFluorescenceGMMIntegrationDataset(self.rboe, [adata], device="cpu", warn_missing=False,
                                                           channel_missing_cutoff=self.channel_missing_cutoff)

        peak_islinear = list(dataset.peak_usemask.values())[0]
        uidx = list(dataset.peak_shares_uididx.values())[0]
        batchidx = list(dataset.peak_shares_batchidx.values())[0]
        tp = list(dataset.peak_usemask.keys())[0]

        params = self.rfim.get_dist_parameters_type(tp, uidx, batchidx, peak_islinear=peak_islinear, update=True)
        mean = params["peak_position_scaled"][0].detach().cpu().numpy()
        covariance = torch.linalg.inv(params["precision"][0].detach()).cpu().numpy()
        shares = params["log_classshares"].softmax(-1)[0].detach().cpu().numpy()
        #scale = params["scale"][0].detach().cpu().numpy()

        ind = pd.Index(self.rfim.channels).get_indexer(channels)
        out_mean = np.arcsinh(mean / self.rfim.class_cofactor[tp])[:,ind]
        out_cov = covariance[:,ind][:,:,ind]
        out_islinear = peak_islinear.detach().cpu().numpy()[0,:,ind].T
        
        #out_mean[~out_islinear] = (out_mean / scale)[~out_islinear]

        return out_mean, out_cov, shares, out_islinear, self.rfim.class_cofactor[tp]
    
    
    def apply_ArcSinh(self, x, tp):
        """ Apply arcsinh for rainbow type tp to data x with same cofactor as used in the fit.
        """
        return np.arcsinh(x / self.rfim.class_cofactor[tp])
    
    
    def add_integrated_rainbow(self, adata, include_shift=True, only_shift=False, addlayer="integrated"):
        """ Add integrated data layer to rainbow bead sample adata.
            Can include the fitted shift.
        """
        adata.layers[addlayer] = adata.layers["raw"].copy()
        
        # optionally add rainbow shift
        if include_shift:
            shift = self.rfim.get_integration_rainbow_shift(adata=adata)
            index = adata.var.index.get_indexer(shift.index)
            adata.layers[addlayer][:,index] = adata.layers[addlayer][:,index] + shift.to_numpy()[None]
        
        if not only_shift:
            factor = self.get_integration_factor(adata=adata)
            index = adata.var.index.get_indexer(factor.index)
            adata.layers[addlayer][:,index] = adata.layers[addlayer][:,index] * factor.to_numpy()[None]
    
    def add_integrated(self, adata, rainbow_batch=None, addlayer="integrated"):
        """ Add integrated data layer to sample adata.
            If given uses 'rainbow_batch', otherwise tries to infer from adata.
            
            As they are usually not used, does not attempt to integrate fluorescence height channels.
        """
        adata.layers[addlayer] = adata.layers["raw"].copy()
        
        factor = self.get_integration_factor(adata=adata, rainbow_batch=rainbow_batch)
        index = adata.var.index.get_indexer(factor.index)
        adata.layers[addlayer][:,index] = adata.layers[addlayer][:,index] * factor.to_numpy()[None]
    
    def add_integration_factor(self, adata, rainbow_batch=None, addkey="rainbow_integration_factor"):
        """ Add integration factors to sample adata.var[addkey].
            If given uses 'rainbow_batch', otherwise tries to infer from adata.
            Defaults to nan if channel was not integrated.
        """
        adata.var[addkey] = np.nan
        intfactor = self.get_integration_factor(adata=adata, rainbow_batch=rainbow_batch)
        adata.var.loc[intfactor.index, addkey] = intfactor

    def make_standardized_fcs(self, filepath, filepath_new, cytoconfig, rainbow_batch=None, force=False, gates=[]):
        """ Read .fcs at filepath, standardize data using the rainbow model, write standardized data to .fcs at filepath_new.
    
            :param filepath: str. Path to the original .fcs file.
    
            :param filepath_new: str. Path to the new .fcs file.
    
            :param cytoconfig: CytometerConfiguration.
    
            :param rainbow_batch: None, str. Either infer batch from file metadata, or give explicitly here.
    
            :param force: bool. Writes a note into files after standardization to make sure it is only applied once, can be overwritten here to enforce additional standardization either way.

            :param gates: iterable. List of gates to apply before the standardization, e.g. to exclude events in the non-linear detector range here as this is not longer possible afterwards.
        """
        markkey = "__cytovanni_rainbow_standardized"
    
        sample = fk.Sample(filepath, cache_original_events=True)
        metadata = sample.metadata
        data_df = sample.as_dataframe(source="orig")
        pns = data_df.columns.get_level_values(1)
        data_df.columns = data_df.columns.get_level_values(0)
        adata = readfcs_sample(filepath, cytoconfig)
        
        if len(gates)>0:
            keepmask = np.all(np.vstack([gate.apply_df(data_df) for gate in gates]), axis=0)
            data_df = data_df.loc[keepmask].copy()
    
        if not force and ((markkey in metadata) and (metadata[markkey]=="True")):
            warnings.warn(f"{filepath} has already been standardized using rainbow beads, skipping the standardization! Override this with force=True")
        else:
            factor = self.get_integration_factor(adata=adata, rainbow_batch=rainbow_batch)
            overlap = data_df.columns.intersection(factor.index)
            data_df[overlap] = data_df[overlap] * factor.loc[overlap].to_numpy()[None]
            metadata[markkey] = "True"
    
        writefcs(filepath_new, data_df, sample_id="", add_metadata=metadata, overwrite_pns=True, pns=pns)

    def get_integrated_spectra(self, spectra, rainbow_batch, spectral=False):
        """ Takes single stain spectra 'spectra', recorded in batch 'rainbow_batch', and standardizes the channels.
            If spectral, normalizes the largest entry per spectrum to one, if not keeps the entry that is currently one at one.

            :param spectra: pd.DataFrame. Single stain spectra, index channel, column dye.
    
            :param rainbow_batch: str. Rainbow bead batch of the spectra.

            :param spectral: bool. Whether to use spectral flow normalization .
        """
        factors = self.get_integration_factor(rainbow_batch=rainbow_batch)
        return scale_spectra_wfactors(spectra, factors, spectral=spectral)
    

    def plot_rainbow_with_PDF(self, adata, channels=None, bins=400):
        """ Plot spectrum of rainbow beads adata.
            Overlays the PDF of the fitted Gaussian.
        """
        if channels is None:
            channels = self.rfim.channels

        self.add_integrated_rainbow(adata, include_shift=True, only_shift=True, addlayer="_integratedos")

        fig, ax = plt.subplots(len(channels), 1, figsize=(8, len(channels)*4))
        if len(channels)<2: ax = [ax]

        mean, covariance, shares, islinear, cofactor = self.get_rainbow_peak_parameters(adata, channels)
        std = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))

        get_data = lambda l, c: np.arcsinh(adata[:,c].layers[l][:,0] / cofactor)

        for i, channel in enumerate(channels):
            ax[i].set_title(f"{channel} Shifted", size=20)
            ax[i].hist(get_data("_integratedos", channel), bins=bins, fill=False, density=True, histtype="step")

            _plot_rfim_pdf(ax[i], i, mean, std, shares, islinear)
