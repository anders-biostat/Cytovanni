"""Rainbow bead calibration: `RainbowCalibrator` fits per-batch channel factors from rainbow bead measurements.

The calibrator combines a scatter model (`RainbowScatterCalibrationModule`) and a fluorescence model
(`RainbowFluorescenceGMMCalibrationModule`) over a shared batch encoder (`RainbowBatchOrdinalEncoder`). Multiplying
raw data of a batch by its calibration factors brings all batches onto a common scale (the 'calibrated' layer).
"""
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
from ..io.utils import _replace_channel_name
from ..exceptions import CalibrationModuleWarning, CalibrationModuleException, MissingChannelException, MissingGPUWarning, CalibrationModuleUntrainedParameterWarning
from .rboe import RainbowBatchOrdinalEncoder
from .rbim_scatter import RainbowScatterCalibrationModule
from .rbim_data import RainbowFluorescenceGMMCalibrationDataset
from .rbim_fluorescence import RainbowFluorescenceGMMCalibrationModule, RainbowFluorescenceGMMCalibrationTrainer
from .utils import scale_spectra_wfactors

import scipy.stats as stats
def _plot_rfim_pdf(ax, i, mean, std, shares, islinear, vline=True):
    """Plot the fitted mixture density (arcsinh scale) of column i onto ax: linear peaks black, nonlinear grey."""
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
    """Geometric mean of the positive array x."""
    return np.exp(np.log(x).mean())

def _restrict_to_available(values, adata):
    """Restrict a per-channel Series (factors or shifts) to the channels in adata.var; warns about the dropped ones."""
    missing = values.index.difference(adata.var.index)
    if len(missing)>0:
        warnings.warn(f"Calibration available for channels {missing.tolist()} that are not in the sample, skipping them.", CalibrationModuleWarning)
    return values.loc[values.index.intersection(adata.var.index, sort=False)]


class RainbowCalibrator():
    """Calibration of detector channels across batches with rainbow beads.

    Rainbow beads have fixed fluorescence and scatter intensities, so differences between their measurements reflect
    changes of the cytometer (PMT voltages, laser power, alignment). From all rainbow bead measurements, the
    calibrator fits one multiplicative factor per batch and channel; multiplying the raw data of a batch by them
    brings all batches onto a common scale (``layers['calibrated']``, see `add_calibrated`).

    A batch ('rainbow_batch') is a period with unchanged cytometer settings (same 'set_hash') on one cytometer, with one
    or more rainbow bead measurements (each with its own 'uid'). Several bead types ('rainbow_type', e.g. lots) can be
    combined if they are linked by a batch in which both were measured.

    Constructing the calibrator from data sets up the batch encoder, fits the scatter model at once (closed form) and
    prepares the fluorescence model, which is then fitted with `fit_fluorescence`.

    Parameters
    ----------
    adatas : list of anndata.AnnData, optional
        Scatter-gated rainbow bead measurements, each after `fit_rainbow_GMM` (``obs['GMM_label']``,
        ``uns['GMM_means']``, ``uns['GMM_asinh_cofactor']``), with ``layers['raw']``, ``uns['cytoconfig']`` and the
        metadata 'cytometer', 'rainbow_batch', 'rainbow_type', 'uid' and 'set_hash' in ``uns`` (as set by
        `cytovanni.io.readfcs_rainbow_sample`); 'UTC' and 'date' are used if present. All need the fluorescence
        channels of the first measurement's ``cytoconfig``. Default: an empty calibrator, as used by `from_saved`.
    include_shift : bool, default True
        Fit a shift per measurement and channel in the fluorescence model (see
        `RainbowFluorescenceGMMCalibrationModule`).

    Attributes
    ----------
    adatas : list of anndata.AnnData
        Training data (not set after `from_saved`).
    rboe : RainbowBatchOrdinalEncoder
        Encoder of cytometers, batches, bead types and measurements.
    rsim : RainbowScatterCalibrationModule
        Scatter model.
    rfim : RainbowFluorescenceGMMCalibrationModule
        Fluorescence model.
    channel_missing_cutoff : float or None
        Value used in the last `fit_fluorescence` (None before).

    Raises
    ------
    CalibrationModuleException
        If a measurement lacks one of the required ``uns``, ``obs`` or layer entries (each missing key is also listed
        in a `CalibrationModuleWarning`), or the number of peaks differs between measurements of one bead type.
    MissingChannelException
        If a measurement lacks one of the fluorescence channels or the scatter channels to calibrate.
    InconsistentBatchException, RepeatedBatchException
        If a batch contains different settings hashes, or a batch name occurs on several cytometers.

    See Also
    --------
    fit_rainbow_GMM : per-measurement peak assignment needed as input.
    plot_eval_RBInt_fit : check the fit.

    Notes
    -----
    Fluorescence model (`RainbowFluorescenceGMMCalibrationModule`): every bead type is a mixture with one component per
    peak; on the arcsinh scale, ``arcsinh((x + s) / f)`` of a bead of peak p in measurement m of batch b is normal with
    mean ``arcsinh(mu_p / (Gamma_b f))`` and a covariance per peak and measurement; the peak shares are fitted per
    measurement. ``mu_p`` is the peak position of the bead type, ``Gamma_b`` the calibration factor of the batch (per
    channel), ``s`` a shift per measurement and channel, f the arcsinh cofactor of the bead type (geometric mean of the
    measurements' ``GMM_asinh_cofactor``). ``ln Gamma`` is constrained to mean 0 per channel over the batches of the
    first fit, which fixes the common scale. Entries of these batches that could not be trained (see
    `fit_fluorescence`) still enter this mean. Peaks whose GMM mean is above
    ``cytoconfig.max_linear_range`` get a free position per batch and no shift, so they do not inform the factors.

    Scatter model (`RainbowScatterCalibrationModule`): the log median of every scatter channel per measurement is a
    position per bead type plus a log shift per batch, fitted by weighted least squares.

    When the model is extended (`extend_fit`), the parameters of the known batches and bead types are frozen and
    only the new ones are fitted, so earlier calibrated data stay valid.

    Examples
    --------
    >>> rbint = RainbowCalibrator(adatas)
    >>> rbint.fit_fluorescence(gpu=True)
    >>> rbint.save("rainbow_calibrator.pt")
    >>> rbint.add_calibrated(adata_sample)  # adds adata_sample.layers["calibrated"]
    """
    REQUIRED_UNS = ["cytometer", "rainbow_batch", "rainbow_type", "uid", "set_hash"]
    REQUIRED_GMM_UNS = ["GMM_means", "GMM_asinh_cofactor"]
    REQUIRED_GMM_OBS = ["GMM_label"]
    REQUIRED_LAYER = ["raw"]
    channel_missing_cutoff = None
    
    def __init__(self, adatas=None, include_shift=True):
        if adatas is not None:
            self._check_adatas(adatas)
            self.adatas = adatas

            # Initialise batch encoder
            df_meta = self._process_adatas_meta(adatas)
            self.rboe = RainbowBatchOrdinalEncoder.from_data(df_meta)

            # Initialize and fit scatter calibration
            self.rsim = RainbowScatterCalibrationModule.from_data(self.rboe, self.adatas)

            # Initialize fluorescence calibration
            dct_Npeak = RainbowFluorescenceGMMCalibrationDataset.get_type_Npeak(self.adatas)
            channels = self.adatas[0].uns["cytoconfig"].channels_fluorescence # have to be the same across all adatas anyway
            self._check_adatas_channels(self.adatas, channels) # make sure they are all present
            class_cofactor = {tp: _geometric_mean(df_meta.loc[df_meta["rainbow_type"]==tp, "GMM_asinh_cofactor"].to_numpy().astype(float))
                              for tp in df_meta["rainbow_type"].unique()} # geometric mean of cofactors across each type
            self.rfim = RainbowFluorescenceGMMCalibrationModule(self.rboe, dct_Npeak, channels, class_cofactor, include_shift=include_shift)
    
    
    @classmethod
    def _check_adatas(cls, adatas):
        """Raise CalibrationModuleException if some adatas lack a required uns/obs/layer entry (warns per key)."""
        abort = False
        
        for key in cls.REQUIRED_UNS:
            missing = sum([key not in ad.uns for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing .uns metadata key {key}!", CalibrationModuleWarning)
        
        for key in cls.REQUIRED_GMM_UNS:
            missing = sum([key not in ad.uns for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing .uns metadata key {key}! Are you sure they were set up with fit_rainbow_GMM?", CalibrationModuleWarning)
        
        for key in cls.REQUIRED_GMM_OBS:
            missing = sum([key not in ad.obs for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing .obs metadata key {key}! Are you sure they were set up with fit_rainbow_GMM?", CalibrationModuleWarning)
        
        for key in cls.REQUIRED_LAYER:
            missing = sum([key not in ad.layers for ad in adatas])
            if missing>0:
                abort = True
                warnings.warn(f"{missing} passed adatas are missing the 'raw' data in .layers!", CalibrationModuleWarning)
        
        if abort:
            raise CalibrationModuleException("Passed adatas are missing some initialisations!")
    
    @classmethod
    def _check_adatas_channels(cls, adatas, channels):
        """Raise MissingChannelException if some adatas lack one of `channels` in adata.var."""
        missing = 0
        for ad in adatas:
            if not np.all(np.isin(channels, ad.var.index)):
                missing += 1
        if missing>0:
            raise MissingChannelException(f"{missing} passed adatas are missing some of the required fluorescence channels {channels}!")
    
    @classmethod
    def _process_adatas_meta(self, adatas):
        """Metadata table of adatas (one row per measurement) from their uns, as needed by the batch encoder."""
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
    
    
    def fit_fluorescence(self, gpu=True, device="cuda:0", verbosity=2, freeze=True, channel_missing_cutoff=-np.inf,
                         cov_penalty=0., cov_penalty_broaden_only=True):
        """Fit the fluorescence model on the training data.

        Builds a `RainbowFluorescenceGMMCalibrationDataset` and runs `RainbowFluorescenceGMMCalibrationTrainer` with its
        default settings: initial fit of the peak positions (SGD on the GMM means), of the covariances (Adam), then
        the full likelihood fit (Adam, 10000 steps on subsamples of at most 1000 beads per measurement).

        Parameters
        ----------
        gpu : bool, default True
            Fit on `device`. If no CUDA device is available, falls back to the CPU with a `MissingGPUWarning` (much
            slower; torch uses one thread unless set with ``torch.set_num_threads``).
        device : str, default "cuda:0"
            Torch device used if `gpu`; ignored otherwise (CPU).
        verbosity : int, default 2
            0 silent, 1 progress messages and bars, 2 also the loss plots of the trainer.
        freeze : bool, default True
            Freeze all parameters after the fit. Freezing cannot be undone: turn it off only to continue training the
            same batches with another call; it is needed for `extend_fit` to keep the known batches fixed.
        channel_missing_cutoff : float, default -inf
            Channels in which every peak of a measurement has a GMM mean below this value (raw units) are treated like
            off-scale peaks (not used for the factors) in that measurement, e.g. a laser that was switched off. The
            default uses all channels.
        cov_penalty : float, default 0
            Weight of a penalty that keeps the peak covariances of every measurement close to their values after the
            initial fit (`RainbowFluorescenceGMMCalibrationTrainer.get_cov_penalty`). Without it, a peak can broaden
            over neighbouring peaks, after which the measurement's factors can drift at little cost in likelihood.
            0 (no penalty) is fine for normal rainbow beads. QC beads that do not fluoresce as much in every channel
            (few bright peaks per laser) may need it; 0.3 with `cov_penalty_broaden_only` works there.
        cov_penalty_broaden_only : bool, default True
            Penalise only broadening of the peak covariances, not narrowing.

        Returns
        -------
        None
            Sets `dataset` and `rfim_trainer` and fits `rfim` in place (moved to the device).

        Raises
        ------
        CalibrationModuleException
            If the calibrator has no training data (loaded with `from_saved`; use `extend_fit` to add data).

        Warns
        -----
        CalibrationModuleWarning
            If `channel_missing_cutoff` differs from the previous fit, or some channels cannot be trained in some
            batches (all peaks off-scale or below `channel_missing_cutoff`; their factors are set to 1 later).
        MissingGPUWarning
            If `gpu` but no CUDA device is available.
        """
        if not hasattr(self, "adatas"):
            raise CalibrationModuleException("RainbowCalibrator was loaded from saved and has no training data attached! If you want to extend the model use .extend() instead!")
        
        if gpu and not torch.cuda.is_available():
            warnings.warn("Tried using GPU but torch doesn't find any available devices, falling back to CPU at much lower training speed!\n Use 'import torch; torch.set_num_threads(N)' before fitting to set the number of CPU cores torch should use, defaults to using only one.", MissingGPUWarning)
            gpu = False
        if not gpu:
            device = "cpu"
        if self.channel_missing_cutoff is not None and self.channel_missing_cutoff!=channel_missing_cutoff:
            warnings.warn("Using a different 'channel_missing_cutoff' than was used for last training!", CalibrationModuleWarning)
        self.channel_missing_cutoff = channel_missing_cutoff
        
        if verbosity>=1: printwtime("Initializing Dataset")
        self.dataset = RainbowFluorescenceGMMCalibrationDataset(self.rboe, self.adatas, device=device, channel_missing_cutoff=self.channel_missing_cutoff)
        self.rfim.to(device)
        self.rfim_trainer = RainbowFluorescenceGMMCalibrationTrainer(self.rfim, cov_penalty=cov_penalty,
                                                                     cov_penalty_broaden_only=cov_penalty_broaden_only)
        
        if verbosity>=1: printwtime("Fit Initialization")
        self.rfim_trainer.fit_init(self.dataset, showprogress=verbosity>=1)
        if verbosity>=2: self.rfim_trainer.plot_fit_init()
        
        if verbosity>=1: printwtime("Full Model Fit")
        self.rfim_trainer.fit(self.dataset, showprogress=verbosity>=1)
        if freeze: self.rfim.freeze()
        if verbosity>=2: self.rfim_trainer.plot_fit()
    
    
    def extend_fit(self, adatas, **fitkwargs):
        """Extend the model to new rainbow bead measurements and fit them, keeping all known parameters fixed.

        New batches, bead types and measurements are added to the encoder and both models; the parameters present
        before are frozen, the scatter model is refitted (closed form) and the fluorescence model is fitted with
        `fit_fluorescence` on the new data only. Factors of the known batches do not change. Works on a calibrator
        loaded with `from_saved`.

        Parameters
        ----------
        adatas : list of anndata.AnnData
            New rainbow bead measurements, prepared as for the constructor. They replace `adatas`: the previous
            training data are not used. Measurements in known batches are possible (their batch factors stay fixed).
        **fitkwargs
            Passed to `fit_fluorescence`.

        Returns
        -------
        None

        Raises
        ------
        CalibrationModuleException
            Before anything is changed: if a measurement lacks required entries, or a new bead type has no link to the
            trained data (it must be measured in a known batch, or in a new batch together with a known type; see
            `RainbowBatchOrdinalEncoder.check_orphan_types`), since its factors would be arbitrary. After the encoder
            was extended: if a known bead type has a different number of peaks.
        MissingChannelException
            If a measurement lacks one of the channels.

        Examples
        --------
        >>> rbint = RainbowCalibrator.from_saved("rainbow_calibrator.pt")
        >>> rbint.extend_fit(new_adatas)
        >>> rbint.save("rainbow_calibrator_extended.pt")
        """
        self._check_adatas(adatas)
        df_meta = self._process_adatas_meta(adatas)
        self.rboe.check_orphan_types(df_meta)
        self.adatas = adatas

        # Extend batch encoder
        extend_inds = self.rboe.extend(df_meta)
        
        # Extend and fit scatter calibrator
        self.rsim.extend(self.rboe, extend_inds, self.adatas)

        # Extend fluorescence calibrator, freeze previous parameters in case this was not done previously
        dct_Npeak = RainbowFluorescenceGMMCalibrationDataset.get_type_Npeak(self.adatas)
        channels = self.adatas[0].uns["cytoconfig"].channels_fluorescence # have to be the same across all adatas anyway
        self._check_adatas_channels(self.adatas, channels) # make sure they are all present
        class_cofactor = {tp: _geometric_mean(df_meta.loc[df_meta["rainbow_type"]==tp, "GMM_asinh_cofactor"].to_numpy().astype(float))
                          for tp in df_meta["rainbow_type"].unique()} # geometric mean of cofactors across each type
        self.rfim.freeze()
        self.rfim.extend(self.rboe, extend_inds, dct_Npeak=dct_Npeak, class_cofactor=class_cofactor)
        
        # Fit extended fluorescence calibrator
        self.fit_fluorescence(**fitkwargs)
    
    
    def export(self):
        """Export the calibrator (encoder, both models, `channel_missing_cutoff`; no training data) as a dict.

        Returns
        -------
        dict
            Keys 'rboe', 'rsim', 'rfim', 'channel_missing_cutoff'; the input of `from_saved` after ``torch.save``.
        """
        exported = {}
        exported["rboe"] = self.rboe.export()
        exported["rsim"] = self.rsim.export()
        exported["rfim"] = self.rfim.export()
        exported["channel_missing_cutoff"] = self.channel_missing_cutoff
        return exported
    
    def save(self, filepath, make_path_structure=False):
        """Save the calibrator (`export`, without training data) with ``torch.save``.

        Parameters
        ----------
        filepath : str
            Output path, by convention ending in '.pt'.
        make_path_structure : bool, default False
            Create missing parent folders. Off by default, so that a mistyped path raises an error.

        Returns
        -------
        None
        """
        exported = self.export()
        if make_path_structure:
            Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True) # ensure path exists
        torch.save(exported, filepath)
    
    @classmethod
    def from_saved(cls, filepath):
        """Load a calibrator saved with `save`.

        Parameters
        ----------
        filepath : str
            Path of the saved file (loaded onto the CPU, ``weights_only=False``: load only trusted files).

        Returns
        -------
        RainbowCalibrator
            Calibrator without training data: it can calibrate data and be extended with `extend_fit`, but not refitted
            with `fit_fluorescence`.
        """
        exported = torch.load(filepath, map_location=torch.device("cpu"), weights_only=False)
        rbint = cls()
        rbint.rboe = RainbowBatchOrdinalEncoder.from_exported(exported["rboe"])
        rbint.rsim = RainbowScatterCalibrationModule.from_exported(exported["rsim"], rbint.rboe)
        rbint.rfim = RainbowFluorescenceGMMCalibrationModule.from_exported(exported["rfim"], rbint.rboe)
        rbint.channel_missing_cutoff = exported["channel_missing_cutoff"]
        return rbint
    
    
    def get_calibration_factor(self, rainbow_batch=None, adata=None):
        """Calibration factors of one batch: multiply its raw data by them to bring it onto the common scale.

        To take data from batch 1 to batch 2 instead, multiply by ``factors1 / factors2``.

        Parameters
        ----------
        rainbow_batch : str, optional
            Batch name. Takes precedence over `adata`; if it is unknown and `adata` is given, the batch is taken from
            `adata` (with an `OrdinalEncoderWarning`).
        adata : anndata.AnnData, optional
            Sample whose batch is used: ``uns['rainbow_batch']`` if it is a known batch, otherwise the batch is inferred
            from 'cytometer', 'date', 'UTC' and the settings hash (`RainbowBatchOrdinalEncoder.find_closest_batch`).

        Returns
        -------
        pandas.Series
            Factor per channel: the calibrated scatter channels (e.g. FSC-A and FSC-H with the factor fitted on FSC-A,
            see ``cytoconfig.get_default_scatter_calibration``), then the fluorescence channels of the model (area
            channels). Fluorescence factors that were not trained for this batch are set to 1.

        Raises
        ------
        OrdinalEncoderException, AutoBatchException
            If neither argument is given, or the batch cannot be found or inferred.

        Warns
        -----
        CalibrationModuleUntrainedParameterWarning
            If some fluorescence factors of the batch were not trained (set to 1, not trustworthy).
        """
        factor_scatter = self.rsim.get_calibration_factor(rainbow_batch=rainbow_batch, adata=adata)
        factor_fluorescence = self.rfim.get_calibration_factor(rainbow_batch=rainbow_batch, adata=adata, fix_untrained=True, warn_untrained=True, return_mask=False)
        factor = pd.concat([factor_scatter, factor_fluorescence])
        return factor
    
    def get_calibration_factor_all_single(self, rainbow_batch):
        """Factors of one batch as in `get_calibration_factor`, without warning, plus the fluorescence trained-mask.

        Parameters
        ----------
        rainbow_batch : str
            Batch name.

        Returns
        -------
        tuple of pandas.Series
            Factors (untrained fluorescence factors set to 1) and boolean mask of the trained fluorescence factors
            (index: fluorescence channels).
        """
        factor_scatter = self.rsim.get_calibration_factor(rainbow_batch=rainbow_batch)
        factor_fluorescence, mask_trained_fluorescence = self.rfim.get_calibration_factor(rainbow_batch=rainbow_batch, fix_untrained=True, warn_untrained=False, return_mask=True)
        factor = pd.concat([factor_scatter, factor_fluorescence])
        return factor, mask_trained_fluorescence
    
    def get_calibration_factors_all(self, warn_untrained=True):
        """Calibration factors of all batches of the model.

        Parameters
        ----------
        warn_untrained : bool, default True
            Warn (`CalibrationModuleUntrainedParameterWarning`, one summary) if some fluorescence factors were not
            trained; they are set to 1.

        Returns
        -------
        pandas.DataFrame
            Shape (batch, channel): index the batch names (encoder order), columns as in `get_calibration_factor`.
            Over the batches of the first fit, the factors have geometric mean 1 per channel.
        """
        outs = [self.get_calibration_factor_all_single(rainbow_batch=label) for label in self.rboe.oes["rainbow_batch"].labels]
        factors = pd.DataFrame([o[0] for o in outs],
             index=self.rboe.oes["rainbow_batch"].labels)
        mask = pd.DataFrame([o[1] for o in outs])
        
        if warn_untrained and ((~mask).sum().sum()>0):
            warnstr = "For some batches, some scaling factors were not trained! They are set to 1."
            n_channel, Nc = ((~mask.to_numpy()).sum(0)>0).sum(), mask.shape[1]
            n_batch, Nb = ((~mask.to_numpy()).sum(1)>0).sum(), mask.shape[0]
            warnstr+= f"\n{n_channel} (out of {Nc}) channels are not trained in some batches, {n_batch} (out of {Nb}) batches have some untrained channels."
            warnings.warn(warnstr, CalibrationModuleUntrainedParameterWarning)
        
        return factors
    
    def get_rainbow_shifts_all(self):
        """Fitted shifts of all rainbow bead measurements, per bead type.

        Returns
        -------
        dict of str to pandas.DataFrame
            Per rainbow type a DataFrame of shape (measurement, channel): index the measurement uids, columns the
            fluorescence channels, values in raw units (see
            `RainbowFluorescenceGMMCalibrationModule.get_calibration_rainbow_shift`).
        """
        outdct = {}
        df = self.rboe.data_df[["uid", "rainbow_type"]]
        for tp in self.rboe.oes["rainbow_type"].labels:
            df_tp = df[df["rainbow_type"]==tp]
            shifts = pd.DataFrame([self.rfim.get_calibration_rainbow_shift(uid=row["uid"], tp=row["rainbow_type"])
                                  for name, row in df_tp.iterrows()], index=df_tp["uid"].to_numpy())
            outdct[tp] = shifts
        
        return outdct
    
    def get_rainbow_peak_parameters(self, adata, channels):
        """Fitted mixture parameters of one rainbow bead measurement, as used in the fluorescence model.

        They describe ``arcsinh((x + s) / f)``, the raw data x after adding the fitted shift s (raw units), with the
        cofactor f of the bead type: means and covariances of the peaks on this arcsinh scale, peak shares, and which
        peaks count as linear. For nonlinear peaks the mean is the free per-batch position. Works wherever the
        fluorescence module is (CPU, or GPU after ``fit_fluorescence(gpu=True)``).

        Parameters
        ----------
        adata : anndata.AnnData
            Rainbow bead measurement known to the model (its uid, batch and rainbow type), prepared as for the
            constructor.
        channels : list of str
            Fluorescence channels to return, in this order.

        Returns
        -------
        mean : numpy.ndarray
            Shape (peak, channel), arcsinh scale.
        covariance : numpy.ndarray
            Shape (peak, channel, channel), arcsinh scale.
        shares : numpy.ndarray
            Shape (peak,), fitted peak shares (sum 1).
        islinear : numpy.ndarray
            Boolean, shape (peak, channel): peak used for the calibration (GMM mean below ``max_linear_range`` and
            channel not missing).
        cofactor : float
            Arcsinh cofactor f of the bead type (raw units).

        Raises
        ------
        CalibrationModuleException
            If the batch, type or uid of `adata` is not in the model.
        """
        # the dataset has to live on the device of the module (cuda after a GPU fit, cpu after from_saved)
        device = next(self.rfim.parameters()).device
        dataset = RainbowFluorescenceGMMCalibrationDataset(self.rboe, [adata], device=device, warn_missing=False,
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
        """Return ``arcsinh(x / f)`` with the cofactor f (raw units) of rainbow type `tp` in the fluorescence model.

        Parameters
        ----------
        x : numpy.ndarray
            Data on the raw (or calibrated) scale.
        tp : str
            Rainbow type.

        Returns
        -------
        numpy.ndarray
            Same shape as `x`.
        """
        return np.arcsinh(x / self.rfim.class_cofactor[tp])
    
    
    def add_calibrated_rainbow(self, adata, include_shift=True, only_shift=False, addlayer="calibrated"):
        """Add a calibrated layer to a rainbow bead measurement known to the model, optionally with its fitted shift.

        The layer is ``(raw + shift) * factor`` per channel, with the shift of the measurement (raw units, fluorescence
        channels only) and the factors of its batch. Channels without shift or factor are copied from ``layers['raw']``.

        Parameters
        ----------
        adata : anndata.AnnData
            Rainbow bead measurement with ``layers['raw']``, whose uid and rainbow type are in the model. Changed in
            place: ``layers[addlayer]`` is set (overwritten).
        include_shift : bool, default True
            Add the fitted shift before multiplying by the factors.
        only_shift : bool, default False
            Only add the shift, do not apply the factors.
        addlayer : str, default "calibrated"
            Name of the layer to write.

        Returns
        -------
        None

        Raises
        ------
        CalibrationModuleException
            If the shift is requested and the uid or rainbow type of `adata` is not in the model.

        Warns
        -----
        CalibrationModuleWarning
            If the model has channels that are not in `adata` (skipped).
        """
        adata.layers[addlayer] = adata.layers["raw"].copy()
        
        # optionally add rainbow shift
        if include_shift:
            shift = _restrict_to_available(self.rfim.get_calibration_rainbow_shift(adata=adata), adata)
            index = adata.var.index.get_indexer(shift.index)
            adata.layers[addlayer][:,index] = adata.layers[addlayer][:,index] + shift.to_numpy()[None]
        
        if not only_shift:
            factor = _restrict_to_available(self.get_calibration_factor(adata=adata), adata)
            index = adata.var.index.get_indexer(factor.index)
            adata.layers[addlayer][:,index] = adata.layers[addlayer][:,index] * factor.to_numpy()[None]
    
    def add_calibrated(self, adata, rainbow_batch=None, addlayer="calibrated"):
        """Add the calibrated layer ``raw * factor`` to a sample.

        Factors as in `get_calibration_factor`: the calibrated scatter channels and the fluorescence (area) channels of
        the model. Fluorescence height channels and all other channels are copied from ``layers['raw']`` unchanged.

        Parameters
        ----------
        adata : anndata.AnnData
            Sample with ``layers['raw']``. Changed in place: ``layers[addlayer]`` is set (overwritten).
        rainbow_batch : str, optional
            Batch of the sample. Default: taken from `adata` (``uns['rainbow_batch']``, or inferred from its date, time
            and settings, see `get_calibration_factor`).
        addlayer : str, default "calibrated"
            Name of the layer to write.

        Returns
        -------
        None

        Raises
        ------
        OrdinalEncoderException, AutoBatchException
            If the batch cannot be found or inferred.

        Warns
        -----
        CalibrationModuleWarning
            If the model has channels that are not in `adata` (skipped).
        CalibrationModuleUntrainedParameterWarning
            If some fluorescence factors of the batch were not trained (set to 1).

        Examples
        --------
        >>> rbint.add_calibrated(adata, rainbow_batch="12_11_2021")
        >>> adata.layers["calibrated"]
        """
        adata.layers[addlayer] = adata.layers["raw"].copy()
        
        factor = _restrict_to_available(self.get_calibration_factor(adata=adata, rainbow_batch=rainbow_batch), adata)
        index = adata.var.index.get_indexer(factor.index)
        adata.layers[addlayer][:,index] = adata.layers[addlayer][:,index] * factor.to_numpy()[None]
    
    def add_calibration_factor(self, adata, rainbow_batch=None, addkey="rainbow_calibration_factor"):
        """Store the calibration factors of the sample's batch in ``adata.var[addkey]``.

        Parameters
        ----------
        adata : anndata.AnnData
            Sample. Changed in place: ``var[addkey]`` is set, NaN for channels without a factor.
        rainbow_batch : str, optional
            Batch of the sample. Default: taken from `adata`, see `get_calibration_factor`.
        addkey : str, default "rainbow_calibration_factor"
            Column of ``adata.var`` to write.

        Returns
        -------
        None
        """
        adata.var[addkey] = np.nan
        intfactor = _restrict_to_available(self.get_calibration_factor(adata=adata, rainbow_batch=rainbow_batch), adata)
        adata.var.loc[intfactor.index, addkey] = intfactor

    def make_calibrated_fcs(self, filepath, filepath_new, cytoconfig, rainbow_batch=None, force=False, gates=[]):
        """Write a calibrated copy of an .fcs file.

        Reads the raw data with FlowKit, multiplies the calibrated channels by the factors of the file's batch
        (`get_calibration_factor`) and writes a new .fcs file with the original metadata and channel names. The
        metadata key ``__cytovanni_rainbow_calibrated`` is set to "True", and a file that already has it is not
        calibrated again (unless `force`).

        Parameters
        ----------
        filepath : str
            Path of the original .fcs file.
        filepath_new : str
            Path of the new .fcs file.
        cytoconfig : CytometerConfiguration
            Configuration of the cytometer; maps the file's channel names to the model's and is used to read the
            metadata for the batch inference.
        rainbow_batch : str, optional
            Batch of the file. Default: inferred from the file's metadata (date, time, settings).
        force : bool, default False
            Calibrate even if the file is marked as calibrated already.
        gates : list of gates, default []
            Gates (with ``apply_df``, on the raw data with the model's channel names) applied before the calibration;
            only the events inside all gates are written. E.g. to remove events outside the linear detector range,
            which cannot be identified after the calibration.

        Returns
        -------
        None

        Raises
        ------
        MissingChannelException
            If none of the calibrated channels is in the file.

        Warns
        -----
        UserWarning
            If the file is already marked as calibrated and `force` is False; the data are then written uncalibrated
            (gates still applied).
        """
        markkey = "__cytovanni_rainbow_calibrated"
    
        sample = fk.Sample(filepath, preprocess=False)
        metadata = sample.metadata
        data_df = sample.as_dataframe(source="raw")
        pns = data_df.columns.get_level_values(1)
        pnn_raw = data_df.columns.get_level_values(0)
        # work on the canonical channel names of cytoconfig (as readfcs_sample does), write back with the original $PnN names
        data_df.columns = [_replace_channel_name(c, cytoconfig) for c in pnn_raw]
        adata = readfcs_sample(filepath, cytoconfig)
        
        if len(gates)>0:
            keepmask = np.all(np.vstack([gate.apply_df(data_df) for gate in gates]), axis=0)
            data_df = data_df.loc[keepmask].copy()
    
        if not force and ((markkey in metadata) and (metadata[markkey]=="True")):
            warnings.warn(f"{filepath} has already been calibrated using rainbow beads, skipping the calibration! Override this with force=True")
        else:
            factor = self.get_calibration_factor(adata=adata, rainbow_batch=rainbow_batch)
            overlap = data_df.columns.intersection(factor.index)
            if len(overlap)==0:
                raise MissingChannelException(f"None of the calibrated channels {factor.index.tolist()} are in {filepath} (channels {data_df.columns.tolist()})!")
            data_df[overlap] = data_df[overlap] * factor.loc[overlap].to_numpy()[None]
            metadata[markkey] = "True"
    
        data_df.columns = pnn_raw
        writefcs(filepath_new, data_df, sample_id="", add_metadata=metadata, overwrite_pns=True, pns=pns)

    def get_calibrated_spectra(self, spectra, rainbow_batch, spectral=False):
        """Calibrate single-stain spectra recorded in one batch and normalise them again.

        Multiplies every channel of the spectra by the calibration factor of `rainbow_batch` and renormalises each
        spectrum, see `scale_spectra_wfactors`.

        Parameters
        ----------
        spectra : pandas.DataFrame
            Spectra of shape (channel, dye) on the raw scale of the batch; every channel needs a factor.
        rainbow_batch : str
            Batch in which the spectra were recorded.
        spectral : bool, default False
            Normalisation: if True the largest entry of every spectrum is 1 (spectral flow); if False the entry that
            is 1 in the input stays 1 (conventional flow, the detector of the dye).

        Returns
        -------
        pandas.DataFrame
            Calibrated spectra, shape (channel, dye).
        """
        factors = self.get_calibration_factor(rainbow_batch=rainbow_batch)
        return scale_spectra_wfactors(spectra, factors, spectral=spectral)
    

    def plot_rainbow_with_PDF(self, adata, channels=None, bins=400):
        """Plot the histograms of a rainbow bead measurement with the fitted mixture density overlaid.

        One panel per channel: ``arcsinh((raw + s) / f)`` of the beads (shift s of the measurement, cofactor f of its
        type), the fitted density of the linear peaks (black) and the nonlinear peaks (grey), and the fitted peak
        positions (grey vertical lines).

        Parameters
        ----------
        adata : anndata.AnnData
            Rainbow bead measurement known to the model. Changed in place: the layer '_calibratedos' (raw plus shift)
            is added.
        channels : list of str, optional
            Fluorescence channels to plot. Default: all channels of the model.
        bins : int, default 400
            Number of histogram bins.

        Returns
        -------
        None
        """
        if channels is None:
            channels = self.rfim.channels

        self.add_calibrated_rainbow(adata, include_shift=True, only_shift=True, addlayer="_calibratedos")

        fig, ax = plt.subplots(len(channels), 1, figsize=(8, len(channels)*4))
        if len(channels)<2: ax = [ax]

        mean, covariance, shares, islinear, cofactor = self.get_rainbow_peak_parameters(adata, channels)
        std = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))

        get_data = lambda l, c: np.arcsinh(adata[:,c].layers[l][:,0] / cofactor)

        for i, channel in enumerate(channels):
            ax[i].set_title(f"{channel} Shifted", size=20)
            ax[i].hist(get_data("_calibratedos", channel), bins=bins, fill=False, density=True, histtype="step")

            _plot_rfim_pdf(ax[i], i, mean, std, shares, islinear)
