"""Fluorescence part of the rainbow calibration: joint mixture model of all rainbow measurements, and its trainer."""
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter, ParameterDict
from torch.distributions.multivariate_normal import _precision_to_scale_tril
import torch.nn.functional as F
import warnings
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import silent_log
from ..torch import diffclamp
from ..torch import ExtendableParameter, CovarianceTrilExtendableParameter, PrecisionExtendableParameter
from ..torch import ASinhMultivariateNormal
from ..exceptions import MissingChannelException, CalibrationModuleException, CalibrationModuleWarning, CalibrationModuleUntrainedParameterWarning
from .rboe import RainbowBatchOrdinalEncoder
from ..torch import pdsave_expand, pdsave_collect


class RainbowFluorescenceGMMCalibrationModule(Module):
    """Joint mixture model of the fluorescence of all rainbow bead measurements, with per-batch channel factors.

    Every rainbow type is a mixture with one component per peak. With the cofactor f of the type, the raw data x of a
    bead in measurement m (batch b) and the shift s_m, ``arcsinh(x / f + sinh(a_m))`` is normal for every peak p, with
    mean ``arcsinh(mu_p * g_b / f)`` and precision ``A A^T`` (per measurement and peak, arcsinh scale). The fitted
    parameters are

    - ``mu_p = exp(param_peakpos)``: peak positions of the type (raw units), shape (peak, channel);
    - ``g_b = exp(param_logscalefactor)``: gain of batch b, shape (batch, channel), ``ln g`` constrained to mean 0 per
      channel over the batches of the first fit; the calibration factor is ``1 / g_b``;
    - ``a_m = param_shift_asinh``: shift per measurement (uid) and channel, ``s_m = sinh(a_m) * f`` in raw units
      (trainable only with `include_shift`);
    - per type: the precision factor A per (uid, peak) (`PrecisionExtendableParameter`, initialised to 5 I), the
      peak share logits per (uid, peak), and free positions ``exp(param_peakpos_nonlinear)`` per (batch, peak).

    Peaks that are not usable (above ``max_linear_range`` or in a missing channel, see
    `RainbowFluorescenceGMMCalibrationDataset.peak_usemask`) use the free per-batch position instead of
    ``mu_p * g_b`` and no shift, so they do not inform the factors. The likelihood includes the Jacobian of the
    arcsinh transform unless `log_prob_onbase`.

    Parameters
    ----------
    rboe : RainbowBatchOrdinalEncoder
        Batch encoder with all batches, types and uids of the training data.
    dct_Npeak : dict of str to int
        Number of peaks per rainbow type.
    channels : list of str
        Fluorescence channels of the model (order of all channel axes).
    class_cofactor : dict of str to float
        Arcsinh cofactor (raw units) per rainbow type.
    include_shift : bool, default True
        Fit the shift per measurement and channel; if False it stays 0.
    log_prob_onbase : bool, default False
        Compute the likelihood of the underlying normal distribution only (no Jacobian), i.e. as for a normal
        distribution on the arcsinh-transformed data instead of the transformed distribution on the raw scale.

    Attributes
    ----------
    df_batch_trained : pandas.DataFrame
        Boolean, shape (batch, channel), index the batch ordinals: factors that were trained (see
        `remember_trainable_batches`).
    frozen_batches : list of int
        Batch ordinals whose parameters are frozen.
    is_trained : bool
        True once the full fit has started.

    See Also
    --------
    RainbowCalibrator : user interface (builds, fits and applies this module).
    RainbowFluorescenceGMMCalibrationTrainer : training schedule.
    """
    def __init__(self, rboe, dct_Npeak, channels, class_cofactor,
                 include_shift=True, log_prob_onbase=False):
        super().__init__()
        self.rboe = rboe
        self.dct_Npeak = dct_Npeak
        self.channels = channels
        self.Nchannels = len(channels)
        self.class_cofactor = class_cofactor
        self.include_shift = include_shift
        self.is_trained = False
        
        self.log_prob_onbase = log_prob_onbase
        
        self.frozen_batches = []
        
        self.init_params()
    
    def init_params_tp(self, tp):
        """Create the parameters of type `tp`: precision factors, peak positions, nonlinear positions, share logits.

        Parameters
        ----------
        tp : str
            Rainbow type (a key of `dct_Npeak`).
        """
        # covariance decomposition, (uid, peak, channel, channel) for simplicity even though most entries will be unused
        #self.param_cov[tp] = CovarianceTrilExtendableParameter((len(self.rboe.oes["uid"]), self.dct_Npeak[tp], self.Nchannels, self.Nchannels), 0, .2, allow_grad=True)
        self.param_cov[tp] = PrecisionExtendableParameter((len(self.rboe.oes["uid"]), self.dct_Npeak[tp], self.Nchannels, self.Nchannels), 0, 5., allow_grad=True)

        # peak position, (peak, channel), does not get extended but we still need .freeze(), init with .init_peakpos_from_dataset
        self.param_peakpos[tp] = ExtendableParameter((self.dct_Npeak[tp], self.Nchannels), 0, 0., allow_grad=True)

        # peak position of off-scale peaks, (batch, peak, channel) for simplicity even though most entries will be unused
        self.param_peakpos_nonlinear[tp] = ExtendableParameter((len(self.rboe.oes["rainbow_batch"]), self.dct_Npeak[tp], self.Nchannels), 0, 0., allow_grad=True)

        # Log Classweights, (uid, peak) for simplicity even though most entries will be unused, extendable along (uid)
        self.param_logclassweights[tp] = ExtendableParameter((len(self.rboe.oes["uid"]), self.dct_Npeak[tp]), 0, 0., allow_grad=True)
    
    def init_params(self):
        """Create all parameters for the types, batches and uids of `rboe`, and reset `df_batch_trained`."""
        self.param_logclassweights = ParameterDict()
        self.param_cov = ParameterDict()
        self.param_peakpos = ParameterDict()
        self.param_peakpos_nonlinear = ParameterDict()
        
        for tp in self.rboe.oes["rainbow_type"].labels:
            self.init_params_tp(tp)
        
        # Channel gain log factor, (batch, channel), normed to zero across all batches, extendable along (batch)
        self.param_logscalefactor = ExtendableParameter((len(self.rboe.oes["rainbow_batch"]), self.Nchannels), 0, 0., allow_grad=True, axis_norm=0)
        
        # Channel shift, (uid, channel), parametrized as asinh
        self.param_shift_asinh = ExtendableParameter((len(self.rboe.oes["uid"]), self.Nchannels), 0, 0., allow_grad=self.include_shift)
        
        self.keys_paramdict = ["param_logclassweights", "param_cov", "param_peakpos", "param_peakpos_nonlinear"]
        self.keys_param = ["param_logscalefactor", "param_shift_asinh"]
        
        self.params = {}
        
        self.df_batch_trained = pd.DataFrame(False, index=np.arange(len(self.rboe.oes["rainbow_batch"])), columns=self.channels)
    
    def init_classshares_from_dataset(self, dataset):
        """Initialise the peak share logits with the log GMM peak shares of the dataset (trainable entries only).

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data.
        """
        for tp in self.param_logclassweights:
            if (tp in dataset.peak_shares_uididx) and (tp in dataset.peak_shares):
                self.param_logclassweights[tp].set_trainable_part(torch.log(dataset.peak_shares[tp]), dataset.peak_shares_uididx[tp])
    
    def init_peakpos_from_dataset(self, dataset):
        """Initialise the peak positions from the GMM means of the dataset (trainable entries only).

        The log peak position of a type is initialised as the mean over its measurements of ``ln(GMM mean)`` (raw
        units, clipped at 1 since the position is positive); the nonlinear position of a batch as ``ln(GMM mean)`` of
        its measurement (one of them if the batch has several). Frozen entries are not overwritten.

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data.
        """
        for tp in self.param_peakpos:
            if tp in dataset.GMM_means:
                device = self.param_peakpos[tp].param_trainable.data.device
                val = np.log(np.clip(dataset.GMM_means[tp], a_min=1, a_max=None))
                self.param_peakpos[tp].set_trainable_part(torch.tensor(val.mean(0)))
                self.param_peakpos_nonlinear[tp].set_trainable_part(torch.tensor(val), dataset.peak_shares_batchidx[tp])
    
    def remember_trainable_batches(self, df_):
        """Mark the trainable (batch, channel) factors of a dataset as trained in `df_batch_trained`.

        Kept as a separate table rather than in the parameter, whose mean normalisation would be affected. Entries of
        frozen batches are not changed. Only the factors use it: an untrained factor is set to 1 by
        `get_calibration_factor`.

        Parameters
        ----------
        df_ : pandas.DataFrame
            `RainbowFluorescenceGMMCalibrationDataset.df_channel_train` (index batch names, columns channels).
        """
        df = df_.copy()
        df.index = self.rboe.oes["rainbow_batch"].transform(df.index)
        self.df_batch_trained = self.df_batch_trained | df.loc[df.index[~np.isin(df.index, self.frozen_batches)], self.channels]
    
    
    def parameters_covariance(self):
        """Iterator over the covariance (precision) parameters, used by the initial covariance fit."""
        return self.param_cov.parameters()
    
    def parameters_peak(self):
        """Iterator over the parameters of the peak positions: positions, nonlinear positions, gains and shifts."""
        return itertools.chain(*[self.param_peakpos.parameters(), self.param_peakpos_nonlinear.parameters(),
                                 self.param_logscalefactor.parameters(), self.param_shift_asinh.parameters()])

    def freeze(self):
        """Freeze all parameters (no longer trained, also after `extend`) and add all batches to `frozen_batches`."""
        for key_paramdict in self.keys_paramdict:
            for param in getattr(self, key_paramdict).values(): param.freeze()
        for key_param in self.keys_param:
            getattr(self, key_param).freeze()
        self.frozen_batches = list(set(self.frozen_batches) | set(self.df_batch_trained.index))
    
    def extend(self, rboe, extend_inds, dct_Npeak={}, class_cofactor={}):
        """Extend the module to new batches, uids and rainbow types; the existing entries become fixed.

        Parameters
        ----------
        rboe : RainbowBatchOrdinalEncoder
            Extended batch encoder (replaces `rboe`).
        extend_inds : dict of str to list of int
            New ordinals per key, the output of `RainbowBatchOrdinalEncoder.extend`.
        dct_Npeak : dict of str to int, default {}
            Number of peaks per rainbow type of the new data; new types are added.
        class_cofactor : dict of str to float, default {}
            Arcsinh cofactor per rainbow type; only new types are added, known types keep their cofactor.

        Returns
        -------
        None

        Raises
        ------
        CalibrationModuleException
            If a known type has a different number of peaks in `dct_Npeak`.
        """
        self.rboe = rboe
        
        # Add parameters for new types:
        for key in dct_Npeak:
            if key not in self.dct_Npeak:
                self.dct_Npeak[key] = dct_Npeak[key]
            else:
                if self.dct_Npeak[key]!=dct_Npeak[key]:
                    raise CalibrationModuleException(f"Rainbow type {key} is already registered as having {self.dct_Npeak[key]} peaks, but extended data says {dct_Npeak[key]} peaks!")
                    
        for key in class_cofactor:
            if key not in self.class_cofactor:
                self.class_cofactor[key] = class_cofactor[key]
        
        # first extend simple parameters
        self.param_logscalefactor.extend(extend_inds["rainbow_batch"])
        self.param_shift_asinh.extend(extend_inds["uid"])
        
        # now extend dicts
        for tp in self.param_cov.keys():
            self.param_cov[tp].extend(extend_inds["uid"])
            self.param_peakpos_nonlinear[tp].extend(extend_inds["rainbow_batch"])
            self.param_logclassweights[tp].extend(extend_inds["uid"])
        
        # finally add dicts for new types
        for tp in self.rboe.oes["rainbow_type"].labels:
            if not tp in self.param_cov.keys():
                self.init_params_tp(tp)
        
        # extend remembered
        for i in range(len(self.rboe.oes["rainbow_batch"])):
            if not i in self.df_batch_trained.index:
                self.df_batch_trained.loc[i] = False
    
    
    def update_params(self, no_cov=False):
        """Compute the current model quantities from the parameters into `params` (with gradients).

        Parameters
        ----------
        no_cov : bool, default False
            Skip the precision matrices (not needed for the initial peak fit).
        """
        
        # scale factor, (batch, channel)
        self.params["scalefactor"] = self.param_logscalefactor().exp()
        
        # channel shift, (uid, channel)
        self.params["channel_shift"] = torch.sinh(self.param_shift_asinh())
        
        # class shares, (uid, peak) for every type
        self.params["log_classshares"] = {}
        for tp in self.param_logclassweights:
            self.params["log_classshares"][tp] = F.log_softmax(self.param_logclassweights[tp](), -1)
        
        # peak positions, (peak, channel) for every type
        self.params["peak_position"] = {}
        for tp in self.param_peakpos:
            self.params["peak_position"][tp] = self.param_peakpos[tp]().exp()
        # non-linear peak positions, (batch, peak, channel) for every type
        self.params["peak_position_nonlinear"] = {}
        for tp in self.param_peakpos:
            self.params["peak_position_nonlinear"][tp] = self.param_peakpos_nonlinear[tp]().exp()
        
        if not no_cov:
            self.params["has_cov"] = True
            # precision, (uid, peak, channel, channel)
            self.params["precision"] = {}
            #self.params["scale_tril"] = {}
            # logdet of precision, (uid, peak)
            self.params["logdet_precision"] = {}
            for tp in self.param_cov:
                self.params["precision"][tp], self.params["logdet_precision"][tp] = self.param_cov[tp]()
                #self.params["scale_tril"][tp] = self.param_covTril[tp]()
                #self.params["scale_tril"][tp] = _precision_to_scale_tril(self.params["precision"][tp])
        else:
            self.params["has_cov"] = False
    
    def get_dist_parameters_type(self, tp, uidx, batchidx, peak_islinear=None, update=True):
        """Model parameters for every event (or measurement) of one rainbow type.

        Parameters
        ----------
        tp : str
            Rainbow type.
        uidx : torch.Tensor
            Encoded uid per event, shape (events,).
        batchidx : torch.Tensor
            Encoded batch per event, shape (events,).
        peak_islinear : torch.Tensor, optional
            Boolean usable-peak mask, shape (events, peak, channel). Where False, the positions are replaced by the
            free position of the batch and the shift is 0. Default: all peaks use ``mu_p * g_b`` and the shift.
        update : bool, default True
            Recompute `params` first (`update_params`).

        Returns
        -------
        dict of str to torch.Tensor
            'shift' (``sinh(a)``, in units of the cofactor, shape (events, 1 or peak, channel)), 'scale' (gain g,
            (events, 1, channel)), 'peak_position' and 'peak_position_scaled' (raw units, without and with the gain,
            (events, peak, channel)), 'log_classshares' ((events, peak)), and if available 'precision' ((events, peak,
            channel, channel), arcsinh scale) and 'logdet_precision' ((events, peak)).
        """
        if update: self.update_params()
        
        params = {}
        params["shift"] = self.params["channel_shift"][uidx][:,None] # (event, peak, channel)
        params["scale"] = self.params["scalefactor"][batchidx][:,None] # (event, peak, channel)
        params["peak_position"] = self.params["peak_position"][tp].repeat(uidx.shape[0],1,1) # (event, peak, channel)
        params["peak_position_scaled"] = params["peak_position"] * params["scale"]
        params["log_classshares"] = self.params["log_classshares"][tp][uidx] # (event, peak)

        if peak_islinear is not None:
            # overwrite after scaling to keep untrained scaling at one
            def overwrite_mask(x, y, mask):
                return x * mask + y * (~mask)
            params["peak_position"] = overwrite_mask(params["peak_position"], self.params["peak_position_nonlinear"][tp][batchidx], peak_islinear)
            params["peak_position_scaled"] = overwrite_mask(params["peak_position_scaled"], self.params["peak_position_nonlinear"][tp][batchidx], peak_islinear)
            params["shift"] = params["shift"] * peak_islinear
            #params["peak_position"][~peak_islinear] = self.params["peak_position_nonlinear"][tp][batchidx][~peak_islinear]
            #params["peak_position_scaled"][~peak_islinear] = self.params["peak_position_nonlinear"][tp][batchidx][~peak_islinear]
        
        if self.params["has_cov"]:
            params["precision"] = self.params["precision"][tp][uidx] # (event, peak, channel, channel)
            params["logdet_precision"] = self.params["logdet_precision"][tp][uidx] # (event, peak)
            #params["scale_tril"] = self.params["scale_tril"][tp][uidx] # (event, peak, channel, channel)
        
        return params
    
    def get_dist_parameters(self, uidx, batchidx, peak_islinear=None, update=True, no_cov=False):
        """`get_dist_parameters_type` for several rainbow types at once.

        Parameters
        ----------
        uidx : dict of str to torch.Tensor
            Encoded uid per event, per type.
        batchidx : dict of str to torch.Tensor
            Encoded batch per event, per type (same keys as `uidx`).
        peak_islinear : dict of str to torch.Tensor, optional
            Usable-peak mask per type, see `get_dist_parameters_type`.
        update : bool, default True
            Recompute `params` first.
        no_cov : bool, default False
            Skip the precision matrices when updating.

        Returns
        -------
        dict of str to dict
            Output of `get_dist_parameters_type` per type.

        Raises
        ------
        ValueError
            If `uidx` and `batchidx` have different keys.
        """
        if set(uidx.keys()) != set(batchidx.keys()):
            raise ValueError
        
        if update: self.update_params(no_cov=no_cov)
        params = {tp:self.get_dist_parameters_type(
                        tp, uidx[tp], batchidx[tp], peak_islinear=None if peak_islinear is None else peak_islinear[tp], update=False)
                for tp in uidx.keys()}
        
        return params
    
    
    def get_nll_batch(self, batch, tp, accumulate=True, accumulate_weight=1.):
        """Negative log-likelihood of a batch of beads of one rainbow type under the mixture.

        The per-bead NLL is ``-logsumexp_p(log share_p + log density_p)``, clamped to [-1000, 1000] with a
        straight-through clamp (forward value clamped, gradient passed through).

        Parameters
        ----------
        batch : dict
            Batch as returned by `RainbowFluorescenceGMMCalibrationDataset.get_batch`.
        tp : str
            Rainbow type.
        accumulate : bool, default True
            Recompute the parameters (`update_params`), compute ``mean(nll * accumulate_weight)`` and backpropagate it
            (gradients accumulate in the parameters). If False, use the current `params` (no update) and return the
            per-bead NLL without backward pass.
        accumulate_weight : float, default 1.0
            Weight of this batch when accumulating, e.g. ``batch size / total`` so that the accumulated gradient equals
            that of the mean over all batches.

        Returns
        -------
        torch.Tensor
            If `accumulate`: the weighted mean loss (scalar, after backward). Else: NLL per bead, shape (events,).
        """
        NLL_CLIP_MIN, NLL_CLIP_MAX = -1e3, 1e3
        
        params = self.get_dist_parameters_type(tp, batch["uid_idx"], batch["batch_idx"], peak_islinear=batch["peak_islinear"], update=accumulate)
        #dist = ASinhMultivariateNormal(loc_raw= params["peak_position_scaled"] / self.class_cofactor[tp],
        #                               scale_tril=params["scale_tril"])
        dist = ASinhMultivariateNormal(loc_raw= params["peak_position_scaled"] / self.class_cofactor[tp],
                                       precision_matrix=params["precision"],
                                       precision_logdet=params["logdet_precision"])
        x_dist = (batch["data"] / self.class_cofactor[tp]) [:,None] + params["shift"]
        llh_separate = dist.log_prob_onbase(x_dist) if self.log_prob_onbase else dist.log_prob(x_dist)
        
        nll = -torch.logsumexp( llh_separate + params["log_classshares"], -1)
        
        nll = diffclamp(nll, NLL_CLIP_MIN, NLL_CLIP_MAX)
        
        if accumulate:
            loss = (nll * accumulate_weight).mean()
            loss.backward()
            return loss
        else:
            return nll
    
    def accumulate_nll_gradients(self, dataset):
        """Draw a new subsample, accumulate the gradients of its mean NLL batch by batch, and return the loss.

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data.

        Returns
        -------
        float
            Mean NLL per bead of the subsample (over all types).
        """
        loaders = dataset.sample_loaders()
        total = sum([len(v) for v in loaders.values()])
        loss = 0.
        for tp, loader in loaders.items():
            for batch in loader:
                loss_batch = self.get_nll_batch(batch, tp, accumulate=True,
                                   accumulate_weight= batch["data"].shape[0] / total
                                  )
                loss += loss_batch.item()
        
        torch.cuda.empty_cache()
        return loss
    
    
    def export(self):
        """Export the module (settings, trained/frozen batches, state dict, encoder) as a dict for `from_exported`."""
        exported = {}
        exported["rboe"] = self.rboe.export()
        
        for key in ["dct_Npeak", "channels", "class_cofactor", "include_shift", "is_trained", "log_prob_onbase", "frozen_batches"]:
            exported[key] = getattr(self, key)
        exported["df_batch_trained"] = pdsave_expand(self.df_batch_trained)
        
        exported["state_dict"] = self.state_dict()
        
        return exported
    
    @classmethod
    def from_exported(cls, exported, rboe=None):
        """Rebuild a module from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export`.
        rboe : RainbowBatchOrdinalEncoder, optional
            Batch encoder to use (shared with the other modules of a `RainbowCalibrator`). Default: rebuilt from
            ``exported['rboe']``.

        Returns
        -------
        RainbowFluorescenceGMMCalibrationModule
        """
        rboe = rboe if rboe is not None else RainbowBatchOrdinalEncoder.from_exported(exported["rboe"])
        
        module = cls(rboe=rboe,
                     dct_Npeak=exported["dct_Npeak"],
                     channels=exported["channels"],
                     class_cofactor=exported["class_cofactor"],
                     include_shift=exported["include_shift"],
                     log_prob_onbase=exported["log_prob_onbase"],
                    )
        module.is_trained = exported["is_trained"]
        module.load_state_dict(exported["state_dict"])
        
        module.frozen_batches = exported["frozen_batches"]
        module.df_batch_trained = pdsave_collect(exported["df_batch_trained"])
        
        return module
    
    
    def get_calibration_factor(self, rainbow_batch=None, adata=None, fct_hash_settings=None, fix_untrained=True, warn_untrained=True, return_mask=False):
        """Fluorescence calibration factors (``1 / g_b``) of one batch: multiply its raw data by them.

        Parameters
        ----------
        rainbow_batch : str, optional
            Batch name (preferred if given).
        adata : anndata.AnnData, optional
            Sample whose batch is used, see `RainbowBatchOrdinalEncoder.transform_adata_batch`.
        fct_hash_settings : callable, optional
            Settings hash function for the batch inference. Default: ``adata.uns['cytoconfig'].fct_hash_settings``.
        fix_untrained : bool, default True
            Set factors that were not trained for the batch to 1.
        warn_untrained : bool, default True
            Warn (`CalibrationModuleUntrainedParameterWarning`) if some factors of the batch were not trained.
        return_mask : bool, default False
            Also return the mask of the trained factors.

        Returns
        -------
        pandas.Series or tuple of pandas.Series
            Factor per channel (index: `channels`); with `return_mask` also the boolean trained mask (same index).
        """
        batchidx = self.rboe.transform_adata_batch(adata=adata, rainbow_batch=rainbow_batch, fct_hash_settings=fct_hash_settings)
        with torch.no_grad():
            scalefactor = pd.Series((-self.param_logscalefactor()).exp()[batchidx].cpu().numpy(), index=self.channels)
        
        trained_mask = self.df_batch_trained.loc[batchidx]
        if warn_untrained & ((~trained_mask).sum()>0):
            warnstr = f"{(~trained_mask).sum()} (out of {trained_mask.shape[0]}) scaling factors were not trained for batch {self.rboe.oes['rainbow_batch'].labels[batchidx]}!"
            if fix_untrained:
                warnstr += f" They are set to 1, will be untrustworthy!"
            else:
                warnstr += f" They are not set to 1, will be untrustworthy!"
            warnings.warn(warnstr, CalibrationModuleUntrainedParameterWarning)
        if fix_untrained:
            scalefactor[~trained_mask] = 1.
        
        if return_mask:
            return scalefactor, trained_mask
        else:
            return scalefactor
    
    def get_calibration_rainbow_shift(self, adata=None, uid=None, tp=None):
        """Fitted shift of one rainbow bead measurement, in raw units: add it to the raw data before the factors.

        The shift is fitted in units of the cofactor of the measurement's rainbow type (``sinh(a)``) and returned as
        ``sinh(a) * f``, so the rainbow type is needed besides the uid. A calibrated rainbow measurement including the
        shift is ``(raw + shift) * factor``.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Rainbow bead measurement whose ``uns['uid']`` and ``uns['rainbow_type']`` are used where `uid` or `tp` are
            not given or not known.
        uid : str, optional
            Uid of the measurement.
        tp : str, optional
            Rainbow type of the measurement.

        Returns
        -------
        pandas.Series
            Shift per channel (index: `channels`), raw units.

        Raises
        ------
        CalibrationModuleException
            If neither `adata` nor both `uid` and `tp` are given, or the uid or type cannot be found.

        Warns
        -----
        CalibrationModuleWarning
            If the given `uid` or `tp` is unknown and `adata` is used instead.
        """
        if (adata is None) and ((uid is None) or (tp is None)):
            raise CalibrationModuleException(f"Getting the rainbow calibration shift requires either adata or both 'uid' and 'tp'!")
        
        # First try using explicit uid to get index
        try_adata = uid is None
        if uid is not None:
            uid_idx = self.rboe.oes["uid"].transform(uid)
            if uid_idx==-1:
                if adata is None:
                    raise CalibrationModuleException(f"Did not find provided uid '{uid}' and no adata was given!")
                else:
                    warnings.warn(f"Uid '{uid}' was not found in ordinal encoder, trying to get it from adata!", CalibrationModuleWarning)
                    try_adata = True
        
        # If not available or failed, try from adata
        if try_adata:
            uid_idx = self.rboe.transform_adata(adata)["uid"]
            if uid_idx==-1 and ("uid" in adata.uns):
                raise CalibrationModuleException(f"Could not obtain uid index from adata, '{adata.uns['uid']}' was not found in ordinal encoder!")
            if uid_idx==-1 and ("uid" not in adata.uns):
                raise CalibrationModuleException(f"Could not obtain uid index from adata, adata.uns has no 'uid'!")
        
        # If tp given, try using it
        try_adata = tp is None
        if tp is not None:
            if not tp in self.class_cofactor:
                if adata is None:
                    raise CalibrationModuleException(f"Did not find provided rainbow type '{tp}' and no adata was given!")
                else:
                    warnings.warn(f"Provided rainbow type '{tp}' was not set up, trying to get it from adata instead!", CalibrationModuleWarning)
                    try_adata = True
        
        # try getting tp from adata:
        if try_adata:
            if not "rainbow_type" in adata.uns:
                raise CalibrationModuleException(f"Could not obtain rainbow type from adata, adata.uns has no 'rainbow_type'!")
            tp = adata.uns["rainbow_type"]
            if not tp in self.class_cofactor:
                raise CalibrationModuleException(f"Rainbow type '{tp}' from adata was not set up!")
        
        # now get shift
        with torch.no_grad():
            shift = pd.Series(torch.sinh(self.param_shift_asinh())[uid_idx].cpu().numpy() * self.class_cofactor[tp], index=self.channels)
        
        return shift

    
class RainbowFluorescenceGMMCalibrationTrainer():
    """Training schedule of `RainbowFluorescenceGMMCalibrationModule` (as run by `RainbowCalibrator.fit_fluorescence`).

    `fit_init` fits the peak positions to the GMM means (`fit_init_peak`), then the covariances alone
    (`fit_init_cov`); `fit` then fits all parameters by maximum likelihood.

    Parameters
    ----------
    module : RainbowFluorescenceGMMCalibrationModule
        Module to train (in place).
    """
    def __init__(self, module):
        self.module = module
        self.is_trained = False
        
    def fit_init_peak(self, dataset, Niter=1000, lr=1e-3, momentum=.6, showprogress=False):
        """Initial fit of the peak positions, gains and shifts to the GMM peak means.

        Initialises the peak shares and positions from the dataset, then minimises the summed squared difference on the
        arcsinh scale between ``arcsinh(GMM mean / f + sinh(a))`` and the modelled peak positions of every measurement
        (SGD). Runs longer than the loss alone would need, which gives a better initialisation of the shifts.
        Interrupting with Ctrl-C stops the loop early.

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data.
        Niter : int, default 1000
            Number of SGD steps.
        lr : float, default 1e-3
            SGD learning rate.
        momentum : float, default 0.6
            SGD momentum.
        showprogress : bool, default False
            Show a progress bar.

        Returns
        -------
        None
            Sets `losshistory_init_means` and `losshistory_init_means_absshift` (mean absolute shift per step).
        """
        self.module.init_classshares_from_dataset(dataset)
        self.module.init_peakpos_from_dataset(dataset)
        
        optimizer = torch.optim.SGD(self.module.parameters_peak(), lr=lr, momentum=momentum)
        self.losshistory_init_means = []
        self.losshistory_init_means_absshift = []
        
        def get_peakpos_loss():
            params = self.module.get_dist_parameters(dataset.peak_shares_uididx, dataset.peak_shares_batchidx, dataset.peak_usemask, no_cov=True)

            real_peakpos_asinh = {tp: torch.asinh(( torch.tensor(v, device=params[tp]["shift"].device)) / self.module.class_cofactor[tp] + params[tp]["shift"])
                                  for tp, v in dataset.GMM_means.items()}
            pred_peakpos_asinh = {tp: torch.asinh( params[tp]["peak_position_scaled"] / self.module.class_cofactor[tp])
                                  for tp in params.keys()}

            loss = 0.
            for tp in params.keys():
                 loss += (real_peakpos_asinh[tp] - pred_peakpos_asinh[tp]).square().sum()
            mean_abs_shift = self.module.params["channel_shift"].detach()[torch.cat(list(dataset.peak_shares_uididx.values()))].abs().mean()
            return loss, mean_abs_shift

        def do_loss(back=True):
            loss, mean_abs_shift = get_peakpos_loss()
            if back: loss.backward()
            self.losshistory_init_means.append(loss.item())
            self.losshistory_init_means_absshift.append(mean_abs_shift.item())
        
        fshow = tqdm if showprogress else (lambda x, **kwargs: x)
        try:
            for i in fshow(range(Niter), mininterval=1):
                optimizer.zero_grad()
                do_loss(True)
                optimizer.step()
        except KeyboardInterrupt:
            pass
        do_loss(False)
    
    def fit_init_cov(self, dataset, Niter=1000, lr=1e-2, showprogress=False):
        """Initial fit of the covariances alone, by maximum likelihood with all other parameters fixed.

        Every step uses a new subsample of the dataset (Adam on the precision parameters). Interrupting with Ctrl-C
        stops the loop early.

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data.
        Niter : int, default 1000
            Number of Adam steps.
        lr : float, default 1e-2
            Adam learning rate.
        showprogress : bool, default False
            Show a progress bar.

        Returns
        -------
        None
            Sets `losshistory_init_cov` (mean NLL per step).
        """
        optimizer_full = torch.optim.SGD(self.module.parameters(), lr=0.) # only for zero_grad so they don't accumulate to infinity
        optimizer = torch.optim.Adam(self.module.parameters_covariance(), lr=lr)
        
        self.losshistory_init_cov = []
        
        try:
            pbar = tqdm(range(Niter), mininterval=1) if showprogress else range(Niter)
            for it in pbar:
                optimizer_full.zero_grad()
                loss = self.module.accumulate_nll_gradients(dataset)
                self.losshistory_init_cov.append(loss)
                if showprogress: pbar.set_postfix({'nll': loss}, refresh=False)
                optimizer.step()
        except KeyboardInterrupt:
            pass
        loss = self.module.accumulate_nll_gradients(dataset)
        self.losshistory_init_cov.append(loss)
    
    def fit_init(self, dataset, showprogress=False):
        """Record the trainable factors, then run `fit_init_peak` and `fit_init_cov` with their defaults.

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data; its `df_channel_train` is stored with `remember_trainable_batches`.
        showprogress : bool, default False
            Show progress bars.

        Returns
        -------
        None
        """
        self.module.remember_trainable_batches(dataset.df_channel_train)
        self.fit_init_peak(dataset, showprogress=showprogress)
        self.fit_init_cov(dataset, showprogress=showprogress)

        torch.cuda.empty_cache()
    
    def plot_fit_init(self):
        """Plot the losses of `fit_init_peak` (squared error, log scale) and `fit_init_cov` (mean NLL) per step."""
        fig, ax = plt.subplots(1,2,figsize=(20,5))
        ax[0].plot(self.losshistory_init_means)
        ax[0].set_yscale("log")
        ax[0].set_title("Initial Mean Fit")
        ax[0].set_xlabel("Iteration")
        ax[0].set_ylabel("MSE")
        ax[1].plot(self.losshistory_init_cov)
        ax[1].set_title("Initial Covariance Fit")
        #ax[1].set_yscale("log")
        ax[1].set_xlabel("Iteration")
        ax[1].set_ylabel("LLH")
        plt.show()

    def fit(self, dataset, Niter=10000, lr=1e-2, warmup=40, showprogress=True):
        """Fit all trainable parameters by maximum likelihood (Adam, a new subsample every step).

        The first call creates the optimizer and marks trainer and module as trained; later calls continue with the
        same optimizer and append to `fit_losshistory`. Interrupting with Ctrl-C stops the loop early.

        Parameters
        ----------
        dataset : RainbowFluorescenceGMMCalibrationDataset
            Training data.
        Niter : int, default 10000
            Number of Adam steps.
        lr : float, default 1e-2
            Adam learning rate.
        warmup : int, default 40
            Increase the learning rate linearly from ``lr / warmup`` to `lr` over the first `warmup` steps (of every
            call); 0 disables it. In a later call with ``warmup=0``, the learning rate of the previous call is kept
            and `lr` has no effect.
        showprogress : bool, default True
            Show a progress bar.

        Returns
        -------
        None
            Sets `fit_losshistory` (mean NLL per step).
        """
        if not self.is_trained:
            self.optimizer = torch.optim.Adam(self.module.parameters(), lr=0 if warmup>0 else lr)
            self.fit_losshistory = []
            self.is_trained = True
            self.module.is_trained = True
        
        try:
            pbar = tqdm(range(Niter), mininterval=1) if showprogress else range(Niter)
            for it in pbar:
                if warmup>0 and it<warmup:
                    self.optimizer.param_groups[0]["lr"] = lr * (it+1)/warmup
                self.optimizer.zero_grad()
                loss = self.module.accumulate_nll_gradients(dataset)
                self.fit_losshistory.append(loss)
                if showprogress: pbar.set_postfix({'nll': loss}, refresh=False)
                self.optimizer.step()
        except KeyboardInterrupt:
            pass
        torch.cuda.empty_cache()
    
    def plot_fit(self):
        """Plot the loss (mean NLL) of `fit` per step."""
        fig, ax = plt.subplots(1,1,figsize=(20,5))
        ax.plot(self.fit_losshistory)
        ax.set_title("Full Fit")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NLL")
        plt.show()
