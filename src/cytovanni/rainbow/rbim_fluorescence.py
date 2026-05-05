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
from ..exceptions import MissingChannelException, IntegrationModuleException, IntegrationModuleWarning, IntegrationModuleUntrainedParameterWarning
from .rboe import RainbowBatchOrdinalEncoder
from ..torch import pdsave_expand, pdsave_collect


class RainbowFluorescenceGMMIntegrationModule(Module):
    """ 
        Integration Module on the fluorescence of rainbow beads.
        Models every type of bead as a Sinh transformed multivariate Normal distribution with a sample dependent covariance matrix.
        The mean positions are fixed per type of rainbow bead, and only scaled (batch-dependent) and shifted (sampmle-dependent)
        such that they reproduce the positions in the samples.
        For peaks that are off-scale and no longer linear, fits the mean position freely (batch-dependent).
    """
    def __init__(self, rboe, dct_Npeak, channels, class_cofactor,
                 include_shift=True, log_prob_onbase=False):
        """ 
            Module for the fluorescence integration with rainbow beads.
            
            :param rboe: RainbowBatchOrdinalEncoder. Batch encoder, set up.
            
            :param dct_Npeak: dict. Dictionary of number of peaks for every bead type.
            
            :param Nchannels: int. Number of fluorescence channels.
            
            :param class_cofactor: dict. Dictionary of arcsinh cofactor for every bead type.
            
            :param include_shift: bool. Whether to include a shift in addition to the scaling in the fit.
            
            :param log_prob_onbase: bool. Whether to get the likelihood directly on the underlying Gaussian instead of the Sinh transformed Gaussian. If True, effectively the same as using a normal Gaussian on the arcsinh transformed data instead of ASinhMultivariateNormal on the raw data.
        """
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
        """ Initialize parameters that are type dependent for one type tp.
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
        """ Initialize all parameters.
        """
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
        """ Init classshares from dataset.
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. Data to extract init from.
        """
        for tp in self.param_logclassweights:
            if (tp in dataset.peak_shares_uididx) and (tp in dataset.peak_shares):
                self.param_logclassweights[tp].set_trainable_part(torch.log(dataset.peak_shares[tp]), dataset.peak_shares_uididx[tp])
    
    def init_peakpos_from_dataset(self, dataset):
        """ Init peakpos as mean across all in dataset.
            
            Initialize as log of clipped GMM mean (after cofactor scaling), since it has to be positive.
            Initialise such that frozen value cannot be overwritten.
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. Data to extract init from.
        """
        for tp in self.param_peakpos:
            if tp in dataset.GMM_means:
                device = self.param_peakpos[tp].param_trainable.data.device
                val = np.log(np.clip(dataset.GMM_means[tp], a_min=1, a_max=None))
                self.param_peakpos[tp].set_trainable_part(torch.tensor(val.mean(0)))
                self.param_peakpos_nonlinear[tp].set_trainable_part(torch.tensor(val), dataset.peak_shares_batchidx[tp])
    
    def remember_trainable_batches(self, df_):
        """ Remember which scaling factors are trained and which aren't.
            Would be nicer to do this in the parameter itself, but messes up the normalizations.
            Only relevant for scale, shift doesn't get trained anyway.
        """
        df = df_.copy()
        df.index = self.rboe.oes["rainbow_batch"].transform(df.index)
        self.df_batch_trained = self.df_batch_trained | df.loc[df.index[~np.isin(df.index, self.frozen_batches)], self.channels]
    
    
    def parameters_covariance(self):
        """ Only parameters that affect the covariance.
        """
        return self.param_cov.parameters()
    
    def parameters_peak(self):
        """ Only parameters that affect the peak positions.
        """
        return itertools.chain(*[self.param_peakpos.parameters(), self.param_peakpos_nonlinear.parameters(),
                                 self.param_logscalefactor.parameters(), self.param_shift_asinh.parameters()])

    def freeze(self):
        """ Freeze all parameters of self.
            Also remember which batches were frozen.
        """
        for key_paramdict in self.keys_paramdict:
            for param in getattr(self, key_paramdict).values(): param.freeze()
        for key_param in self.keys_param:
            getattr(self, key_param).freeze()
        self.frozen_batches = list(set(self.frozen_batches) | set(self.df_batch_trained.index))
    
    def extend(self, rboe, extend_inds, dct_Npeak={}, class_cofactor={}):
        """ Replace rboe by new, extend all parameters by extend_inds.
            
            :param rboe: RainbowBatchOrdinalEncoder. Extended batch encoder.
            
            :param extend_inds: dict. Extended indices dict, outout of rboe.extend.
        """
        self.rboe = rboe
        
        # Add parameters for new types:
        for key in dct_Npeak:
            if key not in self.dct_Npeak:
                self.dct_Npeak[key] = dct_Npeak[key]
            else:
                if self.dct_Npeak[key]!=dct_Npeak[key]:
                    raise IntegrationModuleException(f"Rainbow type {key} is already registered as having {self.dct_Npeak[key]} peaks, but extended data says {dct_Npeak[key]} peaks!")
                    
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
        """ Update self.param with current peak positions etc.
            
            :param no_cov: bool. Whether to include covariance parameters, unnecessary for the initial mean fit.
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
        # non-linear peak positions, (uid, peak, channel) for every type
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
        """ For index tensors uidx and batchidx, get all the peak positions, precisions etc.
            If peak_islinear is given, replace respective means bei their nonlinear position for the batch.
            
            :param tp: str. Bead type key.
            
            :param uidx: torch.Tensor. Unique identifier index.
            
            :param batchids: torch.Tensor. Batch index.
            
            :param peak_islinear: torch.Tensor. Optional, mask of which peaks are in the linear range.
            
            :param update: bool. Whether to update self.params first.
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
        """ For dicts uidx and batchidx, get all the peak positions, precisions etc.
            If peak_islinear is given, replace respective means bei their nonlinear position for the batch.
            
            :param uidx: dict. Unique identifier index for every type.
            
            :param batchids: dict. Batch index for every type.
            
            :param peak_islinear: dict. Optional, mask of which peaks are in the linear range, for every type.
            
            :param update: bool. Whether to update self.params first.
            
            :param no_cov: bool. Whether to not update covariance parameters.
        """
        if set(uidx.keys()) != set(batchidx.keys()):
            raise ValueError
        
        if update: self.update_params(no_cov=no_cov)
        params = {tp:self.get_dist_parameters_type(
                        tp, uidx[tp], batchidx[tp], peak_islinear=None if peak_islinear is None else peak_islinear[tp], update=False)
                for tp in uidx.keys()}
        
        return params
    
    
    def get_nll_batch(self, batch, tp, accumulate=True, accumulate_weight=1.):
        """ Get nll for single bead batch of type tp.
            
            If accumulate, already performs the backward pass and doesn't return anything.
            If accumulating, make sure you pass the appropriate accumulate_weight to emulate single pass with .mean!
            
            If not accumulate, doesn't call .update_params(), otherwise needs to as it deletes the graph.
            
            :param batch: dict. Batched data for one type of peak.
            
            :param tp: str. Bead type.
            
            :param accumulate: bool. Whether to directly accumulate gradients instead of returning the full llh.
            
            :param accumulate_weight: float. For accumulation, proper weight of batch gradients.
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
        """ Accumulate gradients over one sample from dataset, batched for memory conservation.
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. Data to get the sample from.
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
        """ Export self to dict.
        """
        exported = {}
        exported["rboe"] = self.rboe.export()
        
        for key in ["dct_Npeak", "channels", "class_cofactor", "include_shift", "is_trained", "log_prob_onbase", "frozen_batches"]:
            exported[key] = getattr(self, key)
        exported["df_batch_trained"] = pdsave_expand(self.df_batch_trained)
        
        exported["state_dict"] = self.state_dict()
        
        return exported
    
    @classmethod
    def from_exported(cls, exported, rboe=None):
        """ Construct new from exported.
            
            :param exported: dict. Output of previous .export().
            
            :param rboe: RainbowBatchOrdinalEncoder. Optional, batch encoder, already setup.
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
    
    
    def get_integration_factor(self, rainbow_batch=None, adata=None, fct_hash_settings=None, fix_untrained=True, warn_untrained=True, return_mask=False):
        """ Get integration scale factor from rainbow_batch into common units.
            Integrating from batch into common is done by multiplying with factor.

            :param rainbow_batch: str. Optional, batch for which to get the integration factor.

            :param adata: AnnData. Optional, AnnData from which to get the batch for the integration factor.
            
            :param fix_untrained: bool. Whether to fix untrained scale factors to 1.
            
            :param warn_untrained: bool. Whether to warn if some factors are untrained.
            
            :param return_mask: bool. Whether to also return the mask which factors were trained.
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
            warnings.warn(warnstr, IntegrationModuleUntrainedParameterWarning)
        if fix_untrained:
            scalefactor[~trained_mask] = 1.
        
        if return_mask:
            return scalefactor, trained_mask
        else:
            return scalefactor
    
    def get_integration_rainbow_shift(self, adata=None, uid=None, tp=None):
        """ Get integration rainbow sample shift, to be added before scaling.
            Since the shift is applied after the asinh cofactor, also needs rainbow type to calculate correct shift for sample 'uid'.
            Integrating from batch into common is done by adding shift, and afterwards multiplying with factor.

            :param adata: AnnData. Optional, AnnData from which to get the uid and rainbow type for the sample shift.

            :param uid: str. Optional, uid for which to get the integration factor.

            :param tp: str. Optional, rainbow type for which to get the integration factor.
        """
        if (adata is None) and ((uid is None) or (tp is None)):
            raise IntegrationModuleException(f"Getting the rainbow integration shift requires either adata or both 'uid' and 'tp'!")
        
        # First try using explicit uid to get index
        try_adata = uid is None
        if uid is not None:
            uid_idx = self.rboe.oes["uid"].transform(uid)
            if uid_idx==-1:
                if adata is None:
                    raise IntegrationModuleException(f"Did not find provided uid '{uid}' and no adata was given!")
                else:
                    warnings.warn(f"Uid '{uid}' was not found in ordinal encoder, trying to get it from adata!", IntegrationModuleWarning)
                    try_adata = True
        
        # If not available or failed, try from adata
        if try_adata:
            uid_idx = self.rboe.transform_adata(adata)["uid"]
            if uid_idx==-1 and ("uid" in adata.uns):
                raise IntegrationModuleException(f"Could not obtain uid index from adata, '{adata.uns['uid']}' was not found in ordinal encoder!")
            if uid_idx==-1 and ("uid" not in adata.uns):
                raise IntegrationModuleException(f"Could not obtain uid index from adata, adata.uns has no 'uid'!")
        
        # If tp given, try using it
        try_adata = tp is None
        if tp is not None:
            if not tp in self.class_cofactor:
                if adata is None:
                    raise IntegrationModuleException(f"Did not find provided rainbow type '{tp}' and no adata was given!")
                else:
                    warnings.warn(f"Provided rainbow type '{tp}' was not set up, trying to get it from adata instead!", IntegrationModuleWarning)
                    try_adata = True
        
        # try getting tp from adata:
        if try_adata:
            if not "rainbow_type" in adata.uns:
                raise IntegrationModuleException(f"Could not obtain rainbow type from adata, adata.uns has no 'rainbow_type'!")
            tp = adata.uns["rainbow_type"]
            if not tp in self.class_cofactor:
                raise IntegrationModuleException(f"Rainbow type '{tp}' from adata was not set up!")
        
        # now get shift
        with torch.no_grad():
            shift = pd.Series(torch.sinh(self.param_shift_asinh())[uid_idx].cpu().numpy() * self.class_cofactor[tp], index=self.channels)
        
        return shift

    
class RainbowFluorescenceGMMIntegrationTrainer():
    """ 
        Simple trainer for RainbowFluorescenceGMMIntegrationModule.
    """
    def __init__(self, module):
        """ 
            :param module: RainbowFluorescenceGMMIntegrationModule. Module to train.
        """
        self.module = module
        self.is_trained = False
        
    def fit_init_peak(self, dataset, Niter=1000, lr=1e-3, momentum=.6, showprogress=False):
        """ 
            Initial fit of only the peak positions.
            
            Fit longer than necessary from loss itself to get better shift initialisation!
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. Data to train on.
            
            :param Niter: int. Numer of iterations.
            
            :param lr: float. Learning rate for SGD.
            
            :param momentum: float. Momentum of SGD.
            
            :param showprogress: bool. Whether to show progress bar.
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
        """ 
            Initial fit of only the covariances.
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. Data to train on.
            
            :param Niter: int. Numer of iterations.
            
            :param lr: float. Learning rate for Adam.
            
            :param showprogress: bool. Whether to show progress bar.
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
        """ First fit means separately, then fit covariance separately.
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. Data to train on.
            
            :param plot: bool. Whether to plot losses after fitting is finished.
            
            :param showprogress: bool. Whether to show progress bar.
        """
        self.module.remember_trainable_batches(dataset.df_channel_train)
        self.fit_init_peak(dataset, showprogress=showprogress)
        self.fit_init_cov(dataset, showprogress=showprogress)

        torch.cuda.empty_cache()
    
    def plot_fit_init(self):
        """ Plot initialisation losses.
        """
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
        """ Fit full model.
            
            :param dataset: RainbowFluorescenceGMMIntegrationDataset. The data to fit with.
            
            :param Niter: int. Number of iterations for the fit.
            
            :param lr: float. Learning rate for the fit.
            
            :param warmup: float. Increase learning rate from zero to lr over first warmup iterations.
            
            :param showprogress: bool. Whether to show progress bar.
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
        """ Plot fit loss.
        """
        fig, ax = plt.subplots(1,1,figsize=(20,5))
        ax.plot(self.fit_losshistory)
        ax.set_title("Full Fit")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NLL")
        plt.show()
