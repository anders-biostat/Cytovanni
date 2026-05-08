import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter, ParameterDict
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from ..exceptions import OverlapStandardisationException, OverlapStandardisationWarning
from ..torch import ExtendableParameter, LossContainer, diffclamp
from ..utils import get_cmap, cmap_to_legendhandles
from ..utils import invert_spectra
from .panel import PanelEmbeddingModule, FixedPanelModule
from .moe import MultiOrdinalEncoder
from .model_transformer_smp import pick_smp_cls, DefaultMultiplier
from ..torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict


def pick_transformer_cls(model):
    """ Get appropriate class for saved transformers from name.
    """
    if model=="dummy":
        return DummyTransformer
    elif model=="spectralfit":
        return SpectralFitTransformer
    elif model=="scaling":
        return ScalingTransformer
    else:
        raise NotImplementedError(f"Cannot find Transformer model {model}!")

def pick_panel_cls(panel):
    if panel=="embedding":
        return PanelEmbeddingModule
    elif panel=="fixed":
        return FixedPanelModule
    else:
        raise NotImplementedError(f"Cannot find panel class {panel}!")


class DummyTransformer(Module):
    """ Dummy transformer to showcase necessary parts.
    """
    NAME = "dummy"
    
    def __init__(self):
        super().__init__()
        self.plot_auxloss_dct = {}
        self.param_dummy = Parameter(torch.tensor([], dtype=torch.float32), requires_grad=False)
    
    @classmethod
    def from_data(cls, channels, stains, *args, **kwargs):
        """ Initialize from data.
        """
        dt = cls()
        dt.channels = channels
        dt.stains = stains
        
        return dt
    
    def export(self):
        """ Export self to dict.
        """
        exported = {"channels":self.channels, "stains":self.stains}
        return exported
    
    @classmethod
    def from_exported(cls, exported, cls_smp=None):
        """ Initialize from exported
        """
        dt = cls()
        dt.channels = exported["channels"]
        dt.stains = exported["stains"]
        dt.stain_marker_name = dt.stains
        
        return dt
        
    def forward(self, x, bidx):
        """ Take input and batch idx, output standardised, unmixed and transformed version on which to apply MMD loss.
        """
        return self.apply_transformation( self.standardise(x, bidx) )
    
    def apply_transformation(self, x):
        """ Transform intensities into scale on which MMD should be calculated, e.g. ArcSinh.
        """
        return x
    
    def standardise(self, x, bidx):
        """ Return standardised version of input, i.e. if input is intensities output should be standardised and unmixed intensities.
        """
        return x
    
    def append_history(self):
        """ Append current parameters to parameter history.
        """
        pass
    
    def get_aux_loss(self):
        """ Auxiliary losses for regularisation etc.
        """
        loss = LossContainer({"loss_transformer":0.})
        return loss
    
    @property
    def device(self):
        """ Device the module is currently on.
        """
        return self.param_dummy.device
    
    def freeze(self):
        """ Freeze all parameters.
        """
        pass
    
    def extend(self, moe, extend_inds):
        """ Extend parameters by new batch indices.
        """
        pass


class ScaleFactor(Module):
    """ Small wrapper of ExtendableParameter.
        
        Optionally clipped to maximum abs log value.
    """
    def __init__(self, Nbatch, Nchannel, anchoridxs=[], use_channel_mask=None, trainable=False, mean_norm=False, max_abs_value=None):
        super().__init__()
        self.parameter = ExtendableParameter((Nbatch,Nchannel), 0, 0., allow_grad=trainable, axis_norm=0 if mean_norm else None)
        self.parameter.param_mask_trainable.data[(list(anchoridxs), slice(None))] = 0.
        if use_channel_mask is not None:
            self.parameter.param_mask_trainable.data[(slice(None), ~torch.tensor(use_channel_mask.to_numpy(), dtype=bool))] = 0.
        
        self.max_abs_value = max_abs_value
    
    def anchor_bidx(self, indices):
        """ Anchor all batch indices in iterable indices to not be trainable.
        """
        self.parameter.freeze_selective((indices, slice(None)))
    
    def forward(self):
        if self.max_abs_value is None:
            return self.parameter()
        else:
            return diffclamp(self.parameter(), -self.max_abs_value, self.max_abs_value)
    
    def freeze(self):
        self.parameter.freeze()
    
    def extend(self, inds):
        self.parameter.extend(inds)
    
    @property
    def shape(self):
        return self.parameter.shape
    
    @property
    def device(self):
        return self.parameter.device
    
    @property
    def logfactor(self):
        return self()
    
    @property
    def factor(self):
        return self().exp()
    
    @property
    def L1(self):
        return self.logfactor.abs().mean(0).sum()

class SpectralFitTransformer(Module):
    """ Fits the single stain spectra, optionally scaling factors per stain and channel.
        Transforms data into ArcSinh scale for MMD.
        
        
    """
    NAME = "spectralfit"
    
    def __init__(self, moe, pem, config, smp=None):
        super().__init__()
        self.moe = moe
        self.pem = pem
        self.smp = smp if smp is not None else DefaultMultiplier()
        
        self.stains = self.pem.stains
        self.channels = self.pem.channels
        self.stain_marker_name = self.pem.stain_marker_name
        
        self.config = config
        
        self.plot_auxloss_dct = {}
        if self.pem.TRAINED_EMBEDDING:            self.plot_auxloss_dct.update({"l1_embedding":      "L1 Embedding"})
        if np.any(self.config["scale_stains"]):   self.plot_auxloss_dct.update({"l1_stainlogscale":  "L1 Stain Scale"})
        if np.any(self.config["scale_channels"]): self.plot_auxloss_dct.update({"l1_channellogscale":"L1 Channel Scale"})
        
        self.paramhistory = {"embedding":[], "stainlogscale":[], "channellogscale":[]}
        
        self.init_parameters()
        self.anchor_batches(self.config["anchor_bidx"])
    
    def init_parameters(self):
        self.param_stainscale = ScaleFactor(len(self.moe[self.config["key_scale_stains"]]),
                                            len(self.stains),
                                            use_channel_mask=self.config["scale_stains"], trainable=True,
                                            max_abs_value=self.config["max_abslogstain"])
        
        self.param_channelscale = ScaleFactor(len(self.moe[self.config["key_scale_channels"]]),
                                              len(self.channels),
                                              use_channel_mask=self.config["scale_channels"], trainable=True,
                                              max_abs_value=self.config["max_abslogchannel"])
    
    @property
    def loss_use_stain_mask(self):
        """ Mask which stains should be used for the loss, here just everything.
        """
        return torch.tensor([True]*len(self.stains), dtype=bool, device=self.device)
    
    def anchor_batches(self, anchor_bidx):
        """ Anchor batches to be not trainable, also passes this through to self.pem.
            E.g. {"a":[0,1]} disables training for all parameters associated with indices [0,1] in batch "a".
        """
        if self.config["key_scale_stains"] in anchor_bidx:
            self.param_stainscale.anchor_bidx(anchor_bidx[self.config["key_scale_stains"]])
        if self.config["key_scale_channels"] in anchor_bidx:
            self.param_channelscale.anchor_bidx(anchor_bidx[self.config["key_scale_channels"]])
        self.pem.anchor_batches(anchor_bidx)
    
    @classmethod
    def from_data(cls, moe, pem, smp=None,
                  scale_stains=False, key_scale_stains="batch", scale_channels=False, key_scale_channels=None,
                  arcsinh_cofactor=5000, L1weight_embedding=0., L1weight_stainscale=0., L1weight_channelscale=0.,
                  max_abslogstain=None, max_abslogchannel=None, anchor_bidx={}):
        """ 
            
            :param moe: MultiOrdinalEncoder. Batch encoder for all relevant batch keys.
            
            :params pem: PanelEmbeddingModule. Module that does the spectral embedding and fit.
            
            :params smp: None, ScaleMultiplier. Optional class that takes unmixed values and stain scaling and multiplies them; can be used to selectively detach gradients for the scaling factors if necessary. Can be set here, but usually requires transformer at instantiation! Better to set using .set_smp after creating transformer!
            
            :params scale_stains: bool, pd.Series. Whether model should include a scaling factor for the stains. Either bool, or Series with index stains and entry bool.
            
            :param key_scale_stains: str. Batch key that affects the stain scaling.
            
            :params scale_channels: bool, pd.Series. Whether model should include a scaling factor for the channels. Either bool, or Series with index channels and entry bool.
            
            :param key_scale_channels: str. Batch key that affects the channel scaling.
            
            :param arcsinh_cofactor: float. Cofactor of ArcSinh used for MMD. Should be set to a multiple (3x or so) of the normal cofactor; setting this higher encourages the fit to focus on the markers with high abundances instead of fitting the noise.
            
            :param L1weight_embedding: float. Weighting of L1 embedding regularisation, set to zero to disable.
            
            :param L1weight_stainscale: float. Weighting of L1 stain scaling regularisation, set to zero to disable.
            
            :param L1weight_channelscale: float. Weighting of L1 channel scaling regularisation, set to zero to disable.
            
            :param max_abslogstain: None, float. If not None, log of stain scale factor is clipped to +- this.
            
            :param max_abslogchannel: None, float. If not None, log of channel scale factor is clipped to +- this.
            
            :params anchor_bidx: dict. Anchor all batch indices in dict to not be trainable. E.g. {"a":[0,1]} disables training for all parameters associated with indices [0,1] in batch "a"; also passes this through to pem.
        """
        config = {}
        config["scale_stains"] = pd.Series(scale_stains, index=pem.stains) if isinstance(scale_stains, bool) else scale_stains.loc[pem.stains]
        config["key_scale_stains"] = key_scale_stains
        config["scale_channels"] = pd.Series(scale_channels, index=pem.channels) if isinstance(scale_channels, bool) else scale_channels.loc[pem.channels]
        config["key_scale_channels"] = key_scale_channels if key_scale_channels is not None else key_scale_stains
        config["arcsinh_cofactor"] = arcsinh_cofactor
        
        config["L1weight_embedding"] = L1weight_embedding
        config["L1weight_stainscale"] = L1weight_stainscale
        config["L1weight_channelscale"] = L1weight_channelscale
        
        config["max_abslogstain"] = max_abslogstain
        config["max_abslogchannel"] = max_abslogchannel
        
        config["anchor_bidx"] = anchor_bidx
        
        sft = cls(moe, pem, config, smp)
        return sft
    
    def export(self, include_history=True):
        """ Export self to dict.
        """
        exportdct = {}
        exportdct["pem"] = self.pem.export()
        exportdct["moe"] = self.moe.export()
        exportdct["smp"] = self.smp.export()
        exportdct["smp_name"] = self.smp.NAME
        exportdct["config"] = pdsave_expand_dict(self.config)
        exportdct["state_dict"] = self.state_dict()
        if include_history:
            exportdct["paramhistory"] = self.paramhistory
        
        return exportdct
    
    @classmethod
    def from_exported(cls, exported, cls_smp=None):
        """ Initialize from exported. Optionally needs smp class if custom implementation is used.
        """
        pem = pick_panel_cls(exported["pem"]["class"]).from_exported(exported["pem"])
        moe = MultiOrdinalEncoder.from_exported(exported["moe"])
        config = pdsave_collect_dict(exported["config"])
        
        sft = cls(moe, pem, config)
        
        if "smp" in exported:
            if cls_smp is None: cls_smp = pick_smp_cls(exported["smp_name"])
            sft.set_smp(cls_smp.from_exported(exported["smp"], sft))
        
        sft.load_state_dict(exported["state_dict"])
        if "paramhistory" in exported:
            sft.paramhistory = exported["paramhistory"]
        return sft
    
    def set_smp(self, smp):
        """ Optionally set scale multiplier; i.e. class that takes unmixed values and stain scaling and multiplies them,
            E.g. use UnmxErrorExclusionMultiplier to selectively detach gradients for the scaling factors if necessary.
        """
        self.smp = smp
        
    def forward(self, x, bidx):
        """ Take input and batch idx, output standardised, unmixed and transformed version on which to apply MMD loss.
        """
        return self.apply_transformation( self.standardise(x, bidx) )
    
    def apply_transformation(self, x):
        """ Transform intensities into scale on which MMD should be calculated, e.g. ArcSinh.
        """
        return torch.arcsinh(x/self.config["arcsinh_cofactor"])
    
    def standardise(self, x, bidx, evl=False):
        """ Return standardised version of input, i.e. if input is intensities output should be standardised and unmixed intensities.
        """
        x = x * self.param_channelscale.factor[bidx[self.config["key_scale_channels"]]]
        xc = self.pem.get_unmixed(x, bidx)
        xc = self.smp( xc, self.param_stainscale.factor[bidx[self.config["key_scale_stains"]]] ) # just a multiplication, but optionally with some detached gradients
        return xc
    
    def get_freeparam_standardisetransf(self, bidx):
        """ For given idx_batch, returns both embedding and log scale factors as gradient accumulating parameters,
            as well as function to do the standardisation and transformation that doesn't mask any part of the embedding or scale factors.
            
            Ignores effects of scale multiplier.
        """
        channellogscale = self.param_channelscale.logfactor[bidx[self.config["key_scale_channels"]]].detach()
        channellogscale.requires_grad = True
        channellogscale.grad = torch.zeros_like(channellogscale)
        
        stainlogscale = self.param_stainscale.logfactor[bidx[self.config["key_scale_stains"]]].detach()
        stainlogscale.requires_grad = True
        stainlogscale.grad = torch.zeros_like(stainlogscale)
        
        params, fct_unmix = self.pem.get_freeparam_unmix(bidx)
        params.update({"channellogscale":channellogscale, "stainlogscale":stainlogscale})
        
        def fct_standardisetransf(x):
            x = x * channellogscale.exp()
            xc = fct_unmix(x)
            xc = xc * stainlogscale.exp()
            xct = self.apply_transformation(xc)
            return xct
        
        return params, fct_standardisetransf
    
    def get_freeparam_collect(self, paramlist):
        """ Takes list of 'params' output of get_freeparam_standardisetransf,
            collects gradients and returns dict of numpy arrays of shape (len(paramlist), param_components).
        """
        graddict = {}
        graddict["channellogscale"] = pd.DataFrame(np.vstack([p["channellogscale"].grad.detach().cpu().numpy() for p in paramlist]),
                                                   columns=self.channels)
        graddict["stainlogscale"] = pd.DataFrame(np.vstack([p["stainlogscale"].grad.detach().cpu().numpy() for p in paramlist]),
                                                 columns=self.stains)
        graddict["embedding"] = pd.DataFrame(np.vstack([p["embedding"].grad[:,0].detach().cpu().numpy() for p in paramlist]),
                                             columns=self.pem.panelmodule_components)
        return graddict
    
    @property
    def learned_channellogscale(self):
        logfactor = pd.DataFrame(self.param_channelscale.logfactor.detach().cpu().numpy(),
                                 index=self.moe.labels[self.config["key_scale_channels"]],
                                 columns=self.channels)
        return logfactor
    
    @property
    def learned_stainlogscale(self):
        logfactor = pd.DataFrame(self.param_stainscale.logfactor.detach().cpu().numpy(),
                                 index=self.moe.labels[self.config["key_scale_stains"]],
                                 columns=self.stains)
        return logfactor
    
    def append_history(self):
        """ Append current parameters to parameter history.
        """
        self.paramhistory["embedding"].append(self.pem.get_learned_embedding())
        self.paramhistory["stainlogscale"].append(self.learned_stainlogscale)
        self.paramhistory["channellogscale"].append(self.learned_channellogscale)
    
    def get_aux_loss(self):
        """ Auxiliary L1 losses for regularisation etc.
        """
        L1_emb = self.config["L1weight_embedding"] * self.pem.get_L1()
        L1_stain = self.config["L1weight_stainscale"] * self.param_stainscale.L1
        L1_channel = self.config["L1weight_channelscale"] * self.param_channelscale.L1
        loss = LossContainer({"l1_embedding":L1_emb,
                              "l1_stainlogscale":L1_stain,
                              "l1_channellogscale":L1_channel,
                              "loss_transformer":L1_emb+L1_stain+L1_channel,
                             })
        return loss
    
    @property
    def device(self):
        """ Device the module is currently on.
        """
        return self.param_stainscale.device
    
    def freeze(self):
        """ Freeze all parameters.
        """
        self.pem.freeze()
        self.param_stainscale.freeze()
        self.param_channelscale.freeze()
    
    def extend(self, moe, extend_inds):
        """ Extend parameters by new batch indices.
            All new batches are trainable by default, use .anchor_batches is this should be modified.
        """
        self.moe = moe
        self.pem.extend(moe, extend_inds)
        if self.config["key_scale_stains"] in extend_inds:
            self.param_stainscale.extend(extend_inds[self.config["key_scale_stains"]])
        if self.config["key_scale_channels"] in extend_inds:
            self.param_channelscale.extend(extend_inds[self.config["key_scale_channels"]])
    
    def plot_history_embedding(self, legend=False):
        """ Plot embedding history.
        """
        history_embedding = {key:np.asarray([ph[key] for ph in self.paramhistory["embedding"]]) for key in self.pem.batch_keys}

        labels = [i for key in self.pem.batch_keys for i in key+"__"+self.paramhistory["embedding"][0][key].index]
        cmap = get_cmap(labels)

        fig, ax = plt.subplots(figsize=(15,10))
        counter = 0
        for key in history_embedding:
            for i in range(history_embedding[key].shape[1]):
                ax.plot(history_embedding[key][:,i], color=cmap[labels[counter]], linewidth=1, alpha=.5)
                counter += 1
        ax.set_xlabel("Step", size=20)
        ax.set_ylabel("Embedding", size=20)
        if legend: ax.legend(handles=cmap_to_legendhandles(cmap), loc="upper left")
    
    def plot_history_embedding_byexpvar(self):
        """ Plot embedding history.
        """
        history_embedding = {key:np.asarray([ph[key] for ph in self.paramhistory["embedding"]]) for key in self.pem.batch_keys}
        
        explained_logvariance = {key: np.log(self.pem.panelconfig.pca_explained_variance.loc[self.paramhistory["embedding"][0][key].index].to_numpy())
                                 for key in self.paramhistory["embedding"][0]}
        ex_max = max([explained_logvariance[key].max() for key in explained_logvariance])
        ex_min = min([explained_logvariance[key].min() for key in explained_logvariance])
        
        cmap = matplotlib.colormaps["inferno"]
        norm = lambda x: 1 - (x - ex_min) / (ex_max - ex_min)
        
        fig, ax = plt.subplots(figsize=(15,10))
        counter = 0
        for key in history_embedding:
            for i in range(history_embedding[key].shape[1]):
                ax.plot(history_embedding[key][:,i], color=cmap(norm(explained_logvariance[key][i])), linewidth=1, alpha=.9)
                counter += 1
        ax.set_xlabel("Step", size=20)
        ax.set_ylabel("Embedding", size=20)

    def plot_logscale_history(self, legend=False):
        """ Plot scale factor history.
        """
        history_logscale_stain = {key:np.asarray([ph.loc[key] for ph in self.paramhistory["stainlogscale"]]) for key in self.moe.labels[self.config["key_scale_stains"]]}
        history_logscale_channel = {key:np.asarray([ph.loc[key] for ph in self.paramhistory["channellogscale"]]) for key in self.moe.labels[self.config["key_scale_channels"]]}
        
        labels_stain = self.pem.stains
        labels_channel = self.pem.channels
        
        cmap_stain = get_cmap(labels_stain)
        cmap_channel = get_cmap(labels_channel)
        
        fig, ax = plt.subplots(1,2, figsize=(20,7))
        
        for key in history_logscale_stain:
            for i, label in enumerate(labels_stain):
                ax[0].plot(history_logscale_stain[key][:,i], color=cmap_stain[label], linewidth=1, alpha=.5)
        for key in history_logscale_channel:
            for i, label in enumerate(labels_channel):
                ax[1].plot(history_logscale_channel[key][:,i], color=cmap_channel[label], linewidth=1, alpha=.5)
        
        ax[0].set_xlabel("Step", size=20)
        ax[1].set_xlabel("Step", size=20)
        ax[0].set_ylabel("Scale Factor [Log]", size=20)
        ax[1].set_ylabel("Scale Factor [Log]", size=20)
        ax[0].set_title("Marker Scaling", size=20)
        ax[1].set_title("Channel Scaling", size=20)
        if legend:
            ax[0].legend(handles=cmap_to_legendhandles(cmap_stain), loc="upper left")
            ax[1].legend(handles=cmap_to_legendhandles(cmap_channel), loc="upper left")
    
    def get_standardisationparams_fromad(self, ad, label_include_stain=False):
        """ Get all relevant standardisation parameters from adata, uses adata.uns to get the batch indices.
            
            If label_include_stain, name also includes the fluorophore, otherwise just the marker.

            returns:
                - channel scaling, pd.Series with index channels.
                - stain scaling, pd.Series with index stain_marker_name.
                - spectra, pd.DataFrame with index channels and columns stain_marker_name.
                - unmx_int_matrix, pd.DataFrame with index channels, columns stain_marker_name; multiplying raw data (after rainbow calibration) with this matrix yields standardised unmixed marker abundances.
                - spectra_embedding, np.array of component embedding of spectra.
        """
        bidx = self.moe.transform_ad(ad)

        channelscale = pd.Series(self.param_channelscale.factor[bidx[self.config["key_scale_channels"]]].detach().cpu().numpy()[0], index=self.channels)
        stainscale = pd.Series(self.param_stainscale.factor[bidx[self.config["key_scale_stains"]]].detach().cpu().numpy()[0],
                               index=self.stain_marker_name if label_include_stain else self.pem.markers)
        spectra = pd.DataFrame(self.pem.get_spectra(bidx).detach().cpu().numpy()[0].T, index=self.channels,
                               columns=self.stain_marker_name if label_include_stain else self.pem.markers)
        
        spectra_inv = invert_spectra(spectra).T
        unmx_int_matrix = channelscale.to_numpy()[:,None] * spectra_inv * stainscale.to_numpy()[None]
        
        spectra_embedding = self.pem(bidx, mask_unused=True)[:,0].detach().cpu().numpy()

        return channelscale, stainscale, spectra, unmx_int_matrix, spectra_embedding
    
    def add_standardisation_parameters(self, adata, rbint_key="rainbow_calibration_factor", label_include_stain=False):
        """ Add standardisation parameters to adata.

                - adata.var["sfm_calibration_factor"]: channel calibration factor, padded by nan for other vars
                - adata.varm["sfm_stain_factor"]: stain standardisation factor, column stain, repeated for all vars for easier storage
                - adata.varm["sfm_spectra"]: fitted stain spectra, padded by nan for other vars
                - adata.varm["sfm_standardise_unmx"]: matrix for standardisation and unmixing on calibrated data, adata.to_df(layer="calibrated") @ adata.varm["sfm_standardise_unmx"] yields properly unmixed and calibrated data
                - adata.varm["rb_sfm_standardise_unmx"]: as above but also including the rainbow calibration factors, adata.to_df(layer="raw") @ adata.varm["rb_sfm_standardise_unmx"] yields properly standardised and calibrated data, only added if adata.var[rbint_key] is present
                - adata.uns["panel_embedding"]: np.array of component embedding of spectra
        """
        channelscale, stainscale, spectra, unmx_int_matrix, spectra_embedding = self.get_standardisationparams_fromad(adata, label_include_stain=label_include_stain)

        adata.var["sfm_calibration_factor"] = np.nan
        adata.var.loc[channelscale.index, "sfm_calibration_factor"] = channelscale

        for c in adata.var.index[~np.isin(adata.var.index,spectra.index)]:
            spectra.loc[c] = np.nan
            unmx_int_matrix.loc[c] = 0.
        adata.varm["sfm_spectra"] = spectra.loc[adata.var.index]
        adata.varm["sfm_standardise_unmx"] = unmx_int_matrix.loc[adata.var.index]
        if rbint_key in adata.var:
            adata.varm["rb_sfm_standardise_unmx"] = adata.var[rbint_key].fillna(0).to_numpy()[:,None] * adata.varm["sfm_standardise_unmx"]

        stainscale_ext = pd.DataFrame(np.repeat([stainscale],adata.shape[1], axis=0), columns=stainscale.index, index=adata.var.index)
        adata.varm["sfm_stain_factor"] = stainscale_ext
        
        adata.uns["panel_embedding"] = spectra_embedding


class ScalingTransformer(Module):
    """ Fits the scaling factors per stain.
        Transforms data into ArcSinh scale for MMD.
        
        Doesn't do the anchoring in the normal way, but instead normalizes the parameters at eval time.
    """
    NAME = "scaling"
    
    def __init__(self, moe, pem, config, smp=None):
        super().__init__()
        self.moe = moe
        self.pem = pem
        self.smp = smp if smp is not None else DefaultMultiplier()
        
        self.stains = self.pem.stains
        self.channels = self.pem.channels
        self.stain_marker_name = self.pem.stain_marker_name
        
        self.config = config
        
        self.plot_auxloss_dct = {}
        if np.any(self.config["scale_stains"]):
            self.plot_auxloss_dct.update({"l1_stainlogscale":  "L1 Stain Scale"})
        else:
            warnings.warn("ScalingTransformer has no stains that get scaled! Creating the model still works, but fitting will not!")
        
        self.paramhistory = {"stainlogscale":[]}
        
        self.init_parameters()
        self.anchor_idx = None
        self.anchor_batches(self.config["anchor_bidx"])
    
    def init_parameters(self):
        self.param_stainscale = ScaleFactor(len(self.moe[self.config["key_scale_stains"]]),
                                            len(self.stains),
                                            use_channel_mask=self.config["scale_stains"], trainable=True,
                                            max_abs_value=self.config["max_abslogstain"],
                                            mean_norm=True
                                           )

        self.loss_use_stain_mask = Parameter(torch.tensor(self.config["scale_stains"].to_numpy()), requires_grad=False)
    
    def anchor_batches(self, anchor_bidx):
        """ Anchor batches to be not trainable.
            E.g. {"a":0} sets scaling to one for batch 0 in "a". 
        """
        if self.config["key_scale_stains"] in anchor_bidx:
            #self.param_stainscale.anchor_bidx(anchor_bidx[self.config["key_scale_stains"]])
            try:
                idx = int(anchor_bidx[self.config["key_scale_stains"]])
                self.anchor_idx = [idx]
            except:
                raise ValueError(f"Trying to anchor {anchor_bidx[self.config['key_scale_stains']]}, but this transformer only supports anchoring a single batch!")
            

    @classmethod
    def from_data(cls, moe, panel_spectra, smp=None, scale_stains=False, key_scale_stains="batch", arcsinh_cofactor=5000, L1weight_stainscale=0., max_abslogstain=None, anchor_bidx={}):
        """ 
            
            :param moe: MultiOrdinalEncoder. Batch encoder for all relevant batch keys.
            
            :param panel_spectra: pd.DataFrame. DataFrame containing the dye spectra, where the dyes are the index, and columns the fluorescence channels.
            
            :params smp: None, ScaleMultiplier. Optional class that takes unmixed values and stain scaling and multiplies them; can be used to selectively detach gradients for the scaling factors if necessary. Can be set here, but usually requires transformer at instantiation! Better to set using .set_smp after creating transformer!
            
            :params scale_stains: bool, pd.Series. Whether model should include a scaling factor for the stains. Either bool, or Series with index stains and entry bool.
            
            :param key_scale_stains: str. Batch key that affects the stain scaling.
            
            :param arcsinh_cofactor: float. Cofactor of ArcSinh used for MMD. Should be set to a multiple (3x or so) of the normal cofactor; setting this higher encourages the fit to focus on the markers with high abundances instead of fitting the noise.
            
            :param L1weight_stainscale: float. Weighting of L1 stain scaling regularisation, set to zero to disable.
            
            :param max_abslogstain: None, float. If not None, log of stain scale factor is clipped to +- this.
            
            :params anchor_bidx: dict. Anchor all batch indices in dict to not be trainable. E.g. {"a":[0,1]} disables training for all parameters associated with indices [0,1] in batch "a"; also passes this through to pem.
        """
        pem = FixedPanelModule.from_data(panel_spectra)
        
        config = {}
        config["scale_stains"] = pd.Series(scale_stains, index=pem.stains) if isinstance(scale_stains, bool) else scale_stains.loc[pem.stains]
        config["key_scale_stains"] = key_scale_stains
        config["arcsinh_cofactor"] = arcsinh_cofactor
        
        config["L1weight_stainscale"] = L1weight_stainscale
        
        config["max_abslogstain"] = max_abslogstain
        
        config["anchor_bidx"] = anchor_bidx
        #config["has_anchor"] = (config["key_scale_stains"] in config["anchor_bidx"])
        
        sft = cls(moe, pem, config, smp)
        return sft

    def export(self, include_history=True):
        """ Export self to dict.
        """
        exportdct = {}
        exportdct["pem"] = self.pem.export()
        exportdct["moe"] = self.moe.export()
        exportdct["smp"] = self.smp.export()
        exportdct["smp_name"] = self.smp.NAME
        exportdct["config"] = pdsave_expand_dict(self.config)
        exportdct["state_dict"] = self.state_dict()
        if include_history:
            exportdct["paramhistory"] = self.paramhistory
        
        return exportdct
    
    @classmethod
    def from_exported(cls, exported, cls_smp=None):
        """ Initialize from exported. Optionally needs smp class if custom implementation is used.
        """
        pem = pick_panel_cls(exported["pem"]["class"]).from_exported(exported["pem"])
        moe = MultiOrdinalEncoder.from_exported(exported["moe"])
        config = pdsave_collect_dict(exported["config"])
        
        sft = cls(moe, pem, config)
        
        if "smp" in exported:
            if cls_smp is None: cls_smp = pick_smp_cls(exported["smp_name"])
            sft.set_smp(cls_smp.from_exported(exported["smp"], sft))
        
        sft.load_state_dict(exported["state_dict"])
        if "paramhistory" in exported:
            sft.paramhistory = exported["paramhistory"]
        return sft
    
    def set_smp(self, smp):
        """ Optionally set scale multiplier; i.e. class that takes unmixed values and stain scaling and multiplies them,
            E.g. use UnmxErrorExclusionMultiplier to selectively detach gradients for the scaling factors if necessary.
        """
        self.smp = smp
        
    def forward(self, x, bidx):
        """ Take input and batch idx, output standardised, unmixed and transformed version on which to apply MMD loss.
        """
        return self.apply_transformation( self.standardise(x, bidx) )
    
    def apply_transformation(self, x):
        """ Transform intensities into scale on which MMD should be calculated, e.g. ArcSinh.
        """
        return torch.arcsinh(x/self.config["arcsinh_cofactor"])
    
    def standardise(self, x, bidx, evl=False):
        """ Return standardised version of input, i.e. if input is intensities output should be standardised and unmixed intensities.
        """
        xc = self.pem.get_unmixed(x, bidx).detach() # detach here, but there shouldn't be any gradients anyway
        factor = self.param_stainscale.factor
        if evl and self.anchor_idx is not None:
            factor = factor / factor[self.anchor_idx]
        xc = self.smp( xc, factor[bidx[self.config["key_scale_stains"]]] ) # just a multiplication, but optionally with some detached gradients
        return xc

    @property
    def learned_stainlogscale(self):
        logfactor = self.param_stainscale.logfactor.detach().cpu()
        if self.anchor_idx is not None:
            logfactor = logfactor - logfactor[self.anchor_idx]
        df_logfactor = pd.DataFrame(logfactor.numpy(),
                                    index=self.moe.labels[self.config["key_scale_stains"]],
                                    columns=self.stains)
        return df_logfactor

    @property
    def learned_stainscale(self):
        return np.exp(self.learned_stainlogscale)
    
    def append_history(self):
        """ Append current parameters to parameter history.
        """
        self.paramhistory["stainlogscale"].append(self.learned_stainlogscale)
    
    def get_aux_loss(self):
        """ Auxiliary L1 losses for regularisation etc.
        """
        L1_stain = self.config["L1weight_stainscale"] * self.param_stainscale.L1
        loss = LossContainer({"l1_stainlogscale":L1_stain,
                              "loss_transformer":L1_stain,
                             })
        return loss
    
    @property
    def device(self):
        """ Device the module is currently on.
        """
        return self.param_stainscale.device
    
    def freeze(self):
        """ Freeze all parameters.
        """
        self.param_stainscale.freeze()
    
    def extend(self, moe, extend_inds):
        """ Extend parameters by new batch indices.
            All new batches are trainable by default, use .anchor_batches is this should be modified.
        """
        self.moe = moe
        if self.config["key_scale_stains"] in extend_inds:
            self.param_stainscale.extend(extend_inds[self.config["key_scale_stains"]])

    def plot_logscale_history(self, legend=False):
        """ Plot scale factor history.
        """
        history_logscale_stain = {key:np.asarray([ph.loc[key] for ph in self.paramhistory["stainlogscale"]]) for key in self.moe.labels[self.config["key_scale_stains"]]}
        
        labels_stain = self.pem.stains
        
        cmap_stain = get_cmap(labels_stain)
        
        fig, ax = plt.subplots(1,1, figsize=(10,7))
        
        for key in history_logscale_stain:
            for i, label in enumerate(labels_stain):
                ax.plot(history_logscale_stain[key][:,i], color=cmap_stain[label], linewidth=1, alpha=.5)
        
        ax.set_xlabel("Step", size=20)
        ax.set_ylabel("Scale Factor [Log]", size=20)
        ax.set_title("Marker Scaling", size=20)
        if legend:
            ax.legend(handles=cmap_to_legendhandles(cmap_stain), loc="upper left")

    def get_standardisationparams_fromad(self, ad, label_include_stain=False):
        """ Get all relevant standardisation parameters from adata, uses adata.uns to get the batch indices.
            
            If label_include_stain, name also includes the fluorophore, otherwise just the marker.

            returns:
                - stain scaling, pd.Series with index stain_marker_name.
                - spectra, pd.DataFrame with index channels and columns stain_marker_name.
                - unmx_int_matrix, pd.DataFrame with index channels, columns stain_marker_name; multiplying raw data (after rainbow calibration) with this matrix yields standardised unmixed marker abundances.
        """
        bidx = self.moe.transform_ad(ad)

        factor = self.param_stainscale.factor.detach().cpu()
        if self.anchor_idx is not None:
            factor = factor / factor[self.anchor_idx]
        stainscale = pd.Series(factor[bidx[self.config["key_scale_stains"]]].numpy()[0],
                               index=self.stain_marker_name if label_include_stain else self.pem.markers)
        spectra = pd.DataFrame(self.pem.get_spectra(bidx).detach().cpu().numpy().T, index=self.channels,
                               columns=self.stain_marker_name if label_include_stain else self.pem.markers)
        
        spectra_inv = invert_spectra(spectra).T
        unmx_int_matrix = spectra_inv * stainscale.to_numpy()[None]
        
        return stainscale, spectra, unmx_int_matrix
    
    def add_standardisation_parameters(self, adata, label_include_stain=False):
        """ Add standardisation parameters to adata.

                - adata.varm["sfm_stain_factor"]: stain standardisation factor, column stain, repeated for all vars for easier storage
                - adata.varm["sfm_spectra"]: stain spectra, padded by nan for other vars
                - adata.varm["sfm_standardise_unmx"]: matrix for standardisation and unmixing on calibrated data, adata.to_df(layer="calibrated") @ adata.varm["sfm_standardise_unmx"] yields properly unmixed and calibrated data
        """
        stainscale, spectra, unmx_int_matrix = self.get_standardisationparams_fromad(adata, label_include_stain=label_include_stain)

        for c in adata.var.index[~np.isin(adata.var.index, spectra.index)]:
            spectra.loc[c] = np.nan
            unmx_int_matrix.loc[c] = 0.
        adata.varm["sfm_spectra"] = spectra.loc[adata.var.index]
        adata.varm["sfm_standardise_unmx"] = unmx_int_matrix.loc[adata.var.index]

        #stainscale_ext = pd.DataFrame(np.repeat([stainscale],adata.shape[1], axis=0), columns=stainscale.index, index=adata.var.index)
        adata.var["sfm_stain_factor"] = stainscale

