"""Transformers of the overlap fit: map data of a batch to standardised, unmixed intensities."""
import numpy as np
import pandas as pd
import torch
import warnings
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
    """Return the transformer class for its saved name (``NAME``).

    Parameters
    ----------
    model : {'dummy', 'spectralfit', 'scaling'}
        Name stored in an exported model (``transformer_name``).

    Returns
    -------
    type
        `DummyTransformer`, `SpectralFitTransformer` or `ScalingTransformer`.

    Raises
    ------
    NotImplementedError
        For an unknown name (custom classes are passed to `OverlapFitModel.from_saved` directly).
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
    """Return the panel module class for its saved name.

    Parameters
    ----------
    panel : {'embedding', 'fixed'}
        Saved ``class`` entry of an exported panel module.

    Returns
    -------
    type
        `PanelEmbeddingModule` or `FixedPanelModule`.

    Raises
    ------
    NotImplementedError
        For an unknown name.
    """
    if panel=="embedding":
        return PanelEmbeddingModule
    elif panel=="fixed":
        return FixedPanelModule
    else:
        raise NotImplementedError(f"Cannot find panel class {panel}!")


class DummyTransformer(Module):
    """Identity transformer that shows the interface a transformer provides to `OverlapFitModel`.

    A transformer has ``NAME``, ``channels``, ``stains``, ``stain_marker_name``, ``plot_auxloss_dct`` (aux loss key
    to plot label), the methods below, and maps data of shape (events, channel) with batch indices ``bidx`` (dict
    of batch key to torch.Tensor of int, shape (events,) or (1,)) to standardised data of shape (events, stain).
    Create it with `from_data`. This class itself returns its input unchanged (stains = channels) and has no
    trainable parameters, so it works with `OverlapFitModel` for evaluation (distances between batches without
    standardisation, `add_ad_standardised`, export and saving), but `OverlapFitModel.fit` has nothing to fit.
    """
    NAME = "dummy"
    
    def __init__(self):
        super().__init__()
        self.plot_auxloss_dct = {}
        self.param_dummy = Parameter(torch.tensor([], dtype=torch.float32), requires_grad=False)
    
    @classmethod
    def from_data(cls, channels, stains=None, *args, **kwargs):
        """Create the identity transformer for the given channels.

        Parameters
        ----------
        channels : list of str
            Input channels.
        stains : list of str, optional
            Names of the output columns, one per channel (the data are passed on unchanged). Default: `channels`.
        *args, **kwargs
            Ignored.

        Returns
        -------
        DummyTransformer

        Raises
        ------
        ValueError
            If `stains` does not have one name per channel.
        """
        stains = list(channels) if stains is None else list(stains)
        if len(stains)!=len(channels):
            raise ValueError(f"DummyTransformer is the identity, it needs one stain per channel "
                             f"({len(stains)} stains, {len(channels)} channels)!")
        dt = cls()
        dt.channels = list(channels)
        dt.stains = stains
        dt.stain_marker_name = stains
        
        return dt
    
    def export(self, include_history=False):
        """Export to a dict (channels and stains).

        Parameters
        ----------
        include_history : bool, default False
            Ignored (there is no parameter history).

        Returns
        -------
        dict
        """
        exported = {"channels":self.channels, "stains":self.stains}
        return exported
    
    @classmethod
    def from_exported(cls, exported, cls_smp=None):
        """Restore from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export`.
        cls_smp : type, optional
            Ignored (no scale multiplier).

        Returns
        -------
        DummyTransformer
        """
        dt = cls()
        dt.channels = exported["channels"]
        dt.stains = exported["stains"]
        dt.stain_marker_name = dt.stains
        
        return dt
        
    def forward(self, x, bidx):
        """Standardise and transform the data onto the scale of the loss (``apply_transformation(standardise(x))``).

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Input data (linear scale).
        bidx : dict of str to torch.Tensor
            Batch indices per batch key.

        Returns
        -------
        torch.Tensor of shape (events, stain)
        """
        return self.apply_transformation( self.standardise(x, bidx) )
    
    def apply_transformation(self, x):
        """Transform standardised intensities onto the scale of the loss (identity here; arcsinh in the real
        transformers).

        Parameters
        ----------
        x : torch.Tensor of shape (events, stain)

        Returns
        -------
        torch.Tensor of shape (events, stain)
        """
        return x
    
    @property
    def loss_use_stain_mask(self):
        """Stains used in the loss: torch.Tensor of bool, shape (stain,), all True here."""
        return torch.tensor([True]*len(self.stains), dtype=bool, device=self.device)

    def standardise(self, x, bidx, evl=False):
        """Return the standardised, unmixed intensities (the input unchanged here).

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Input data (linear scale).
        bidx : dict of str to torch.Tensor
            Batch indices per batch key.
        evl : bool, default False
            Ignored (evaluation mode of the real transformers).

        Returns
        -------
        torch.Tensor of shape (events, stain)
        """
        return x
    
    def append_history(self):
        """Append the current parameters to the parameter history (nothing here)."""
        pass
    
    def get_aux_loss(self):
        """Return the regularisation losses.

        Returns
        -------
        cytovanni.torch.LossContainer
            At least ``'loss_transformer'``, the sum that is added to the fit loss (0 here).
        """
        loss = LossContainer({"loss_transformer":0.})
        return loss
    
    @property
    def device(self):
        """torch.device the module is on."""
        return self.param_dummy.device
    
    def freeze(self):
        """Freeze all parameters (nothing here)."""
        pass
    
    def extend(self, moe, extend_inds):
        """Extend the parameters to new batches (nothing here).

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Extended batch encoder.
        extend_inds : dict of str to list of int
            New batch indices per batch key, output of `MultiOrdinalEncoder.extend`.
        """
        pass


class ScaleFactor(Module):
    """Log scale factors of shape (batch, feature), e.g. per batch and stain or channel.

    Wraps a `cytovanni.torch.ExtendableParameter` that holds the log factors (initialised at 0, i.e. factor 1).
    With `max_abs_value`, the output log factors are clamped to [-max_abs_value, max_abs_value] by a
    straight-through clamp (`cytovanni.torch.diffclamp`): only the forward value is clamped, the gradient passes
    through unchanged, and the stored parameter can exceed the limit.

    Parameters
    ----------
    Nbatch : int
        Number of batches (rows).
    Nchannel : int
        Number of features (columns): stains or channels.
    anchoridxs : list of int, default []
        Batches whose factors are not trainable (stay at 1).
    use_channel_mask : array-like of bool, shape (Nchannel,), optional
        Features that are scaled; the others keep factor 1 for all batches, also for batches added later by
        `extend`. Default: all features.
    trainable : bool, default False
        Whether the factors are trainable at all.
    mean_norm : bool, default False
        Normalise the log factors to mean 0 over batches (for every feature).
    max_abs_value : float, optional
        Limit of the absolute log factor (straight-through clamp, see above). Default: no limit.
    """
    def __init__(self, Nbatch, Nchannel, anchoridxs=[], use_channel_mask=None, trainable=False, mean_norm=False, max_abs_value=None):
        super().__init__()
        self.parameter = ExtendableParameter((Nbatch,Nchannel), 0, 0., allow_grad=trainable, axis_norm=0 if mean_norm else None)
        self.parameter.param_mask_trainable.data[(list(anchoridxs), slice(None))] = 0.
        # features that are never scaled (also not for batches added later by extend)
        self.idx_fixed_features = [] if use_channel_mask is None else list(np.where(~np.asarray(use_channel_mask, dtype=bool))[0])
        if len(self.idx_fixed_features)>0:
            self.parameter.param_mask_trainable.data[(slice(None), self.idx_fixed_features)] = 0.
        
        self.max_abs_value = max_abs_value
    
    def anchor_bidx(self, indices):
        """Freeze the factors of the given batches at their current values.

        Parameters
        ----------
        indices : int or list of int
            Batch indices.
        """
        self.parameter.freeze_selective((indices, slice(None)))

    def add_to_fixed(self, values, bidx=None, cidx=None):
        """Add values to the fixed (non-trainable) part of the log factors.

        Parameters
        ----------
        values : torch.Tensor
            Log values to add, shape of the indexed part (batch, feature).
        bidx : index expression, optional
            Batch indices to update. Default: all batches.
        cidx : index expression, optional
            Feature indices to update. Default: all features.

        Warns
        -----
        OverlapStandardisationWarning
            If a log factor then exceeds `max_abs_value` (its output is clamped).
        """
        if bidx is None and cidx is None:
            sl = None
        else:
            sl = (slice(None) if bidx is None else bidx,
                  slice(None) if cidx is None else cidx)
        self.parameter.add_to_fixed(values, sl=sl)
        if self.max_abs_value is not None and (self.parameter().detach().abs() > self.max_abs_value).any():
            warnings.warn(f"Log scale exceeds max_abs_value={self.max_abs_value} after adding values, it is clipped!", OverlapStandardisationWarning)
    
    def forward(self):
        """Return the log factors, torch.Tensor of shape (batch, feature), clamped if `max_abs_value` is set."""
        if self.max_abs_value is None:
            return self.parameter()
        else:
            return diffclamp(self.parameter(), -self.max_abs_value, self.max_abs_value)
    
    def freeze(self):
        """Make all current factors non-trainable."""
        self.parameter.freeze()
    
    def extend(self, inds):
        """Add rows for new batches; the existing factors are frozen.

        New batches start at factor 1 and are trainable, except for the features excluded by `use_channel_mask`,
        which stay fixed at factor 1.

        Parameters
        ----------
        inds : list of int
            New batch indices (the parameter grows to ``max(inds) + 1`` rows).
        """
        Nold = self.parameter.shape[0]
        self.parameter.extend(inds)
        if len(self.idx_fixed_features)>0:
            self.parameter.param_mask_trainable.data[Nold:, self.idx_fixed_features] = 0.
    
    @property
    def shape(self):
        """Shape (batch, feature) of the factors."""
        return self.parameter.shape
    
    @property
    def device(self):
        """torch.device of the factors."""
        return self.parameter.device
    
    @property
    def logfactor(self):
        """Log factors (natural log), torch.Tensor of shape (batch, feature), with gradient."""
        return self()
    
    @property
    def factor(self):
        """Factors, ``exp(logfactor)``, torch.Tensor of shape (batch, feature)."""
        return self().exp()
    
    @property
    def L1(self):
        """L1 regularisation: absolute log factors, mean over batches, summed over features (scalar tensor)."""
        return self.logfactor.abs().mean(0).sum()

class SpectralFitTransformer(Module):
    """Transformer that fits the dye spectra per batch (panel embedding) and optional scale factors.

    Standardisation of data x (events, channel) of a batch b, on the calibrated (linear) scale::

        standardise(x) = unmix(x * channel_factor[b], spectra(embedding[b])) * stain_factor[b]

    with the spectra generated from the fitted embedding by the `PanelEmbeddingModule` (`pem`), one channel factor
    per batch of `key_scale_channels` and channel, and one stain factor per batch of `key_scale_stains` and stain.
    The loss is computed on ``arcsinh(standardise(x) / arcsinh_cofactor)``, over all stains. The anchor batches
    (`anchor_bidx`) keep their parameters fixed.

    Create it with `from_data` (which builds the `config` dict) or restore it with `from_exported`.

    Parameters
    ----------
    moe : MultiOrdinalEncoder
        Batch encoder for all batch keys of the model.
    pem : PanelEmbeddingModule
        Panel module that generates the spectra from the per-batch embedding.
    config : dict
        Settings, as built by `from_data` (keys ``scale_stains``, ``key_scale_stains``, ``scale_channels``,
        ``key_scale_channels``, ``arcsinh_cofactor``, ``L1weight_embedding``, ``L1weight_stainscale``,
        ``L1weight_channelscale``, ``max_abslogstain``, ``max_abslogchannel``, ``anchor_bidx``).
    smp : ScaleMultiplier, optional
        Applies the stain factors; default `DefaultMultiplier`. See `set_smp`.
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
        """Create the stain and channel scale factors (`ScaleFactor`, factor 1) from `config`."""
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
        """Stains used in the loss: torch.Tensor of bool, shape (stain,), all True here."""
        return torch.tensor([True]*len(self.stains), dtype=bool, device=self.device)
    
    def anchor_batches(self, anchor_bidx):
        """Freeze the parameters of the given batches (scale factors and, via `pem`, the embedding).

        Parameters
        ----------
        anchor_bidx : dict of str to list of int
            Batch indices per batch key, e.g. ``{"a": [0, 1]}`` freezes all parameters indexed by batches 0 and 1 of
            key "a" (build it with `MultiOrdinalEncoder.transform_dict_simple`).
        """
        if self.config["key_scale_stains"] in anchor_bidx:
            self.param_stainscale.anchor_bidx(anchor_bidx[self.config["key_scale_stains"]])
        if self.config["key_scale_channels"] in anchor_bidx:
            self.param_channelscale.anchor_bidx(anchor_bidx[self.config["key_scale_channels"]])
        self.pem.anchor_batches(anchor_bidx)

    def add_stain_logscale_toself(self, logscale):
        """Add log stain factors (e.g. of another transformer) to the fixed part of the stain scaling, in place.

        Parameters
        ----------
        logscale : pandas.DataFrame of shape (batch, stain)
            Natural-log factors, e.g. ``other.learned_stainlogscale``: index the batch labels of `key_scale_stains`
            (the same set as here), columns stains of this transformer (a subset is allowed).

        Raises
        ------
        OverlapStandardisationException
            If the batch labels differ or `logscale` has stains that this transformer does not have.

        Warns
        -----
        OverlapStandardisationWarning
            If a combined log factor exceeds `max_abslogstain` (its output is clamped).
        """
        logscale_self = self.learned_stainlogscale
        
        if not set(logscale_self.index) == set(logscale.index):
            raise OverlapStandardisationException(f"The normalization that should be incorporated uses different batches!")
        if not logscale.columns.isin(logscale_self.columns).all():
            new = [i for i in logscale.columns if not i in logscale_self.columns]
            raise OverlapStandardisationException(f"The normalization that should be incorporated has markers that are not currently present!\n{new}")
        
        logscale = logscale.loc[logscale_self.index]
        logscale_torch = torch.tensor(logscale.to_numpy())
        colind = logscale_self.columns.get_indexer(logscale.columns)
        
        self.param_stainscale.add_to_fixed(logscale_torch, cidx=colind)
    
    @classmethod
    def from_data(cls, moe, pem, smp=None,
                  scale_stains=False, key_scale_stains="batch", scale_channels=False, key_scale_channels=None,
                  arcsinh_cofactor=5000, L1weight_embedding=0., L1weight_stainscale=0., L1weight_channelscale=0.,
                  max_abslogstain=None, max_abslogchannel=None, anchor_bidx={}):
        """Create a spectral-fit transformer.

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Batch encoder for all batch keys of the model.
        pem : PanelEmbeddingModule
            Panel module that generates the spectra per batch; defines ``stains`` and ``channels``.
        smp : ScaleMultiplier, optional
            Applies the stain factors. Default `DefaultMultiplier`. Multipliers like `UnmxErrorExclusionMultiplier`
            need the transformer itself, so set them with `set_smp` after creating the transformer.
        scale_stains : bool or pandas.Series of bool, default False
            Fit a factor per stain: True/False for all stains, or a Series with index the stains (``pem.stains``).
            Stains that are not scaled keep factor 1.
        key_scale_stains : str, default "batch"
            Batch key of the stain factors (one factor per batch of this key and stain).
        scale_channels : bool or pandas.Series of bool, default False
            Fit a factor per channel, as `scale_stains` (Series index: the channels).
        key_scale_channels : str, optional
            Batch key of the channel factors. Default: `key_scale_stains`.
        arcsinh_cofactor : float, default 5000
            Cofactor (raw intensity units) of the arcsinh transform the loss is computed on. A cofactor several
            times (about 3x) the usual one puts more weight on the high-abundance markers and less on the noise
            around zero.
        L1weight_embedding : float, default 0
            Weight of the L1 regularisation of the embedding (``pem.get_L1``); 0 disables it.
        L1weight_stainscale : float, default 0
            Weight of the L1 regularisation of the log stain factors (`ScaleFactor.L1`); 0 disables it.
        L1weight_channelscale : float, default 0
            Weight of the L1 regularisation of the log channel factors; 0 disables it.
        max_abslogstain : float, optional
            Limit of the absolute log stain factor: the output is clamped, straight-through (the gradient passes
            and the stored parameter can exceed the limit, see `ScaleFactor`). Default: no limit.
        max_abslogchannel : float, optional
            The same for the log channel factors.
        anchor_bidx : dict of str to list of int, default {}
            Batches whose parameters stay fixed, e.g. ``{"a": [0, 1]}`` for batches 0 and 1 of key "a"; applied to
            the scale factors of the matching keys and passed on to `pem`. Build it with
            `MultiOrdinalEncoder.transform_dict_simple`.

        Returns
        -------
        SpectralFitTransformer

        Examples
        --------
        >>> anchor_bidx = moe.transform_dict_simple({"batch": ["18-APR-2024"]})
        >>> transformer = SpectralFitTransformer.from_data(moe, pem, scale_stains=True, key_scale_stains="batch",
        ...                                                 L1weight_embedding=2e-7, L1weight_stainscale=1e-7,
        ...                                                 anchor_bidx=anchor_bidx)
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
    
    def export(self, include_history=False):
        """Export the transformer (panel module, encoder, scale multiplier, config, parameters) to a dict.

        Parameters
        ----------
        include_history : bool, default False
            Include the parameter history (``paramhistory``).

        Returns
        -------
        dict
            Restore with `from_exported`.
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
        """Restore a transformer from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export`.
        cls_smp : type, optional
            Scale multiplier class, for a custom one. Default: chosen by its saved name.

        Returns
        -------
        SpectralFitTransformer or ScalingTransformer
            The restored transformer, of the class this is called on (on the CPU).
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
        """Set the scale multiplier that applies the stain factors to the unmixed data.

        E.g. `UnmxErrorExclusionMultiplier`, to block the gradient of some stain factors for selected events.

        Parameters
        ----------
        smp : ScaleMultiplier
            The multiplier (created with this transformer).
        """
        self.smp = smp
        
    def forward(self, x, bidx):
        """Standardise the data and transform it onto the scale of the loss.

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Input data on the scale the model is fitted on (usually calibrated, linear).
        bidx : dict of str to torch.Tensor of int
            Batch indices per batch key, shape (events,) or (1,) (e.g. from `SampleOverlapDataset.sample_single`
            or `MultiOrdinalEncoder.transform_ad`).

        Returns
        -------
        torch.Tensor of shape (events, stain)
            ``arcsinh(standardise(x, bidx) / arcsinh_cofactor)``.
        """
        return self.apply_transformation( self.standardise(x, bidx) )
    
    def apply_transformation(self, x):
        """Transform standardised intensities onto the scale of the loss: ``arcsinh(x / arcsinh_cofactor)``.

        Parameters
        ----------
        x : torch.Tensor
            Standardised, unmixed intensities (linear scale).

        Returns
        -------
        torch.Tensor
            Same shape as `x`.
        """
        return torch.arcsinh(x/self.config["arcsinh_cofactor"])
    
    def standardise(self, x, bidx, evl=False):
        """Return the standardised, unmixed intensities.

        Applies the channel factors, unmixes with the batch's spectra and applies the stain factors (via the scale
        multiplier).

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Input data on the scale the model is fitted on (usually calibrated, linear).
        bidx : dict of str to torch.Tensor of int
            Batch indices per batch key, shape (events,) or (1,).
        evl : bool, default False
            Not used here (same signature as `ScalingTransformer.standardise`).

        Returns
        -------
        torch.Tensor of shape (events, stain)
            Standardised, unmixed intensities (linear scale).
        """
        x = x * self.param_channelscale.factor[bidx[self.config["key_scale_channels"]]]
        xc = self.pem.get_unmixed(x, bidx)
        xc = self.smp( xc, self.param_stainscale.factor[bidx[self.config["key_scale_stains"]]] ) # just a multiplication, but optionally with some detached gradients
        return xc
    
    def get_freeparam_standardisetransf(self, bidx):
        """Return detached copies of a batch's parameters and a function that applies them, for gradient analyses.

        The copies (embedding and log channel and stain factors of the batch) are leaf tensors that accumulate
        gradients. Unused components and unscaled stains or channels enter with their current value, but are not
        masked, so the gradient with respect to every parameter can be read off. The scale multiplier is not used
        (plain multiplication).

        Parameters
        ----------
        bidx : dict of str to torch.Tensor of int
            Batch indices of one batch per batch key, shape (1,).

        Returns
        -------
        params : dict of str to torch.Tensor
            ``'embedding'``, ``'channellogscale'``, ``'stainlogscale'``.
        fct_standardisetransf : callable
            Maps x (events, channel) to ``arcsinh(standardised / arcsinh_cofactor)`` (events, stain).

        Raises
        ------
        NotImplementedError
            For a `FixedPanelModule` (no embedding).
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
        """Collect the gradients accumulated in the `params` of several `get_freeparam_standardisetransf` calls.

        Parameters
        ----------
        paramlist : list of dict
            ``params`` outputs of `get_freeparam_standardisetransf` after backward passes.

        Returns
        -------
        dict of str to pandas.DataFrame
            ``'channellogscale'`` (columns channels), ``'stainlogscale'`` (columns stains) and ``'embedding'``
            (columns the panel components); one row per entry of `paramlist`.
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
        """Log channel factors (natural log), pandas.DataFrame of shape (batch, channel).

        Index: the batch labels of `key_scale_channels`; columns: the channels.
        """
        logfactor = pd.DataFrame(self.param_channelscale.logfactor.detach().cpu().numpy(),
                                 index=self.moe.labels[self.config["key_scale_channels"]],
                                 columns=self.channels)
        return logfactor
    
    @property
    def learned_stainlogscale(self):
        """Log stain factors (natural log), pandas.DataFrame of shape (batch, stain).

        Index: the batch labels of `key_scale_stains`; columns: the stains (dyes).
        """
        logfactor = pd.DataFrame(self.param_stainscale.logfactor.detach().cpu().numpy(),
                                 index=self.moe.labels[self.config["key_scale_stains"]],
                                 columns=self.stains)
        return logfactor
    
    def append_history(self):
        """Append the current embedding and log scale factors to ``paramhistory`` (called once per fit step)."""
        self.paramhistory["embedding"].append(self.pem.get_learned_embedding())
        self.paramhistory["stainlogscale"].append(self.learned_stainlogscale)
        self.paramhistory["channellogscale"].append(self.learned_channellogscale)
    
    def get_aux_loss(self):
        """Return the weighted L1 regularisation losses.

        Returns
        -------
        cytovanni.torch.LossContainer
            ``'l1_embedding'``, ``'l1_stainlogscale'``, ``'l1_channellogscale'`` and their sum
            ``'loss_transformer'`` (added to the fit loss).
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
        """torch.device the module is on."""
        return self.param_stainscale.device
    
    def freeze(self):
        """Make all parameters (embedding, scale factors) non-trainable."""
        self.pem.freeze()
        self.param_stainscale.freeze()
        self.param_channelscale.freeze()
    
    def extend(self, moe, extend_inds):
        """Extend the parameters to new batches; the existing parameters are frozen.

        New batches start at the initial values and are trainable; use `anchor_batches` to freeze some of them.

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Extended batch encoder (after `MultiOrdinalEncoder.extend`).
        extend_inds : dict of str to list of int
            New batch indices per batch key, the output of `MultiOrdinalEncoder.extend`.
        """
        self.moe = moe
        self.pem.extend(moe, extend_inds)
        if self.config["key_scale_stains"] in extend_inds:
            self.param_stainscale.extend(extend_inds[self.config["key_scale_stains"]])
        if self.config["key_scale_channels"] in extend_inds:
            self.param_channelscale.extend(extend_inds[self.config["key_scale_channels"]])
    
    def plot_history_embedding(self, legend=False):
        """Plot the embedding of every (component, batch) per fit step, coloured by batch key and batch.

        Needs the parameter history (fit in this session, or saved with ``include_history=True``).

        Parameters
        ----------
        legend : bool, default False
            Show a legend of the colours.
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
        """Plot the embedding history as `plot_history_embedding`, coloured by the explained variance of the component.

        Darker lines are components with a larger (log) PCA explained variance
        (``pem.panelconfig.pca_explained_variance``).
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
        """Plot the log stain factors (left) and log channel factors (right) of every batch per fit step.

        Needs the parameter history. Colours: stains resp. channels.

        Parameters
        ----------
        legend : bool, default False
            Show legends of the colours.
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
        """Return the standardisation parameters of one sample's batches (batch labels from ``ad.uns``).

        Parameters
        ----------
        ad : anndata.AnnData
            Sample; ``ad.uns`` needs the keys of ``moe.necessary_keys``.
        label_include_stain : bool, default False
            Name the stains by ``stain_marker_name`` (dye and marker); otherwise by the marker names only.

        Returns
        -------
        channelscale : pandas.Series
            Channel factor per channel (index the channels).
        stainscale : pandas.Series
            Stain factor per stain (index the stain names, see `label_include_stain`).
        spectra : pandas.DataFrame of shape (channel, stain)
            Spectra of the sample's batch (index the channels, columns the stain names).
        unmx_int_matrix : pandas.DataFrame of shape (channel, stain)
            Combined standardisation and unmixing matrix: calibrated data (events, channel) @ this matrix gives the
            standardised, unmixed intensities, the same as `standardise`.
        spectra_embedding : numpy.ndarray of shape (component,)
            Embedding of the spectra (empty for a `FixedPanelModule`).
        """
        bidx = self.moe.transform_ad(ad)

        if self.pem.TRAINED_EMBEDDING:
            spectra_np = self.pem.get_spectra(bidx).detach().cpu().numpy()[0] # (batch, stain, channel)
            spectra_embedding = self.pem(bidx, mask_unused=True)[:,0].detach().cpu().numpy()
        else: # FixedPanelModule: one (stain, channel) matrix for all batches, no embedding
            spectra_np = self.pem.get_spectra(bidx).detach().cpu().numpy()
            spectra_embedding = np.zeros(0)

        channelscale = pd.Series(self.param_channelscale.factor[bidx[self.config["key_scale_channels"]]].detach().cpu().numpy()[0], index=self.channels)
        stainscale = pd.Series(self.param_stainscale.factor[bidx[self.config["key_scale_stains"]]].detach().cpu().numpy()[0],
                               index=self.stain_marker_name if label_include_stain else self.pem.markers)
        spectra = pd.DataFrame(spectra_np.T, index=self.channels,
                               columns=self.stain_marker_name if label_include_stain else self.pem.markers)
        
        spectra_inv = invert_spectra(spectra.T) # spectra is (channel, stain), invert_spectra takes (stain, channel)
        unmx_int_matrix = channelscale.to_numpy()[:,None] * spectra_inv * stainscale.to_numpy()[None]

        return channelscale, stainscale, spectra, unmx_int_matrix, spectra_embedding
    
    def add_standardisation_parameters(self, adata, calibration_factor_key="rainbow_calibration_factor", label_include_stain=False):
        """Store the standardisation parameters of the sample's batches in `adata` (in place).

        Writes (from `get_standardisationparams_fromad`):

        - ``adata.var['sfm_calibration_factor']``: channel factor, NaN for the other variables.
        - ``adata.varm['sfm_stain_factor']``: DataFrame (var, stain), the stain factors repeated in every row.
        - ``adata.varm['sfm_spectra']``: DataFrame (var, stain), the batch's spectra, NaN rows for other variables.
        - ``adata.varm['sfm_standardise_unmx']``: DataFrame (var, stain), standardisation and unmixing matrix on the
          calibrated data, 0 rows for other variables: ``adata.to_df(layer='calibrated') @
          adata.varm['sfm_standardise_unmx']`` gives the standardised, unmixed intensities.
        - ``adata.varm['rb_sfm_standardise_unmx']``, only if ``adata.var[calibration_factor_key]`` exists: the same matrix with the
          rainbow calibration factors included (channels without factor: 0), for the raw data:
          ``adata.to_df(layer='raw') @ adata.varm['rb_sfm_standardise_unmx']``.
        - ``adata.uns['panel_embedding']``: numpy.ndarray, embedding of the spectra.

        Parameters
        ----------
        adata : anndata.AnnData
            Sample; ``adata.uns`` needs the keys of ``moe.necessary_keys``, ``adata.var`` the transformer's channels.
        calibration_factor_key : str, default "rainbow_calibration_factor"
            Column of ``adata.var`` with the rainbow calibration factors (`RainbowCalibrator.add_calibration_factor`).
        label_include_stain : bool, default False
            Stain names, see `get_standardisationparams_fromad`.
        """
        channelscale, stainscale, spectra, unmx_int_matrix, spectra_embedding = self.get_standardisationparams_fromad(adata, label_include_stain=label_include_stain)

        adata.var["sfm_calibration_factor"] = np.nan
        adata.var.loc[channelscale.index, "sfm_calibration_factor"] = channelscale

        adata.varm["sfm_spectra"] = spectra.reindex(adata.var.index)
        adata.varm["sfm_standardise_unmx"] = unmx_int_matrix.reindex(adata.var.index, fill_value=0.)
        if calibration_factor_key in adata.var:
            adata.varm["rb_sfm_standardise_unmx"] = adata.var[calibration_factor_key].fillna(0).to_numpy()[:,None] * adata.varm["sfm_standardise_unmx"]

        stainscale_ext = pd.DataFrame(np.repeat([stainscale],adata.shape[1], axis=0), columns=stainscale.index, index=adata.var.index)
        adata.varm["sfm_stain_factor"] = stainscale_ext
        
        adata.uns["panel_embedding"] = spectra_embedding


class ScalingTransformer(Module):
    """Transformer that fits only one scale factor per batch and stain, with fixed spectra.

    Standardisation of data x (events, channel) of a batch b: ``unmix(x, spectra) * stain_factor[b]``, with the
    same spectra for all batches (`FixedPanelModule`). On data that is already unmixed (e.g. after a
    `SpectralFitTransformer`), identity spectra (`SampleOverlapDataset.make_dummy_spectra`) make it a pure
    rescaling of the markers, typically fitted with ``DDIST_separate=True`` (one loss per marker) in the
    `OverlapFitModel`. The loss uses only the scaled stains (`loss_use_stain_mask`), on
    ``arcsinh(standardise(x) / arcsinh_cofactor)``.

    The anchoring differs from `SpectralFitTransformer`: during the fit, the log factors are normalised to mean 0
    over batches (for every stain) and no batch is frozen. The anchor batch (at most one) only rescales the result:
    `standardise` with ``evl=True``, `learned_stainlogscale`, `learned_stainscale` and the standardisation
    parameters express all factors relative to the anchor batch (anchor factor 1).

    Create it with `from_data` or restore it with `from_exported`.

    Parameters
    ----------
    moe : MultiOrdinalEncoder
        Batch encoder for all batch keys of the model.
    pem : FixedPanelModule
        Panel module with the fixed spectra; defines ``stains`` and ``channels``.
    config : dict
        Settings, as built by `from_data` (keys ``scale_stains``, ``key_scale_stains``, ``arcsinh_cofactor``,
        ``L1weight_stainscale``, ``max_abslogstain``, ``anchor_bidx``).
    smp : ScaleMultiplier, optional
        Applies the stain factors; default `DefaultMultiplier`. See `set_smp`.

    Warns
    -----
    OverlapStandardisationWarning
        If no stain is scaled (the model can be created, but not fitted).
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
            warnings.warn("ScalingTransformer has no stains that get scaled! Creating the model still works, but fitting will not!", OverlapStandardisationWarning)
        
        self.paramhistory = {"stainlogscale":[]}
        
        self.init_parameters()
        self.anchor_idx = None
        self.anchor_batches(self.config["anchor_bidx"])
    
    def init_parameters(self):
        """Create the stain scale factors (`ScaleFactor`, mean-normalised over batches) and `loss_use_stain_mask`."""
        self.param_stainscale = ScaleFactor(len(self.moe[self.config["key_scale_stains"]]),
                                            len(self.stains),
                                            use_channel_mask=self.config["scale_stains"], trainable=True,
                                            max_abs_value=self.config["max_abslogstain"],
                                            mean_norm=True
                                           )

        self.loss_use_stain_mask = Parameter(torch.tensor(self.config["scale_stains"].to_numpy()), requires_grad=False)
    
    def anchor_batches(self, anchor_bidx):
        """Set the anchor batch, relative to which the factors are expressed after the fit (nothing is frozen).

        Parameters
        ----------
        anchor_bidx : dict of str to int or list of int
            Batch index per batch key, e.g. ``{"a": 0}`` or ``{"a": [0]}``: batch 0 of "a" gets factor 1. Only the
            entry for `key_scale_stains` is used, other keys are ignored.

        Raises
        ------
        ValueError
            If the entry for `key_scale_stains` is not a single integer index.
        """
        if self.config["key_scale_stains"] in anchor_bidx:
            idx = np.atleast_1d(anchor_bidx[self.config["key_scale_stains"]])
            if idx.shape!=(1,) or not np.issubdtype(idx.dtype, np.integer):
                raise ValueError(f"Trying to anchor {anchor_bidx[self.config['key_scale_stains']]}, but this transformer only supports anchoring a single batch index (int or list of one int)!")
            self.anchor_idx = [int(idx[0])]

    def add_stain_logscale_toself(self, logscale):
        """Add log stain factors (e.g. of another transformer) to the fixed part of the stain scaling, in place.

        Parameters
        ----------
        logscale : pandas.DataFrame of shape (batch, stain)
            Natural-log factors, e.g. ``other.learned_stainlogscale``: index the batch labels of `key_scale_stains`
            (the same set as here), columns stains of this transformer (a subset is allowed).

        Raises
        ------
        OverlapStandardisationException
            If the batch labels differ or `logscale` has stains that this transformer does not have.

        Notes
        -----
        The factors here are normalised to mean 0 over batches and expressed relative to the anchor batch, so only
        the differences of `logscale` between batches take effect: its mean over batches (without anchor) or its
        value at the anchor batch cancels.

        Warns
        -----
        OverlapStandardisationWarning
            If a combined log factor exceeds `max_abslogstain` (its output is clamped).
        """
        logscale_self = self.learned_stainlogscale
        
        if not set(logscale_self.index) == set(logscale.index):
            raise OverlapStandardisationException(f"The normalization that should be incorporated uses different batches!")
        if not logscale.columns.isin(logscale_self.columns).all():
            new = [i for i in logscale.columns if not i in logscale_self.columns]
            raise OverlapStandardisationException(f"The normalization that should be incorporated has markers that are not currently present!\n{new}")
        
        logscale = logscale.loc[logscale_self.index]
        logscale_torch = torch.tensor(logscale.to_numpy())
        colind = logscale_self.columns.get_indexer(logscale.columns)
        
        self.param_stainscale.add_to_fixed(logscale_torch, cidx=colind)
            

    @classmethod
    def from_data(cls, moe, panel_spectra, smp=None, scale_stains=False, key_scale_stains="batch", arcsinh_cofactor=5000, L1weight_stainscale=0., max_abslogstain=None, anchor_bidx={}):
        """Create a scaling transformer.

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Batch encoder for all batch keys of the model.
        panel_spectra : pandas.DataFrame of shape (stain, channel)
            Fixed spectra, index the stain (dye) names, columns the channels; e.g.
            `SampleOverlapDataset.make_dummy_spectra` for already unmixed data.
        smp : ScaleMultiplier, optional
            Applies the stain factors. Default `DefaultMultiplier`. Multipliers that need the transformer are set
            with `set_smp` after creating it.
        scale_stains : bool or pandas.Series of bool, default False
            Fit a factor per stain: True/False for all stains, or a Series with index the stains. Only scaled stains
            enter the loss; the others keep factor 1. With no scaled stain the model cannot be fitted.
        key_scale_stains : str, default "batch"
            Batch key of the factors (one factor per batch of this key and stain).
        arcsinh_cofactor : float, default 5000
            Cofactor (raw intensity units) of the arcsinh transform the loss is computed on. A cofactor several
            times (about 3x) the usual one puts more weight on the high-abundance markers and less on the noise
            around zero.
        L1weight_stainscale : float, default 0
            Weight of the L1 regularisation of the log factors; 0 disables it.
        max_abslogstain : float, optional
            Limit of the absolute log factor: the output is clamped, straight-through (the gradient passes and the
            stored parameter can exceed the limit, see `ScaleFactor`). The limit applies to the mean-normalised
            factors, before they are expressed relative to the anchor. Default: no limit.
        anchor_bidx : dict of str to int or list of int, default {}
            Anchor batch, e.g. ``{"batch": 0}`` or ``{"batch": [0]}``: all factors are expressed relative to batch
            index 0 of `key_scale_stains`. Only a single batch can be anchored. Default: no anchor, factors with
            geometric mean 1 over batches.

        Returns
        -------
        ScalingTransformer

        Raises
        ------
        ValueError
            If `anchor_bidx` holds more than one batch index for `key_scale_stains`.

        Examples
        --------
        >>> anchor_bidx = moe.transform_dict_simple({"batchmix": "18-APR-2024"})
        >>> transformer = ScalingTransformer.from_data(moe, dataset.make_dummy_spectra(), scale_stains=True,
        ...                                            key_scale_stains="batchmix", anchor_bidx=anchor_bidx)
        >>> sfm = OverlapFitModel.from_data(moe, transformer, dataset, DDIST_separate=True).to(device)
        >>> sfm.fit(Niter=800, lr=5e-3)
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

    def export(self, include_history=False):
        """Export the transformer (panel module, encoder, scale multiplier, config, parameters) to a dict.

        Parameters
        ----------
        include_history : bool, default False
            Include the parameter history (``paramhistory``).

        Returns
        -------
        dict
            Restore with `from_exported`.
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
        """Restore a transformer from the output of `export`.

        Parameters
        ----------
        exported : dict
            Output of `export`.
        cls_smp : type, optional
            Scale multiplier class, for a custom one. Default: chosen by its saved name.

        Returns
        -------
        SpectralFitTransformer or ScalingTransformer
            The restored transformer, of the class this is called on (on the CPU).
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
        """Set the scale multiplier that applies the stain factors to the unmixed data.

        E.g. `UnmxErrorExclusionMultiplier`, to block the gradient of some stain factors for selected events.

        Parameters
        ----------
        smp : ScaleMultiplier
            The multiplier (created with this transformer).
        """
        self.smp = smp
        
    def forward(self, x, bidx):
        """Standardise the data and transform it onto the scale of the loss.

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Input data on the scale the model is fitted on (usually calibrated, linear).
        bidx : dict of str to torch.Tensor of int
            Batch indices per batch key, shape (events,) or (1,) (e.g. from `SampleOverlapDataset.sample_single`
            or `MultiOrdinalEncoder.transform_ad`).

        Returns
        -------
        torch.Tensor of shape (events, stain)
            ``arcsinh(standardise(x, bidx) / arcsinh_cofactor)``.
        """
        return self.apply_transformation( self.standardise(x, bidx) )
    
    def apply_transformation(self, x):
        """Transform standardised intensities onto the scale of the loss: ``arcsinh(x / arcsinh_cofactor)``.

        Parameters
        ----------
        x : torch.Tensor
            Standardised, unmixed intensities (linear scale).

        Returns
        -------
        torch.Tensor
            Same shape as `x`.
        """
        return torch.arcsinh(x/self.config["arcsinh_cofactor"])
    
    def standardise(self, x, bidx, evl=False):
        """Return the standardised, unmixed intensities: unmixing with the fixed spectra, then the stain factors.

        Parameters
        ----------
        x : torch.Tensor of shape (events, channel)
            Input data on the scale the model is fitted on.
        bidx : dict of str to torch.Tensor of int
            Batch indices per batch key, shape (events,) or (1,).
        evl : bool, default False
            Express the factors relative to the anchor batch, for applying the fitted model
            (`OverlapFitModel.add_ad_standardised` sets it). False uses the mean-normalised factors of the fit.

        Returns
        -------
        torch.Tensor of shape (events, stain)
            Standardised, unmixed intensities (linear scale).
        """
        xc = self.pem.get_unmixed(x, bidx).detach() # detach here, but there shouldn't be any gradients anyway
        factor = self.param_stainscale.factor
        if evl and self.anchor_idx is not None:
            factor = factor / factor[self.anchor_idx]
        xc = self.smp( xc, factor[bidx[self.config["key_scale_stains"]]] ) # just a multiplication, but optionally with some detached gradients
        return xc

    @property
    def learned_stainlogscale(self):
        """Log stain factors (natural log) relative to the anchor batch, pandas.DataFrame of shape (batch, stain).

        Index: the batch labels of `key_scale_stains`; columns: the stains. Without anchor: mean 0 over batches.
        """
        logfactor = self.param_stainscale.logfactor.detach().cpu()
        if self.anchor_idx is not None:
            logfactor = logfactor - logfactor[self.anchor_idx]
        df_logfactor = pd.DataFrame(logfactor.numpy(),
                                    index=self.moe.labels[self.config["key_scale_stains"]],
                                    columns=self.stains)
        return df_logfactor

    @property
    def learned_stainscale(self):
        """Stain factors, ``exp(learned_stainlogscale)``, pandas.DataFrame of shape (batch, stain)."""
        return np.exp(self.learned_stainlogscale)
    
    def append_history(self):
        """Append the current log stain factors to ``paramhistory`` (called once per fit step)."""
        self.paramhistory["stainlogscale"].append(self.learned_stainlogscale)
    
    def get_aux_loss(self):
        """Return the weighted L1 regularisation loss.

        Returns
        -------
        cytovanni.torch.LossContainer
            ``'l1_stainlogscale'`` and ``'loss_transformer'`` (the same value, added to the fit loss).
        """
        L1_stain = self.config["L1weight_stainscale"] * self.param_stainscale.L1
        loss = LossContainer({"l1_stainlogscale":L1_stain,
                              "loss_transformer":L1_stain,
                             })
        return loss
    
    @property
    def device(self):
        """torch.device the module is on."""
        return self.param_stainscale.device
    
    def freeze(self):
        """Make all scale factors non-trainable."""
        self.param_stainscale.freeze()
    
    def extend(self, moe, extend_inds):
        """Extend the factors to new batches; the existing factors are frozen.

        New batches start at factor 1 and are trainable; they do not enter the mean normalisation.

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Extended batch encoder (after `MultiOrdinalEncoder.extend`).
        extend_inds : dict of str to list of int
            New batch indices per batch key, the output of `MultiOrdinalEncoder.extend`.
        """
        self.moe = moe
        if self.config["key_scale_stains"] in extend_inds:
            self.param_stainscale.extend(extend_inds[self.config["key_scale_stains"]])

    def plot_logscale_history(self, legend=False):
        """Plot the log stain factors (relative to the anchor) of every batch per fit step, coloured by stain.

        Needs the parameter history (fit in this session, or saved with ``include_history=True``).

        Parameters
        ----------
        legend : bool, default False
            Show a legend of the colours.
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
        """Return the standardisation parameters of one sample's batch (batch labels from ``ad.uns``).

        The factors are relative to the anchor batch. For a `FixedPanelModule` the marker names and
        ``stain_marker_name`` are both the index of the spectra, so `label_include_stain` makes no difference.

        Parameters
        ----------
        ad : anndata.AnnData
            Sample; ``ad.uns`` needs the keys of ``moe.necessary_keys``.
        label_include_stain : bool, default False
            Name the stains by ``stain_marker_name``; otherwise by the marker names.

        Returns
        -------
        channelscale : None
            No channel factors in this transformer (same return values as
            `SpectralFitTransformer.get_standardisationparams_fromad`).
        stainscale : pandas.Series
            Stain factor per stain (index the stain names).
        spectra : pandas.DataFrame of shape (channel, stain)
            The fixed spectra (index the channels, columns the stain names).
        unmx_int_matrix : pandas.DataFrame of shape (channel, stain)
            Combined unmixing and scaling matrix: data (events, channel) @ this matrix gives the standardised,
            unmixed intensities, the same as ``standardise(..., evl=True)``.
        spectra_embedding : None
            No panel embedding (fixed spectra).
        """
        bidx = self.moe.transform_ad(ad)

        factor = self.param_stainscale.factor.detach().cpu()
        if self.anchor_idx is not None:
            factor = factor / factor[self.anchor_idx]
        stainscale = pd.Series(factor[bidx[self.config["key_scale_stains"]]].numpy()[0],
                               index=self.stain_marker_name if label_include_stain else self.pem.markers)
        spectra = pd.DataFrame(self.pem.get_spectra(bidx).detach().cpu().numpy().T, index=self.channels,
                               columns=self.stain_marker_name if label_include_stain else self.pem.markers)
        
        spectra_inv = invert_spectra(spectra.T) # spectra is (channel, stain), invert_spectra takes (stain, channel)
        unmx_int_matrix = spectra_inv * stainscale.to_numpy()[None]
        
        return None, stainscale, spectra, unmx_int_matrix, None
    
    def add_standardisation_parameters(self, adata, label_include_stain=False):
        """Store the standardisation parameters of the sample's batch in `adata` (in place).

        Writes (from `get_standardisationparams_fromad`):

        - ``adata.varm['sfm_stain_factor']``: DataFrame (var, stain), the stain factors repeated in every row.
        - ``adata.varm['sfm_spectra']``: DataFrame (var, stain), the spectra, NaN rows for other variables.
        - ``adata.varm['sfm_standardise_unmx']``: DataFrame (var, stain), unmixing and scaling matrix, 0 rows for
          other variables: ``adata.to_df(layer='calibrated') @ adata.varm['sfm_standardise_unmx']`` gives the
          standardised, unmixed intensities.

        The rows are matched to ``adata.var`` by channel name. For a transformer on already unmixed data (identity
        spectra over marker names, `SampleOverlapDataset.make_dummy_spectra`), the channels are marker names that
        are usually not in ``adata.var``, so the matrices are then all NaN / 0; use `learned_stainscale` instead.

        Parameters
        ----------
        adata : anndata.AnnData
            Sample; ``adata.uns`` needs the keys of ``moe.necessary_keys``.
        label_include_stain : bool, default False
            Stain names, see `get_standardisationparams_fromad`.
        """
        _, stainscale, spectra, unmx_int_matrix, _ = self.get_standardisationparams_fromad(adata, label_include_stain=label_include_stain)

        adata.varm["sfm_spectra"] = spectra.reindex(adata.var.index)
        adata.varm["sfm_standardise_unmx"] = unmx_int_matrix.reindex(adata.var.index, fill_value=0.)

        stainscale_ext = pd.DataFrame(np.repeat([stainscale],adata.shape[1], axis=0), columns=stainscale.index, index=adata.var.index)
        adata.varm["sfm_stain_factor"] = stainscale_ext

