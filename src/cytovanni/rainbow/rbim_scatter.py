"""Scatter part of the rainbow calibration: per-batch factors of the scatter channels from bead medians."""
import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter, ParameterDict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import silent_log
from ..torch import ExtendableParameter
from ..exceptions import MissingChannelException, CalibrationModuleException
from .rboe import RainbowBatchOrdinalEncoder

class RainbowScatterCalibrationModule(Module):
    """Calibration of the scatter channels with rainbow (or other) beads: one factor per batch and fitted channel.

    Model: the log median of a fitted scatter channel in a measurement is ``position[type] + logshift[batch]``, with
    one position per bead type and one log shift per batch (``logshift`` constrained to mean 0 per channel over the
    batches of the first fit). The calibration factor of a batch is ``exp(-logshift)``; the factor fitted on one
    channel can be applied to several (e.g. FSC-A and FSC-H). The fit is a weighted least squares problem, solved
    exactly in `fit` (on the CPU). The cytometer is not modelled: `RainbowBatchOrdinalEncoder` ensures that batch
    names are unique across cytometers.

    Construct it with `from_data` (or `from_exported`); the constructor only creates an empty module.

    Attributes
    ----------
    config : dict
        'channels' (fitted channels), 'scatter_calibration' (dict of fitted channel to the channels it is applied
        to), and the flattened 'scatter_calibration_source' / 'scatter_calibration_target'.
    rboe : RainbowBatchOrdinalEncoder
        Batch encoder (shared with the `RainbowCalibrator`).
    losshistory : list of dict
        Loss before and after the last `fit` (key 'mse').

    See Also
    --------
    RainbowCalibrator : user interface (builds and applies this module).
    """
    
    MIN_MEDIAN = 1 # clip to minimal value in case laser was turned off etc.
    
    def __init__(self):
        super().__init__()
        self.config = {}
        self.has_training_data = False
        self.is_trained = False
    
    @classmethod
    def from_data(cls, rboe, adatas, scatter_calibration=None):
        """Build the module from rainbow bead measurements and fit it.

        Parameters
        ----------
        rboe : RainbowBatchOrdinalEncoder
            Batch encoder that contains all batches and types of `adatas`.
        adatas : list of anndata.AnnData
            Scatter-gated bead measurements with ``layers['raw']`` and the 'rainbow_batch' and 'rainbow_type' of
            `rboe` in ``uns``.
        scatter_calibration : dict of str to list of str, optional
            Channels on which a factor is fitted, each with the channels it is applied to, e.g.
            ``{'FSC-A': ['FSC-A', 'FSC-H']}``. Checked with ``check_scatter_calibration`` of the first measurement's
            cytoconfig. Default: ``get_default_scatter_calibration()`` of that cytoconfig.

        Returns
        -------
        RainbowScatterCalibrationModule
            Fitted (and frozen) module.

        Raises
        ------
        MissingChannelException
            If a measurement lacks one of the fitted channels.
        CalibrationModuleException
            If a batch or type is not in `rboe`, or a median is not finite.
        """
        if scatter_calibration is None:
            scatter_calibration = adatas[0].uns["cytoconfig"].get_default_scatter_calibration()
        else:
            adatas[0].uns["cytoconfig"].check_scatter_calibration(scatter_calibration)
        
        channels = np.asarray(list(scatter_calibration.keys()))
        for ad in adatas:
            missing = list(channels[~np.isin(channels, ad.var.index)])
            if len(missing)>0:
                raise MissingChannelException(f"Channels {missing} are required to fit the rainbow scatter calibration, but not present in adata!")
        
        rsim = cls()
        rsim.rboe = rboe
        rsim.config["channels"] = channels
        rsim.config["scatter_calibration"] = scatter_calibration
        rsim.config["scatter_calibration_source"] = [c for k,v in scatter_calibration.items() for c in [k]*len(v)]
        rsim.config["scatter_calibration_target"] = [c for k,v in scatter_calibration.items() for c in v]
        
        rsim.init_parameters()
        rsim.collect_data(adatas)
        rsim.fit()
        
        return rsim
    
    def export(self):
        """Export the module (config, `is_trained`, state dict, encoder; no data) as a dict for `from_exported`."""
        exported = {}
        exported["rboe"] = self.rboe.export()
        exported["config"] = self.config
        exported["is_trained"] = self.is_trained
        exported["state_dict"] = self.state_dict()
        
        return exported
    
    @classmethod
    def from_exported(cls, exported, rboe=None):
        """Rebuild a module from the output of `export` (without training data).

        Parameters
        ----------
        exported : dict
            Output of `export`.
        rboe : RainbowBatchOrdinalEncoder, optional
            Batch encoder to use (shared with the other modules of a `RainbowCalibrator`). Default: rebuilt from
            ``exported['rboe']``.

        Returns
        -------
        RainbowScatterCalibrationModule
        """
        rsim = cls()
        rsim.rboe = rboe if rboe is not None else RainbowBatchOrdinalEncoder.from_exported(exported["rboe"])
        rsim.config = exported["config"]
        rsim.is_trained = exported["is_trained"]
        
        rsim.init_parameters()
        rsim.load_state_dict(exported["state_dict"])
        
        return rsim
    
    def init_parameters(self):
        """Create the parameters: log position per (type, channel), init 11; log shift per (batch, channel), init 0."""
        # position of bead type in fixed units
        self.param_logposition = ExtendableParameter((len(self.rboe.oes["rainbow_type"]),  len(self.config["channels"])),
                                                     0, 11., allow_grad=True)
        # offset of each batch from fixed units
        self.param_logshift    = ExtendableParameter((len(self.rboe.oes["rainbow_batch"]), len(self.config["channels"])),
                                                     0,  0., allow_grad=True, axis_norm=0)
    
    def freeze(self):
        """Freeze all parameters, so that later fits (after `extend`) only change new entries."""
        self.param_logposition.freeze()
        self.param_logshift.freeze()
    
    def collect_data(self, adatas):
        """Collect the training data: encoded batch and type, log median per fitted channel, sqrt of the event number.

        The log median per measurement and channel is the median of ``ln(layers['raw'])`` over the events (negative
        values give NaN and are ignored, zeros count as -inf), clipped below at ``ln(MIN_MEDIAN)`` = 0, e.g. for a
        laser that was off.

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Bead measurements (replace any previous training data).

        Raises
        ------
        CalibrationModuleException
            If a batch or type is not in `rboe`.
        """
        metadata_df = pd.DataFrame([self.rboe.transform_adata(ad) for ad in adatas])
        self.data_batch_idx = torch.tensor(metadata_df["rainbow_batch"].to_numpy()).long()
        self.data_type_idx  = torch.tensor(metadata_df["rainbow_type"].to_numpy()).long()
        
        if (self.data_batch_idx==-1).sum()>0:
            raise CalibrationModuleException("Some of the passed adatas have a 'rainbow_batch' that was not set up in the ordinal encoder!")
        if (self.data_type_idx==-1).sum()>0:
            raise CalibrationModuleException("Some of the passed adatas have a 'rainbow_type' that was not set up in the ordinal encoder!")
        # warn if not all in data!! or just use adatas for the whole setup

        get_med = lambda ad, key: np.clip(np.nanmedian(silent_log(ad[:,key].layers["raw"][:,0])), a_min=np.log(self.MIN_MEDIAN), a_max=None)
        self.data_logmedian = torch.tensor([[get_med(ad, key) for key in self.config["channels"]] for ad in adatas], dtype=torch.float32)
        self.data_sqrtN = torch.sqrt(torch.tensor([ad.shape[0] for ad in adatas], dtype=torch.float32))
        
        self.has_training_data = True
    
    def get_prediction(self):
        """Predicted log median per training measurement and fitted channel, shape (measurement, channel)."""
        return self.param_logposition()[self.data_type_idx] + self.param_logshift()[self.data_batch_idx]
    
    def get_loss(self, weighted=True):
        """Mean squared error between the predicted and measured log medians (natural log, clipped medians).

        Parameters
        ----------
        weighted : bool, default True
            Weight the squared error of every measurement by the square root of its number of events.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        diff = (self.get_prediction() - self.data_logmedian).square()
        loss = ((self.data_sqrtN[:,None] if weighted else 1.) * diff).mean()
        return loss
    
    def fit(self):
        """Fit the trainable parameters exactly by minimising the weighted loss of `get_loss`, then freeze them.

        The prediction is affine in the trainable entries of both parameters (position plus batch shift, the shift
        mean-normed over ``param_mask_meannorm``, frozen entries constant), and the channels are independent, so the
        exact solution is one weighted linear least squares solve per channel. The problem is degenerate (e.g. a
        common shift of all mean-normed batch shifts); of its solutions, the one closest to the current values is
        taken (minimum-norm change), the point that gradient descent from these values converges to.

        Returns
        -------
        None
            Sets `losshistory` (loss before and after) and `is_trained`.

        Raises
        ------
        CalibrationModuleException
            If there are no training data, or the medians or the result are not finite.
        """
        if not self.has_training_data:
            raise CalibrationModuleException("RainbowScatterCalibrationModule needs training data for the fit!")
        if not torch.isfinite(self.data_logmedian).all():
            raise CalibrationModuleException("Non-finite medians in the training data of RainbowScatterCalibrationModule!")
        
        self.losshistory = [{"mse":self.get_loss().item()}]
        
        pos, shift = self.param_logposition, self.param_logshift
        tidx, bidx = self.data_type_idx.numpy(), self.data_batch_idx.numpy()
        sqrtw = self.data_sqrtN.sqrt().numpy().astype(np.float64)
        with torch.no_grad():
            residual = (self.get_prediction() - self.data_logmedian).numpy().astype(np.float64)
        mask_pos = pos.param_mask_trainable.data.numpy()!=0
        mask_shift = shift.param_mask_trainable.data.numpy()!=0
        meannorm = shift.param_mask_meannorm.data.numpy()
        delta_pos, delta_shift = np.zeros(mask_pos.shape), np.zeros(mask_shift.shape)
        # channels are independent; columns: trainable type positions, trainable (raw, before mean norm) batch shifts
        for c in range(residual.shape[1]):
            types, batches = np.where(mask_pos[:,c])[0], np.where(mask_shift[:,c])[0]
            if len(types)+len(batches)==0:
                continue
            J_pos = (tidx[:,None]==types[None]).astype(np.float64)
            J_shift = (bidx[:,None]==batches[None]) - meannorm[batches,c][None] / meannorm[:,c].sum()
            J = np.concatenate([J_pos, J_shift], axis=1) * sqrtw[:,None]
            delta = np.linalg.lstsq(J, -sqrtw * residual[:,c], rcond=None)[0]
            delta_pos[types,c], delta_shift[batches,c] = delta[:len(types)], delta[len(types):]
        if not (np.all(np.isfinite(delta_pos)) and np.all(np.isfinite(delta_shift))):
            raise CalibrationModuleException("RainbowScatterCalibrationModule fit gave non-finite parameters!")
        for param, delta in [(pos, delta_pos), (shift, delta_shift)]:
            param.param_trainable.data += torch.as_tensor(delta, dtype=torch.float32)
        self.losshistory.append({"mse":self.get_loss().item()})
        
        self.is_trained = True
        self.freeze()
    
    def plot(self):
        """Plot the loss before and after the fit, and predicted against measured medians per channel (by bead type).

        Raises
        ------
        CalibrationModuleException
            If the module is not trained or has no training data (e.g. after `from_exported`).
        """
        if not self.is_trained:
            raise CalibrationModuleException("RainbowScatterCalibrationModule has not been trained yet!")
        if not self.has_training_data:
            raise CalibrationModuleException("RainbowScatterCalibrationModule has no training data to plot!")
        Nc = len(self.config["channels"])
        fig, ax = plt.subplots(1,1+Nc,figsize=(6*(1+Nc),5))
        
        history = pd.DataFrame(self.losshistory)
        ax[0].plot(history["mse"], marker="o", label="MSE")
        ax[0].set_yscale("log")
        ax[0].set_xticks([0, 1], ["initial", "fitted"])
        ax[0].set_xlabel("Fit", size=15)
        ax[0].set_ylabel("Losses", size=15)
        ax[0].legend(prop={'size': 15})
        
        prediction = self.get_prediction().detach().exp().numpy()
        real = self.data_logmedian.exp().numpy()
        for tp in self.data_type_idx.unique():
            mask = self.data_type_idx==tp
            for i in range(Nc):
                ax[1+i].scatter(real[mask,i], prediction[mask,i], label=self.rboe.oes["rainbow_type"].labels[tp])
        for i in range(Nc):
            line = [min(real[:,i].min(), prediction[:,i].min())-.05, max(real[:,i].max(), prediction[:,i].max())+.05]
            ax[1+i].legend()
            ax[1+i].plot(line, line, color="black", zorder=-1)
            ax[1+i].set_title(self.config["channels"][i], size=15)
            ax[1+i].set_xlabel("Median Real", size=12)
            ax[1+i].set_ylabel("Median Predicted", size=12)
            ax[1+i].set_xscale("log")
            ax[1+i].set_yscale("log")
        
        plt.tight_layout()
    
    def extend_standalone(self, df, adatas):
        """Extend the encoder and the module to new measurements and fit them, without a `RainbowCalibrator`.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata of the new measurements, one row per measurement, as for `RainbowBatchOrdinalEncoder.from_data`.
        adatas : list of anndata.AnnData
            New bead measurements (the only training data of the new fit).

        Returns
        -------
        None

        Raises
        ------
        CalibrationModuleException
            If a new rainbow type has no link to the trained data (`RainbowBatchOrdinalEncoder.check_orphan_types`).
        """
        self.rboe.check_orphan_types(df)
        extend_inds = self.rboe.extend(df)
        self.param_logposition.extend(extend_inds["rainbow_type"])
        self.param_logshift.extend(extend_inds["rainbow_batch"])
        self.collect_data(adatas)
        
        self.fit()
    
    def extend(self, rboe, extend_inds, adatas):
        """Extend the module to an already extended encoder and fit the new entries (the known ones stay fixed).

        Parameters
        ----------
        rboe : RainbowBatchOrdinalEncoder
            Extended batch encoder (replaces `rboe`; checks are the caller's job).
        extend_inds : dict of str to list of int
            New ordinals per key, the output of `RainbowBatchOrdinalEncoder.extend`.
        adatas : list of anndata.AnnData
            New bead measurements (the only training data of the new fit).

        Returns
        -------
        None
        """
        self.rboe = rboe
        self.param_logposition.extend(extend_inds["rainbow_type"])
        self.param_logshift.extend(extend_inds["rainbow_batch"])
        self.collect_data(adatas)
        
        self.fit()
    
    def get_calibration_factor(self, rainbow_batch=None, adata=None, fct_hash_settings=None):
        """Scatter calibration factors of one batch: multiply its raw data by them.

        Parameters
        ----------
        rainbow_batch : str, optional
            Batch name (preferred if given).
        adata : anndata.AnnData, optional
            Sample whose batch is used, see `RainbowBatchOrdinalEncoder.transform_adata_batch`.
        fct_hash_settings : callable, optional
            Settings hash function for the batch inference. Default: ``adata.uns['cytoconfig'].fct_hash_settings``.

        Returns
        -------
        pandas.Series
            Factor per target channel (``config['scatter_calibration_target']``, e.g. FSC-A and FSC-H with the factor
            fitted on FSC-A).
        """
        batchidx = self.rboe.transform_adata_batch(adata=adata, rainbow_batch=rainbow_batch, fct_hash_settings=fct_hash_settings)
        with torch.no_grad():
            scalefactor = pd.Series((-self.param_logshift())[batchidx].cpu().exp().numpy(), index=self.config["channels"])
        scalefactor = scalefactor.loc[self.config["scatter_calibration_source"]]
        scalefactor.index = self.config["scatter_calibration_target"]
        return scalefactor
    