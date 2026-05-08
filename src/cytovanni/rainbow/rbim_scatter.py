import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter, ParameterDict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..utils import silent_log
from ..torch import ExtendableParameter
from ..exceptions import MissingChannelException, CalibrationModuleException
from .rboe import RainbowBatchOrdinalEncoder

class RainbowScatterCalibrationModule(Module):
    """
        Calibration module on the scatter channels of rainbow (or other) beads.
        Assumes that every bead type has a fixed position for all the calibrated scatter channels
        (taken as the median of the data), batch effects are simply factors along all the channels.
        
        Cheap, so runs only on the CPU.
        
        RainbowBatchOrdinalEncoder ensures unique batch names across cytometers, so we completely ignore that key here.
        
        - better interface, inst from adatas etc.
    """
    
    MIN_MEDIAN = 1 # clip to minimal value in case laser was turned off etc.
    
    def __init__(self):
        super().__init__()
        self.config = {}
        self.has_training_data = False
        self.is_trained = False
    
    @classmethod
    def from_data(cls, rboe, adatas, scatter_calibration=None):
        """ 
            :param rboe: RainbowBatchOrdinalEncoder. Batch encoder, already setup.
            
            :param adatas: iterable. Adatas for the samples that should be used for the fitting.
            
            :param scatter_calibration: dict. Optional, if None uses the default of the cytoconfig of the first adata. Dictionary of channels on which a factor should be fitted, with the channels that it should be applied to. E.g. to fit a factor for the forward scatter and apply it to 'FSC-A' and 'FSC-H' use {'FSC-A':['FSC-A','FSC-H']}. Sanity checks for this are also run through the cytoconfig of the first adata.
        """
        if scatter_calibration is None:
            scatter_calibration = adatas[0].uns["cytoconfig"].get_default_scatter_calibration()
        else:
            scatter_calibration = adatas[0].uns["cytoconfig"].check_scatter_calibration(scatter_calibrate)
        
        channels = np.asarray(list(scatter_calibration.keys()))
        for ad in adatas:
            missing = list(channels[~np.in1d(channels, ad.var.index)])
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
        """ Export self to dict.
        """
        exported = {}
        exported["rboe"] = self.rboe.export()
        exported["config"] = self.config
        exported["is_trained"] = self.is_trained
        exported["state_dict"] = self.state_dict()
        
        return exported
    
    @classmethod
    def from_exported(cls, exported, rboe=None):
        """ Construct new from exported.
            
            :param exported: dict. Output of previous .export().
            
            :param rboe: RainbowBatchOrdinalEncoder. Optional, batch encoder, already setup.
        """
        rsim = cls()
        rsim.rboe = rboe if rboe is not None else RainbowBatchOrdinalEncoder.from_exported(exported["rboe"])
        rsim.config = exported["config"]
        rsim.is_trained = exported["is_trained"]
        
        rsim.init_parameters()
        rsim.load_state_dict(exported["state_dict"])
        
        return rsim
    
    def init_parameters(self):
        """ Initialize trainable parameters.
        """
        # position of bead type in fixed units
        self.param_logposition = ExtendableParameter((len(self.rboe.oes["rainbow_type"]),  len(self.config["channels"])),
                                                     0, 11., allow_grad=True)
        # offset of each batch from fixed units
        self.param_logshift    = ExtendableParameter((len(self.rboe.oes["rainbow_batch"]), len(self.config["channels"])),
                                                     0,  0., allow_grad=True, axis_norm=0)
    
    def freeze(self):
        """ Freeze all parameters of self so they cannot be trained any further.
        """
        self.param_logposition.freeze()
        self.param_logshift.freeze()
    
    def collect_data(self, adatas):
        """ Collect data from adatas.
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
    
    def get_loss(self, weighted=True):
        """ Get loss. Simply the mean squared error between medians in batch and prediction, weighted by sqrt(N).
        """
        prediction = self.param_logposition()[self.data_type_idx] + self.param_logshift()[self.data_batch_idx]
        diff = (prediction - self.data_logmedian).square()
        loss = ((self.data_sqrtN[:,None] if weighted else 1.) * diff).mean()
        return loss
    
    def fit(self, Niter=3000, lr=5e-3, showprogress=False, minloss=1e-8):
        """ Fit self.
            Automatically freezes all parameters afterwards.
            
            :param Niter: int. Number of iterations.
            
            :param lr: float. Learning rate.
            
            :param showprogress: bool. Whether to show progress bar for fit.
            
            :param minloss: float. Minimal loss at which to stop fitting for numerical reasons.
        """
        if not self.has_training_data:
            raise CalibrationModuleException("RainbowScatterCalibrationModule needs training data for the fit!")
        
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=.5)
        self.losshistory = []

        fshow = tqdm if showprogress else (lambda x, **kwargs: x)
        for i in fshow(range(Niter), mininterval=1):
            optimizer.zero_grad()
            loss = self.get_loss()
            loss.backward()
            optimizer.step()
            self.losshistory.append({"mse":loss.item()})
            if loss.item()<minloss:
                break
        
        self.is_trained = True
        self.freeze()
    
    def plot(self):
        """ Plot overview over training and results.
        """
        if not self.is_trained:
            raise CalibrationModuleException("RainbowScatterCalibrationModule has not been trained yet!")
        if not self.has_training_data:
            raise CalibrationModuleException("RainbowScatterCalibrationModule has no training data to plot!")
        Nc = len(self.config["channels"])
        fig, ax = plt.subplots(1,1+Nc,figsize=(6*(1+Nc),5))
        
        history = pd.DataFrame(self.losshistory)
        ax[0].plot(history["mse"], label="MSE")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Optimizer Step", size=15)
        ax[0].set_ylabel("Losses", size=15)
        ax[0].legend(prop={'size': 15})
        
        prediction = (self.param_logposition()[self.data_type_idx] + self.param_logshift()[self.data_batch_idx]).detach().exp().numpy()
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
        """ Extend module on its own.
            
            :param df: pd.DataFrame. New metadata with which to extend rboe.
            
            :param adatas: iterable. New data to continue training on.
        """
        initial_types, initial_batches = self.rboe.oes["rainbow_type"].labels, self.rboe.oes["rainbow_batch"].labels
        
        # for every new added type, ensure that there is some batch overlap so it can be properly calibrated
        new_orphan_types = []
        for tp in set(df["rainbow_type"].unique())-set(initial_types):
            batches = df.loc[df["rainbow_type"]==tp, "rainbow_batch"].unique()
            batch_overlap = set(batches) & set(initial_batches)
            # if there are overlapping batches for the new type with the old trained ones, everything works
            if len(batch_overlap)==0:
                # Otherwise check that there are already trained types in the new data that have some batch overlap with the new types
                otheroverlap = (np.in1d(df["rainbow_type"], initial_types) & np.in1d(df["rainbow_batch"], batches)).sum()
                if otheroverlap==0:
                    new_orphan_types.append(tp)
        if len(new_orphan_types)>0:
            raise CalibrationModuleException(f"Newly added rainbow types {new_orphan_types} have no overlap with previously trained on data! Final calibration for these types will not be consistent with the others, aborting.")
        
        extend_inds = self.rboe.extend(df)
        self.param_logposition.extend(extend_inds["rainbow_type"])
        self.param_logshift.extend(extend_inds["rainbow_batch"])
        self.collect_data(adatas)
        
        self.fit()
    
    def extend(self, rboe, extend_inds, adatas):
        """ Extend module, assuming the checks, extending rboe etc. are done somewhere else.
            
            :param rboe: RainbowBatchOrdinalEncoder. Extended batch encoder.
            
            :param extend_inds: dict. Extended indices dict, outout of rboe.extend.
            
            :param adatas: iterable. New adatas to continue training on.
        """
        self.rboe = rboe
        self.param_logposition.extend(extend_inds["rainbow_type"])
        self.param_logshift.extend(extend_inds["rainbow_batch"])
        self.collect_data(adatas)
        
        self.fit()
    
    def get_calibration_factor(self, rainbow_batch=None, adata=None, fct_hash_settings=None):
        """ Get calibration scale factor from rainbow_batch into common units.
            Calibrating from batch into common is done by multiplying with factor.

            :param rainbow_batch: str. Optional, batch for which to get the calibration factor.

            :param adata: AnnData. Optional, AnnData from which to get the batch for the calibration factor.
        """
        batchidx = self.rboe.transform_adata_batch(adata=adata, rainbow_batch=rainbow_batch, fct_hash_settings=fct_hash_settings)
        with torch.no_grad():
            scalefactor = pd.Series((-self.param_logshift())[batchidx].cpu().exp().numpy(), index=self.config["channels"])
        scalefactor = scalefactor.loc[self.config["scatter_calibration_source"]]
        scalefactor.index = self.config["scatter_calibration_target"]
        return scalefactor
    