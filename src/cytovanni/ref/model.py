import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

from ..exceptions import OverlapStandardisationException
from ..torch import MMDLoss, GeomSampleLoss, LossContainer
from ..utils import apply_arcsinh
from .moe import MultiOrdinalEncoder
from .model_transformer import pick_transformer_cls
from .model_weight import pick_weight_cls, DummyWeights

def _sum_loss_separate_markers(fct, x, y, wx, wy):
    x_ = x[...,None].swapaxes(0,1).contiguous()
    y_ = y[...,None].swapaxes(0,1).contiguous()
    wx_ = wx.expand((x.shape[-1], ) + wx.shape).contiguous() if wx is not None else None
    wy_ = wy.expand((y.shape[-1], ) + wy.shape).contiguous() if wy is not None else None
    return fct(x_, y_, wx_, wy_).sum(0)

class OverlapFitModel(Module):
    """ Module that does the overlap fit standardisation.
        Automatically runs on the device that dataset sends its data to.
        
    """
    def __init__(self, moe, transformer, dataset=None, dataset_test=None, markerweight=None, checkpoint_DDIST=True, bandwidth_MMD=None, DDIST_mode="sinnkhorn", DDIST_kwargs={}, DDIST_separate=False, eval_only=False):
        """ 
            
            :param moe: MultiOrdinalEncoder. Batch encoder for all relevant batch keys.
            
            :param transformer: Module. The class that handles the spectral standardisation.
            
            :param dataset: None, SampleOverlapDataset. Dataset with the data to be trained on.
            
            :param dataset_test: None, SampleOverlapDataset. Optional, dataset with test data.
            
            :param markerweight: None, Module. Optionally, the class that handles the upweighting of rare celltypes etc.
            
            :param checkpoint_DDIST: bool. Whether to checkpoint DDIST loss, achieves a much lower memory footprint at marginal computational cost.
            
            :param bandwidth_MMD: None, float. Bandwidth of MMD loss, treat separately from DDIST_kwargs as this gets estimated by the model.
            
            :param DDIST_mode: str. Which loss to use, 'MMD' is my MMD implementation, 'sinkhorn' and 'hausdorff' use the respective SampleLoss (wrapped) from geomloss.
            
            :param DDIST_kwargs: dict. Additional keyword arguments for the loss.

            :param DDIST_separate: bool. Whether to apply the loss across all stains separately, or summed across each stain individually.
            
            :param eval_only: bool. If True, doesn't prepare the loss etc. for training for loaded model.
        """
        super().__init__()
        self.moe = moe
        self.transformer = transformer
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.markerweight = markerweight if markerweight is not None else DummyWeights()

        if (self.dataset is not None) and (not np.array_equal(self.dataset.channels, self.transformer.channels)):
            raise ValueError("Dataset and transformer have different channels or a different channel ordering! Fix this, otherwise results will be wrong.")
        
        self.checkpoint_DDIST = checkpoint_DDIST
        self.bandwidth_MMD = bandwidth_MMD
        self.DDIST_mode = DDIST_mode
        self.DDIST_kwargs = DDIST_kwargs
        self.DDIST_separate = DDIST_separate
        
        if not eval_only:
            if self.DDIST_mode=="MMD":
                self.ddist_fct = MMDLoss(bandwidth=self.bandwidth_MMD, **self.DDIST_kwargs)
            else:
                self.ddist_fct = GeomSampleLoss(self.DDIST_mode, **self.DDIST_kwargs)
            if self.DDIST_separate:
                self.ddist = lambda *args: _sum_loss_separate_markers(self.ddist_fct, *args)
            else:
                self.ddist = lambda *args: self.ddist_fct(*args)
        
        self.losshistory = {"train":[], "test":[], "train_DDIST_mat":[], "test_DDIST_mat":[]}
        self.plot_loss_dct = {"ddist":"DDIST"}
    
    @classmethod
    def from_data(cls, moe, transformer, dataset, dataset_test=None, markerweight=None, checkpoint_DDIST=True, DDIST_mode="sinkhorn", DDIST_kwargs={}, DDIST_separate=False):
        """ Initialize from data, also sets the MMD bandwidth from dataset if MMD is used.
            
            :param moe: MultiOrdinalEncoder. Batch encoder for all relevant batch keys.
            
            :param transformer: Module. The class that handles the spectral standardisation.
            
            :param dataset: SampleOverlapDataset. Dataset with the data to be trained on.
            
            :param dataset_test: None, SampleOverlapDataset. Optional, dataset with test data.
            
            :param markerweight: None, Module. Optionally, the class that handles the upweighting of rare celltypes etc.
            
            :param checkpoint_DDIST: bool. Whether to checkpoint DDIST loss, achieves a much lower memory footprint at marginal computationl cost.
            
            :param DDIST_mode: str. Which loss to use, 'MMD' is my MMD implementation, 'sinkhorn' and 'hausdorff' use the respective SampleLoss (wrapped) from geomloss.
            
            :param DDIST_kwargs: dict. Additional keyword arguments for the loss.

            :param DDIST_separate: bool. Whether to apply the loss across all stains separately, or summed across each stain individually.
        """
        sfm = cls(moe, transformer, dataset=dataset, dataset_test=dataset_test,
                  markerweight=markerweight, checkpoint_DDIST=checkpoint_DDIST, DDIST_mode=DDIST_mode, DDIST_kwargs=DDIST_kwargs, DDIST_separate=DDIST_separate)
        sfm.set_MMD_bandwidth_from_dataset(dataset)
        
        return sfm
    
    def export(self, include_history=True):
        """ Export self to dict.
        """
        exportdct = {}
        exportdct["moe"] = self.moe.export()
        exportdct["transformer"] = self.transformer.export(include_history=include_history)
        exportdct["transformer_name"] = self.transformer.NAME
        exportdct["markerweight"] = self.markerweight.export()
        exportdct["markerweight_name"] = self.markerweight.NAME
        exportdct["checkpoint_DDIST"] = self.checkpoint_DDIST
        exportdct["bandwidth_MMD"] = self.bandwidth_MMD
        exportdct["DDIST_mode"] = self.DDIST_mode
        exportdct["DDIST_kwargs"] = self.DDIST_kwargs
        if include_history:
            exportdct["losshistory"] = self.losshistory
        return exportdct
    
    def save(self, savepath, include_history=False):
        """ Export self to file.
        """
        exportdct = self.export(include_history=include_history)
        from pathlib import Path
        Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True)
        torch.save(exportdct, savepath)
    
    @classmethod
    def from_saved(cls, savepath, cls_transformer=None, cls_markerweight=None, cls_smp=None, eval_only=True):
        """ Initialize from saved.
            Will miss dataset and (optionally) test dataset and markerweight,
            if training should be continued they need to be set manually.
            
            :param cls_transformer: class. If using custom transformer that is not implemented by me, pass the class here to instantiate from exported properly!
            
            :param cls_markerweight: class. If using custom marker weight that is not implemented by me, pass the class here to instantiate from exported properly!
            
            :param cls_smp: class.
            
            :param eval_only: bool. If True, doesn't prepare the loss etc. for training for loaded model.
        """
        exported = torch.load(savepath, map_location=torch.device("cpu"), weights_only=False)
        moe = MultiOrdinalEncoder.from_exported(exported["moe"])
        if cls_transformer is None: cls_transformer = pick_transformer_cls(exported["transformer_name"])
        transformer = cls_transformer.from_exported(exported["transformer"], cls_smp=cls_smp)
        if cls_markerweight is None: cls_markerweight = pick_weight_cls(exported["markerweight_name"])
        markerweight = cls_markerweight.from_exported(exported["markerweight"], transformer)
        
        sfm = cls(moe, transformer, markerweight=markerweight, checkpoint_DDIST=exported["checkpoint_DDIST"], bandwidth_MMD=exported["bandwidth_MMD"],
                  DDIST_mode=exported["DDIST_mode"], DDIST_kwargs=exported["DDIST_kwargs"], eval_only=eval_only)
        if "losshistory" in exported:
            sfm.losshistory = exported["losshistory"]
        
        return sfm
    
    def estimate_MMD_bandwidth(self, dataset):
        """ Estimate MMD bandwidth on transformed data across whole dataset, use median.
            
            Not ideal, has to be estimated on untrained model etc.
            But probably still preferrable to varying bandwidth for every comparison,
            and setting by hand is much worse.
        """
        with torch.no_grad():
            datas = dataset.sample()["data"]
            bandwidths = []
            for data in datas:
                bandwidths.append(MMDLoss.estimate_bandwidth(self.transformer(data["x"], data["idx_batch"]))[None])
            torch.cuda.empty_cache()
            return torch.cat(bandwidths).cpu().median().item()
    
    def set_MMD_bandwidth_from_dataset(self, dataset):
        """ Set MMD bandwidth from dataset, overwrites old loss.
            Also sends self to same device as dataset.
            
            Only does anything if DDIST is MMD.
        """
        if self.DDIST_mode=="MMD":
            self.to(dataset.device)
            self.bandwidth_MMD = self.estimate_MMD_bandwidth(dataset)
            self.ddist = MMDLoss(bandwidth=self.bandwidth_MMD, **self.DDIST_kwargs).to(dataset.device)
        
    
    def get_DDIST_single(self, x, y, wx=None, wy=None, trainx=True, trainy=True):
        """ Get DDIST loss between transformed datas x and y.
            Can use event specific weights for either of them.
            Can choose whether to pass through gradients to either of the,.
        """
        fct_detach = lambda x, train: x if train else x.detach()
        if self.checkpoint_DDIST:
            return torch.utils.checkpoint.checkpoint(self.ddist, fct_detach(x, trainx), fct_detach(y, trainy), wx, wy, use_reentrant=False)
        else:
            return self.ddist(fct_detach(x, trainx), fct_detach(y, trainy), wx, wy)
    
    def get_DDIST_from_dataset(self, dataset):
        """ Get DDIST loss from full dataset.
            Return LossContainer the the mean loss.
            Also returns DataFrame (index, columns uid key from dataset.meta) with DDIST loss split by connection.
        """
        batch = dataset.sample()
        stain_mask = self.transformer.loss_use_stain_mask
        transformed = [self.transformer(d["x"], d["idx_batch"]) for d in batch["data"]]
        DDIST_weights = [self.markerweight(trf, d["idx_batch"], d["samp_cov"]) for trf, d in zip(transformed, batch["data"])]
        
        DDIST_full = torch.cat([self.get_DDIST_single(transformed[il][:,stain_mask], transformed[ir][:,stain_mask], DDIST_weights[il], DDIST_weights[ir], tl, tr)[None]
                                  for il, ir, tl, tr in zip(batch["cidx_left"], batch["cidx_right"], batch["dotr_left"], batch["dotr_right"])])
        DDIST = DDIST_full.mean()
        loss = LossContainer({"ddist":DDIST}, N=1)
        loss["loss"] = loss["ddist"]
        
        DDIST_mat = np.full((len(dataset), len(dataset)), np.nan)
        DDIST_mat[batch["cidx_left"], batch["cidx_right"]] = DDIST_full.detach().cpu().numpy()
        DDIST_mat = pd.DataFrame(DDIST_mat, index=dataset.meta[dataset.key_uid].tolist(), columns=dataset.meta[dataset.key_uid].tolist())
        
        return loss, DDIST_mat
    
    def get_DDIST(self, train=True):
        """ get_DDIST_from_dataset, either using the training or test dataset.
            Dummy if no test data available.
        """
        if train:
            return self.get_DDIST_from_dataset(self.dataset)
        else:
            if not self.dataset_test is None:
                with torch.no_grad():
                    return self.get_DDIST_from_dataset(self.dataset_test)
            else:
                return LossContainer({"ddist":np.nan, "loss":np.nan}, N=1), pd.DataFrame()
    
    def fit(self, Niter=500, lr=2e-2, optimizer="Adam", momentum=.8):
        """ Fit model.
            Can be safely interrupted with KeyboardInterrupt.
            
            :param Niter: int. Number of fit iterations.
            
            :param lr: float. Learning rate used by Adam.
        """
        if self.dataset is None:
            raise OverlapStandardisationException("No training data available for training! Make sure dataset is set.")
        if optimizer=="Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer=="SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        
        try:
            iterator = tqdm(range(Niter), mininterval=1, desc="Overlap Fit", postfix="LOSS={:.4f}, DDIST={:.4f}".format(0., 0.))
            for i in iterator:
                # Training step
                optimizer.zero_grad()
                self.transformer.append_history()
                loss, DDIST_mat = self.get_DDIST(train=True)
                auxloss = self.transformer.get_aux_loss()
                loss_sum = loss["loss"] + auxloss["loss_transformer"]
                loss_sum.backward()
                optimizer.step()
                loss.update(auxloss.to_numberdict())
                loss.update({"step":i})
                
                iterator.postfix = "LOSS={:.4f}, DDIST={:.4f}".format(loss["loss"].item(), loss["ddist"].item())
                
                self.losshistory["train"].append(loss.to_numberdict())
                self.losshistory["train_DDIST_mat"].append(DDIST_mat)
                
                loss, DDIST_mat = self.get_DDIST(train=False)
                loss.update({"step":i})
                self.losshistory["test"].append(loss.to_numberdict())
                self.losshistory["test_DDIST_mat"].append(DDIST_mat)
                
                torch.cuda.empty_cache()
                
        except KeyboardInterrupt:
            pass
    
    def plot_losshistory(self):
        """ Plot loss history.
        """
        fig, ax = plt.subplots(2,1,figsize=(10,8))
        losshistory_train = pd.DataFrame(self.losshistory["train"])
        
        for key, label in self.plot_loss_dct.items():
            ax[0].plot(losshistory_train["step"], losshistory_train[key], label=label)
        ax[0].set_yscale("log")
        ax[0].legend()
        ax[0].set_xlabel("Step")
        
        for key, label in self.transformer.plot_auxloss_dct.items():
            ax[1].plot(losshistory_train["step"], losshistory_train[key], label=label)
        ax[1].set_yscale("log")
        ax[1].legend()
        ax[1].set_xlabel("Step")

    def add_ad_standardised(self, ad, key_layer="calibrated", fixed_batch=None,
                          addkey="unmx_standardised", add_arcsinh=True, arcsinh_cofactor=1500,
                          batch_size=3e4):
        """ Add standardised, unmixed data to ad.

            :param ad: AnnData. Sample that should be standardised.

            :param key_layer: str. Layer in ad.layers to use for the raw intensities.

            :param fixed_batch: None, dict. If None, batch is inferred from ad.uns; otherwise should be dict containing all relevant batch keys.

            :param addkey: str. Final data is added to ad.obsm[addkey].

            :param add_arcsinh: bool. If True, also add ArcSinh transformed data to ad.obsm[addkey+'_arcsinh'].

            :param arcsinh_cofactor: float. Cofactor for ArcSinh.

            :param batch_size: int, float. To not overload the device, pass data through in batches of batch_size events. Usually no meaningful speed benefit to making this larger.
        """
        with torch.no_grad():
            x_full = torch.as_tensor(ad[:,self.transformer.channels].layers[key_layer], dtype=torch.float32)
            batches = torch.split(x_full, int(batch_size))
            ints = []
            for x in batches:
                x = x.to(self.transformer.device)
                bidx = self.moe.transform_ad(ad) if fixed_batch is None else self.moe.transform_dict(fixed_batch)
                ints.append(self.transformer.standardise(x, bidx, evl=True).cpu().numpy())
            xc_int = pd.DataFrame(np.vstack(ints), index=ad.obs.index, columns=self.transformer.stain_marker_name)
            ad.obsm[addkey] = xc_int
            if add_arcsinh:
                ad.obsm[addkey+"_arcsinh"] = apply_arcsinh(ad.obsm[addkey], cofactor=arcsinh_cofactor)


