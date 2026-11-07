"""The overlap fit: batch standardisation fitted on samples measured in several batches (`OverlapFitModel`)."""
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
    """Loss `fct` on every marker (column) of x (events, stain) and y separately, batched over markers; summed."""
    x_ = x[...,None].swapaxes(0,1).contiguous()
    y_ = y[...,None].swapaxes(0,1).contiguous()
    wx_ = wx.expand((x.shape[-1], ) + wx.shape).contiguous() if wx is not None else None
    wy_ = wy.expand((y.shape[-1], ) + wy.shape).contiguous() if wy is not None else None
    return fct(x_, y_, wx_, wy_).sum(0)

def _sum_loss_separate_markers_loop(fct, x, y, wx, wy):
    """Same as `_sum_loss_separate_markers`, looping over markers, for losses that cannot be batched (MMDLoss)."""
    return sum(fct(x[:,[i]], y[:,[i]], wx, wy) for i in range(x.shape[-1]))

class OverlapFitModel(Module):
    """Batch standardisation model fitted on samples measured in several batches (the overlap fit).

    The transformer maps the data of every batch to standardised, unmixed values (fitted spectra and/or scale
    factors per batch); this model fits the transformer's parameters such that measurements of the same sample in
    different batches look alike. All batches are fitted jointly: in each step the dataset draws a subsample of
    every sample and a set of connections (pairs of measurements of the same sample in different batches, see
    `ConnectionSampler`), and the loss is the mean distribution distance (``DDIST``, by default the Sinkhorn
    divergence) over these connections, computed on the transformer output (arcsinh scale), plus the transformer's
    regularisation. The anchor batches set the common scale (their parameters stay fixed; a `ScalingTransformer`
    instead expresses its factors relative to the anchor batch after the fit); with ``max_anchor_dist > 1``, batches
    are also linked to the anchors through other batches.

    Use `from_data` to create a model for fitting (it also sets the MMD bandwidth), `fit` to train it, `save` /
    `from_saved` to store it and `add_ad_standardised` (or the transformer's ``add_standardisation_parameters``) to
    apply it. The model does not move itself to the data's device (except `from_data` with ``DDIST_mode='MMD'``):
    call ``.to(device)`` with the dataset's device.

    Parameters
    ----------
    moe : MultiOrdinalEncoder
        Batch encoder for all batch keys of the model.
    transformer : torch.nn.Module
        Standardisation module (`SpectralFitTransformer` or `ScalingTransformer`). Its channels must equal the
        datasets' channels, in the same order.
    dataset : SampleOverlapDataset, optional
        Training data. Required for `fit`.
    dataset_test : SampleOverlapDataset, optional
        Test data; its loss is recorded in every step of `fit` (no training on it).
    markerweight : torch.nn.Module, optional
        Per-event weighting in the loss, e.g. `RareMarkerWeights`. Default: `DummyWeights` (no weighting).
    checkpoint_DDIST : bool, default True
        Recompute the loss in the backward pass (``torch.utils.checkpoint``) instead of storing intermediate
        results: much lower memory use at a small extra compute cost.
    bandwidth_MMD : float, optional
        Only for ``DDIST_mode='MMD'``: kernel bandwidth (on squared distances), kept separate from `DDIST_kwargs`
        because `from_data` estimates it. None: the MMD loss estimates it anew for every comparison.
    DDIST_mode : {'sinkhorn', 'hausdorff', 'MMD'}, default 'sinkhorn'
        Distribution distance: 'sinkhorn' and 'hausdorff' use `cytovanni.torch.GeomSampleLoss` (geomloss),
        'MMD' uses `cytovanni.torch.MMDLoss` (Gaussian kernel).
    DDIST_kwargs : dict, default {}
        Further keyword arguments of the loss, e.g. ``{"blur": .5, "reach": 1.}`` for 'sinkhorn' (these are the
        `GeomSampleLoss` defaults).
    DDIST_separate : bool, default False
        If True, compute the loss on every stain separately (1-D distributions) and sum over the stains. If False,
        compute one loss on the joint distribution over all stains used for the loss
        (``transformer.loss_use_stain_mask``).
    eval_only : bool, default False
        Skip creating the loss function (`setup_ddist`), for a loaded model that is only applied. Call `setup_ddist`
        before training or computing losses.

    Raises
    ------
    ValueError
        If the channels of a dataset differ from the transformer's channels or their order.

    Attributes
    ----------
    losshistory : dict
        Per step of `fit`: ``'train'`` and ``'test'`` (list of dicts of loss values, with ``'step'``),
        ``'train_DDIST_mat'`` and ``'test_DDIST_mat'`` (list of per-connection loss matrices, see
        `get_DDIST_from_dataset`).
    """
    def __init__(self, moe, transformer, dataset=None, dataset_test=None, markerweight=None, checkpoint_DDIST=True, bandwidth_MMD=None, DDIST_mode="sinkhorn", DDIST_kwargs={}, DDIST_separate=False, eval_only=False):
        super().__init__()
        self.moe = moe
        self.transformer = transformer
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.markerweight = markerweight if markerweight is not None else DummyWeights()

        for ds in [self.dataset, self.dataset_test]:
            self._check_dataset_channels(ds)
        
        self.checkpoint_DDIST = checkpoint_DDIST
        self.bandwidth_MMD = bandwidth_MMD
        self.DDIST_mode = DDIST_mode
        self.DDIST_kwargs = DDIST_kwargs
        self.DDIST_separate = DDIST_separate
        
        if not eval_only:
            self.setup_ddist()
        
        self.losshistory = {"train":[], "test":[], "train_DDIST_mat":[], "test_DDIST_mat":[]}
        self.plot_loss_dct = {"ddist":"DDIST"}
    
    @classmethod
    def from_data(cls, moe, transformer, dataset, dataset_test=None, markerweight=None, checkpoint_DDIST=True, DDIST_mode="sinkhorn", DDIST_kwargs={}, DDIST_separate=False):
        """Create a model for fitting; with ``DDIST_mode='MMD'`` also estimate the MMD bandwidth on `dataset`.

        For MMD, the bandwidth is estimated with the untrained transformer (`estimate_MMD_bandwidth`), and the model
        is moved to ``dataset.device``. For the other modes, move the model with ``.to(device)`` yourself.

        Parameters
        ----------
        moe : MultiOrdinalEncoder
            Batch encoder for all batch keys of the model.
        transformer : torch.nn.Module
            Standardisation module (`SpectralFitTransformer` or `ScalingTransformer`).
        dataset : SampleOverlapDataset
            Training data.
        dataset_test : SampleOverlapDataset, optional
            Test data, see the class description.
        markerweight : torch.nn.Module, optional
            Per-event weighting in the loss, e.g. `RareMarkerWeights`. Default: no weighting.
        checkpoint_DDIST : bool, default True
            Recompute the loss in the backward pass to save memory.
        DDIST_mode : {'sinkhorn', 'hausdorff', 'MMD'}, default 'sinkhorn'
            Distribution distance, see the class description.
        DDIST_kwargs : dict, default {}
            Further keyword arguments of the loss.
        DDIST_separate : bool, default False
            True: loss per stain, summed over stains; False: one loss on the joint distribution.

        Returns
        -------
        OverlapFitModel

        Examples
        --------
        >>> sfm = OverlapFitModel.from_data(moe, transformer, dataset, markerweight=markerweight,
        ...                                 DDIST_mode="sinkhorn", DDIST_kwargs={"blur": .5, "reach": 1.}).to(device)
        >>> sfm.fit(Niter=1500)
        """
        sfm = cls(moe, transformer, dataset=dataset, dataset_test=dataset_test,
                  markerweight=markerweight, checkpoint_DDIST=checkpoint_DDIST, DDIST_mode=DDIST_mode, DDIST_kwargs=DDIST_kwargs, DDIST_separate=DDIST_separate)
        sfm.set_MMD_bandwidth_from_dataset(dataset)
        
        return sfm
    
    def setup_ddist(self, device=None):
        """(Re-)create the loss function ``self.ddist`` from `DDIST_mode`, `DDIST_kwargs`, `bandwidth_MMD` and
        `DDIST_separate`.

        Parameters
        ----------
        device : str or torch.device, optional
            Move the loss module to this device.

        Raises
        ------
        NotImplementedError
            If `DDIST_mode` is not 'MMD', 'sinkhorn' or 'hausdorff'.
        """
        if self.DDIST_mode=="MMD":
            self.ddist_fct = MMDLoss(bandwidth=self.bandwidth_MMD, **self.DDIST_kwargs)
        else:
            self.ddist_fct = GeomSampleLoss(self.DDIST_mode, **self.DDIST_kwargs)
        if device is not None:
            self.ddist_fct = self.ddist_fct.to(device)
        if self.DDIST_separate:
            fct_separate = _sum_loss_separate_markers_loop if self.DDIST_mode=="MMD" else _sum_loss_separate_markers
            self.ddist = lambda *args: fct_separate(self.ddist_fct, *args)
        else:
            self.ddist = lambda *args: self.ddist_fct(*args)
    
    def export(self, include_history=True):
        """Export the model (encoder, transformer, marker weighting, loss settings) to a dict.

        The datasets are not included.

        Parameters
        ----------
        include_history : bool, default True
            Include the loss history and the transformer's parameter history.

        Returns
        -------
        dict
            Restore with `from_exported`.
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
        exportdct["DDIST_separate"] = self.DDIST_separate
        if include_history:
            exportdct["losshistory"] = self.losshistory
        return exportdct
    
    def save(self, savepath, include_history=False):
        """Save the model to a file (``torch.save`` of `export`); creates the parent folder if needed.

        Parameters
        ----------
        savepath : str
            File path, e.g. ``"model.pt"``.
        include_history : bool, default False
            Include the loss and parameter history (needed for `plot_losshistory` and the transformer's history
            plots after loading; makes the file larger).
        """
        exportdct = self.export(include_history=include_history)
        from pathlib import Path
        Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True)
        torch.save(exportdct, savepath)
    
    @classmethod
    def from_exported(cls, exported, cls_transformer=None, cls_markerweight=None, cls_smp=None, eval_only=True):
        """Restore a model from the output of `export`.

        The encoder, transformer and marker weighting are restored; the datasets are not. To continue training,
        set ``dataset`` (and ``dataset_test``) and create the loss with `setup_ddist` (or pass ``eval_only=False``).

        Parameters
        ----------
        exported : dict
            Output of `export`.
        cls_transformer : type, optional
            Transformer class, for a custom transformer. Default: chosen by its saved name (`SpectralFitTransformer`,
            `ScalingTransformer`).
        cls_markerweight : type, optional
            Marker weighting class, for a custom one. Default: chosen by its saved name.
        cls_smp : type, optional
            Scale multiplier class of the transformer, for a custom one. Default: chosen by its saved name.
        eval_only : bool, default True
            Skip creating the loss function (see the class description).

        Returns
        -------
        OverlapFitModel
            On the CPU; move it with ``.to(device)``.
        """
        moe = MultiOrdinalEncoder.from_exported(exported["moe"])
        if cls_transformer is None: cls_transformer = pick_transformer_cls(exported["transformer_name"])
        transformer = cls_transformer.from_exported(exported["transformer"], cls_smp=cls_smp)
        if cls_markerweight is None: cls_markerweight = pick_weight_cls(exported["markerweight_name"])
        markerweight = cls_markerweight.from_exported(exported["markerweight"], transformer)
        
        sfm = cls(moe, transformer, markerweight=markerweight, checkpoint_DDIST=exported["checkpoint_DDIST"], bandwidth_MMD=exported["bandwidth_MMD"],
                  DDIST_mode=exported["DDIST_mode"], DDIST_kwargs=exported["DDIST_kwargs"], DDIST_separate=exported["DDIST_separate"],
                  eval_only=eval_only)
        if "losshistory" in exported:
            sfm.losshistory = exported["losshistory"]
        
        return sfm
    
    @classmethod
    def from_saved(cls, savepath, cls_transformer=None, cls_markerweight=None, cls_smp=None, eval_only=True):
        """Load a model saved with `save`.

        The file is read with ``torch.load(weights_only=False)``, i.e. unpickled: load only files you trust.

        Parameters
        ----------
        savepath : str
            File written by `save`.
        cls_transformer, cls_markerweight, cls_smp : type, optional
            Classes for custom components, see `from_exported`.
        eval_only : bool, default True
            Skip creating the loss function, see `from_exported`.

        Returns
        -------
        OverlapFitModel
            On the CPU; move it with ``.to(device)``.

        Examples
        --------
        >>> sfm = OverlapFitModel.from_saved("model.pt").to("cuda:0")
        >>> sfm.add_ad_standardised(adata)
        """
        exported = torch.load(savepath, map_location=torch.device("cpu"), weights_only=False)
        return cls.from_exported(exported, cls_transformer=cls_transformer, cls_markerweight=cls_markerweight, cls_smp=cls_smp, eval_only=eval_only) 
    
    def estimate_MMD_bandwidth(self, dataset):
        """Estimate one MMD bandwidth for the whole dataset: the median of per-sample estimates.

        Draws one subsample of every sample (`SampleOverlapDataset.sample`), applies the current transformer and
        estimates the bandwidth (mean squared distance between events, `MMDLoss.estimate_bandwidth`) on the stains of
        the loss (``transformer.loss_use_stain_mask``). With `DDIST_separate`, per stain (median over stains and
        samples). The estimate uses the untrained model, but a fixed bandwidth is probably still preferable to one
        that varies between comparisons, and much better than setting it by hand.

        Parameters
        ----------
        dataset : SampleOverlapDataset
            Data to estimate on.

        Returns
        -------
        float
            Bandwidth (on squared distances, arcsinh scale).
        """
        with torch.no_grad():
            datas = dataset.sample()["data"]
            stain_mask = self.transformer.loss_use_stain_mask
            bandwidths = []
            for data in datas:
                x = self.transformer(data["x"], data["idx_batch"])[:,stain_mask]
                if self.DDIST_separate: # loss is applied per marker, so estimate per marker as well
                    bandwidths += [MMDLoss.estimate_bandwidth(x[:,[i]])[None] for i in range(x.shape[-1])]
                else:
                    bandwidths.append(MMDLoss.estimate_bandwidth(x)[None])
            torch.cuda.empty_cache()
            return torch.cat(bandwidths).cpu().median().item()
    
    def set_MMD_bandwidth_from_dataset(self, dataset):
        """For ``DDIST_mode='MMD'``: move the model to ``dataset.device``, estimate the bandwidth and recreate the loss.

        Does nothing for the other modes.

        Parameters
        ----------
        dataset : SampleOverlapDataset
            Data to estimate on (`estimate_MMD_bandwidth`).
        """
        if self.DDIST_mode=="MMD":
            self.to(dataset.device)
            self.bandwidth_MMD = self.estimate_MMD_bandwidth(dataset)
            self.setup_ddist(device=dataset.device)
        
    
    def get_DDIST_single(self, x, y, wx=None, wy=None, trainx=True, trainy=True):
        """Compute the distribution distance between two transformed samples.

        Parameters
        ----------
        x, y : torch.Tensor of shape (events, stain)
            Transformer outputs of the two samples (only the stains of the loss).
        wx, wy : torch.Tensor of shape (events,), optional
            Per-event weights (marker weighting). Default: uniform.
        trainx, trainy : bool, default True
            Pass gradients through `x` / `y`; False detaches that side.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        fct_detach = lambda x, train: x if train else x.detach()
        if self.checkpoint_DDIST:
            return torch.utils.checkpoint.checkpoint(self.ddist, fct_detach(x, trainx), fct_detach(y, trainy), wx, wy, use_reentrant=False)
        else:
            return self.ddist(fct_detach(x, trainx), fct_detach(y, trainy), wx, wy)
    
    def _check_dataset_channels(self, dataset):
        """Raise ValueError if `dataset` (if not None) lacks exactly the transformer's channels in the same order."""
        if (dataset is not None) and (not np.array_equal(dataset.channels, self.transformer.channels)):
            raise ValueError("Dataset and transformer have different channels or a different channel ordering! Fix this, otherwise results will be wrong.")

    def get_DDIST_from_dataset(self, dataset):
        """Compute the loss of one step on a dataset: one subsample of every sample and one set of connections.

        Parameters
        ----------
        dataset : SampleOverlapDataset
            Data to compute the loss on.

        Returns
        -------
        loss : cytovanni.torch.LossContainer
            ``'ddist'`` and ``'loss'``: mean distance over the drawn connections (with gradient).
        DDIST_mat : pandas.DataFrame of shape (sample, sample)
            Distance per connection, row = left and column = right sample of the connection (``cidx_left``,
            ``cidx_right``), NaN for pairs not drawn; index and columns ``dataset.meta[dataset.key_uid]``.

        Raises
        ------
        ValueError
            If the dataset's channels differ from the transformer's channels.
        """
        self._check_dataset_channels(dataset)
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
        """Run `get_DDIST_from_dataset` on the training or the test dataset.

        Parameters
        ----------
        train : bool, default True
            True: training dataset (with gradient). False: test dataset, without gradient; if there is none, a
            LossContainer with NaN and an empty DataFrame.

        Returns
        -------
        tuple of (cytovanni.torch.LossContainer, pandas.DataFrame)
            See `get_DDIST_from_dataset`.
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
        """Fit the transformer's parameters on the training dataset.

        Every step draws new subsamples and connections (`get_DDIST`), adds the transformer's regularisation
        (``transformer.get_aux_loss``) and takes one optimiser step; the test loss (if a test dataset is set) and
        the parameters (``transformer.append_history``) are recorded in every step. There is no convergence check:
        inspect `plot_losshistory`. Can be interrupted with KeyboardInterrupt (keeps the state of the last step).
        Calling it again continues the fit (with a new optimiser).

        Parameters
        ----------
        Niter : int, default 500
            Number of steps.
        lr : float, default 0.02
            Learning rate.
        optimizer : {'Adam', 'SGD'}, default 'Adam'
            Optimiser.
        momentum : float, default 0.8
            Momentum, only for 'SGD'.

        Raises
        ------
        OverlapStandardisationException
            If no training dataset is set.
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
        """Plot the training loss (top) and the transformer's regularisation losses (bottom) per step, log scale."""
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
                          batch_size=3e4, name_by="both"):
        """Standardise and unmix one sample and add the result to ``ad.obsm`` (in place).

        Applies ``transformer.standardise`` with the batch indices of the sample (for a `ScalingTransformer`, the
        factors relative to its anchor batch). Adds ``ad.obsm[addkey]`` (pandas.DataFrame of shape (events, stain),
        standardised unmixed intensities, linear scale) and, if `add_arcsinh`, ``ad.obsm[addkey + '_arcsinh']``
        (``arcsinh(x / arcsinh_cofactor)``).

        Parameters
        ----------
        ad : anndata.AnnData
            Sample to standardise. Needs the transformer's channels in ``ad.var`` and, without `fixed_batch`, the
            keys of ``moe.necessary_keys`` in ``ad.uns``.
        key_layer : str, default "calibrated"
            Layer with the input data, on the scale the model was fitted on (usually rainbow-calibrated intensities).
        fixed_batch : dict, optional
            Batch labels to use instead of those in ``ad.uns``: one label per key of ``moe.necessary_keys``.
        addkey : str, default "unmx_standardised"
            Key of the result in ``ad.obsm``.
        add_arcsinh : bool, default True
            Also add the arcsinh-transformed result.
        arcsinh_cofactor : float, default 1500
            Cofactor (raw intensity units) of that transform (not the one used in the fit).
        batch_size : int or float, default 30000
            Events per chunk passed to the device at once; larger chunks are usually not faster.
        name_by : {'both', 'marker', 'stain'}, default 'both'
            Column names: ``transformer.stain_marker_name``, the markers or the stains (dyes).

        Raises
        ------
        OverlapStandardisationException
            If `name_by` is not one of the options.
        """
        if not name_by in ['marker', 'stain', 'both']:
            raise OverlapStandardisationException(f"Cannot name by {name_by}, available options are 'marker', 'stain', 'both'")
        if name_by=="marker":  columns = self.transformer.pem.markers
        elif name_by=="stain": columns = self.transformer.stains
        else:                  columns = self.transformer.stain_marker_name
        with torch.no_grad():
            x_full = torch.as_tensor(ad[:,self.transformer.channels].layers[key_layer], dtype=torch.float32)
            batches = torch.split(x_full, int(batch_size))
            ints = []
            for x in batches:
                x = x.to(self.transformer.device)
                bidx = self.moe.transform_ad(ad) if fixed_batch is None else self.moe.transform_dict(fixed_batch)
                ints.append(self.transformer.standardise(x, bidx, evl=True).cpu().numpy())
            xc_int = pd.DataFrame(np.vstack(ints), index=ad.obs.index, columns=columns)
            ad.obsm[addkey] = xc_int
            if add_arcsinh:
                ad.obsm[addkey+"_arcsinh"] = apply_arcsinh(ad.obsm[addkey], cofactor=arcsinh_cofactor)

    def incorporate_stain_scaling(self, sfm_other, cls_transformer=None, cls_markerweight=None, cls_smp=None):
        """Return a copy of this model with the stain scaling of another model added on top.

        Adds ``sfm_other.transformer.learned_stainlogscale`` (log factor per batch and stain) to the fixed part of
        the copy's stain log scale, so the copy applies both scalings. A
        copy (via `export` / `from_exported`, ``eval_only=True``) is returned so that the scaling cannot be added
        twice by accident. If the transformer clips its stain scaling (`max_abslogstain`), the combined scaling is
        clipped as well, with an OverlapStandardisationWarning.

        Parameters
        ----------
        sfm_other : OverlapFitModel
            Model whose transformer's stain scaling is added. Its stain-scaling batch key must have the same batch
            labels, and its stain names must all be in this model's ``transformer.stains`` (dye names for a panel
            embedding).
        cls_transformer, cls_markerweight, cls_smp : type, optional
            Classes for custom components, see `from_exported`.

        Returns
        -------
        OverlapFitModel
            The combined model (on the CPU).

        Raises
        ------
        OverlapStandardisationException
            If the batches differ or `sfm_other` has stains that this model does not have.
        """
        self_new = type(self).from_exported(self.export(), cls_transformer=cls_transformer, cls_markerweight=cls_markerweight, cls_smp=cls_smp)
        self_new.transformer.add_stain_logscale_toself(sfm_other.transformer.learned_stainlogscale)
        return self_new


