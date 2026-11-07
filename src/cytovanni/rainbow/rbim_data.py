"""Training data of the rainbow fluorescence model, and the subsampler also used by the Sinkhorn factor fits."""
import numpy as np
import pandas as pd
import torch
import warnings
import itertools

from ..exceptions import CalibrationModuleException, CalibrationModuleWarning

class DataLoader():
    """Iterate over one subsample of one rainbow type of a `RainbowFluorescenceGMMCalibrationDataset` in batches.

    Parameters
    ----------
    dataset : RainbowFluorescenceGMMCalibrationDataset
        Dataset to read from.
    tp : str
        Rainbow type.
    sample_indices : torch.Tensor
        Event indices (into the type's data) of the subsample, shape (events,).
    batch_size : int
        Events per batch.

    Notes
    -----
    Iterating yields the dicts of `RainbowFluorescenceGMMCalibrationDataset.get_batch`. ``len()`` is the number of
    events, not of batches (used to weight the batches).
    """
    def __init__(self, dataset, tp, sample_indices, batch_size):
        self.dataset = dataset
        self.tp = tp
        self.sample_indices = sample_indices
        self.batch_size = batch_size
        
        self.sample_indices_split = torch.split(self.sample_indices, self.batch_size)
    
    def __iter__(self):
        for indices in self.sample_indices_split:
            yield self.dataset.get_batch(self.tp, indices)
    
    def __len__(self):
        return len(self.sample_indices)

def _intype_GMM_crosstab_shares(index, label, N_sample, N_peak):
    """Share of every peak label per sample, shape (N_sample, N_peak), like a row-normalised crosstab but faster.

    `index` is the sample index (in range(N_sample)) and `label` the peak label (in range(N_peak)) of every bead;
    samples without beads get uniform shares 1/N_peak.
    """
    count = torch.zeros((N_sample, N_peak), dtype=torch.float32)
    for lb in torch.unique(label):
        u, c = torch.unique(index[label==lb], return_counts=True)
        count[u,lb] = c.to(torch.float32)
    total = count.sum(-1, keepdims=True)
    return torch.where(total>0, count / total.clamp(min=1.), torch.full_like(count, 1./N_peak))

class Sampler():
    """Draw random subsamples of at most `Nsubsample` indices from each of several groups.

    Parameters
    ----------
    indices_list : list of torch.Tensor
        One 1-D int64 tensor of indices per group (e.g. the events of one measurement or batch). Groups may be empty.
    Nsubsample : int
        Maximum number of indices drawn per group in every call of `sample`; groups with fewer contribute all of
        theirs (in random order).
    device : str, default "cpu"
        Torch device for the permutations. On a CUDA device, one large permutation is used for all groups, which is
        much faster than one permutation per group.

    Notes
    -----
    Every call of `sample` draws a new subsample without replacement within each group.
    """
    def __init__(self, indices_list, Nsubsample, device="cpu"):
        self.indices_list = indices_list
        self.Nsubsample = Nsubsample
        self.N = len(indices_list)
        self.device = device
        
        self.on_gpu = torch.tensor(0).to(device).is_cuda
        
        # index which group it is from
        self.indices_group = torch.cat([torch.tensor(ind, dtype=torch.int64).repeat(self.indices_list[ind].shape[0]) for ind in range(self.N)])
        
        if self.on_gpu:
            self.total = sum([x.shape[0] for x in self.indices_list])
            
            # Flattened available indices
            self.indices = torch.cat(self.indices_list).to(self.device)
            # scaled by total number of indices
            self.indices_group_add = self.total * self.indices_group.to(self.device)
            #
            cumsum = [0] + list(np.cumsum([x.shape[0] for x in self.indices_list]))[:-1]
            self.indices_choose = torch.cat([torch.arange(inds.shape[0])[:self.Nsubsample] + cumsum[i]
                                             for i, inds in enumerate(self.indices_list)]).to(self.device)
    
    def permutation_cuda(self):
        """Subsample with one large permutation on the GPU (returns a CPU tensor).

        Ranks of a random permutation, offset by group so that every group keeps its own range, argsorted again: a
        random order within every group. The first `Nsubsample` of every group are returned, groups in order.
        """
        full_permutation = torch.randperm(self.total, device=self.device, dtype=torch.int64)
        argsort_permutation = torch.argsort(full_permutation) + self.indices_group_add
        permutation = torch.argsort(argsort_permutation)
        subset_permutation = permutation[self.indices_choose]
        subsample_indices = self.indices[subset_permutation].cpu()
        return subsample_indices
    
    def permutation_cpu(self):
        """Subsample with one permutation per group (CPU path); groups in order."""
        get_permutation = lambda inds: inds[torch.randperm(len(inds), device=self.device)[:self.Nsubsample].cpu()]
        return torch.cat([get_permutation(inds) for inds in self.indices_list])
    
    def sample(self, split=False):
        """Draw a new subsample.

        Parameters
        ----------
        split : bool, default False
            Return the subsample split by group.

        Returns
        -------
        torch.Tensor or tuple
            If not `split`: the drawn indices of all groups concatenated (groups in order, ``min(len(group),
            Nsubsample)`` each), on the CPU for CPU `indices_list`. If `split`: a tuple ``(indices, groups)``, with a
            tuple of one index tensor per non-empty group and a tensor of the numbers of these groups (empty groups
            are left out).
        """
        if self.on_gpu:
            inds = self.permutation_cuda()
        else:
            inds = self.permutation_cpu()
        if not split:
            return inds
        
        # both permutations return the groups in order, min(Nsubsample, group size) indices each;
        # empty groups are left out, as are their group numbers
        counts = [min(len(x), self.Nsubsample) for x in self.indices_list]
        splitinds = torch.split(inds, counts)
        nonempty = [g for g, c in enumerate(counts) if c>0]
        return tuple(splitinds[g] for g in nonempty), torch.tensor(nonempty, dtype=torch.int64)

class RainbowFluorescenceGMMCalibrationDataset():
    """Training data of `RainbowFluorescenceGMMCalibrationModule`: rainbow bead events per type, with their indices.

    Collects the raw fluorescence data (``layers['raw']``, channels ``cytoconfig.channels_fluorescence`` of the first
    measurement), the GMM labels and peak means of `fit_rainbow_GMM`, and the encoded batch, type and uid of every
    bead, grouped by rainbow type. Marks which peaks are usable for the calibration, and provides random subsamples of
    at most `Nsubsample` beads per measurement for every training step.

    Parameters
    ----------
    rboe : RainbowBatchOrdinalEncoder
        Batch encoder that already contains all batches, types and uids of `adatas`.
    adatas : list of anndata.AnnData
        Rainbow bead measurements after `fit_rainbow_GMM` (see `RainbowCalibrator`).
    device : str, default "cpu"
        Torch device of the data and the subsampling, e.g. 'cpu' or 'cuda:0'.
    Nsubsample : int, default 1000
        Beads drawn per measurement (in total over its peaks) in every training step; measurements with fewer usable
        beads contribute all of them.
    batch_size : int, default 20000
        Events per batch of the loaders of `sample_loaders` (memory limit only; the gradients are accumulated over
        all batches of a step).
    data_on_device : bool, default True
        Move the data to `device` at once (rainbow data are small).
    channel_missing_cutoff : float, default -inf
        Channels in which every peak of a measurement has a GMM mean below this value (raw units) are treated like
        off-scale peaks in that measurement (e.g. a laser switched off). The default uses all channels.
    warn_missing : bool, default True
        Warn if some channels cannot be trained in some batches (`df_channel_train` is set in any case).

    Attributes
    ----------
    present_types : numpy.ndarray of str
        Rainbow types in `adatas`.
    data : dict of str to torch.Tensor
        Raw fluorescence data per type, shape (events, channel).
    GMM_means : dict of str to numpy.ndarray
        ``uns['GMM_means']`` per type, shape (measurement, peak, channel), raw scale.
    peak_usemask : dict of str to torch.Tensor
        Boolean, shape (measurement, peak, channel): peak usable for the calibration (``GMM_means`` below
        ``cytoconfig.max_linear_range`` of the measurement, and the channel not missing, see
        `channel_missing_cutoff`). The other peaks are the "nonlinear" peaks of the model.
    peak_shares : dict of str to torch.Tensor
        Share of every GMM peak per measurement, shape (measurement, peak), from the beads that are not off-scale in
        all channels.
    df_channel_train : pandas.DataFrame
        Boolean, shape (batch, channel), index the names of the batches in `adatas`: True if the channel has a usable
        peak in some measurement of the batch, i.e. its factor can be trained.

    Raises
    ------
    CalibrationModuleException
        If some measurements were not fitted with `fit_rainbow_GMM`, the number of peaks differs within a type, or a
        batch, type or uid is not in `rboe`.

    Notes
    -----
    Beads whose GMM peak is unusable in every channel are left out of the subsamples.
    """
    def __init__(self, rboe, adatas, device="cpu", Nsubsample=1000, batch_size=20000, data_on_device=True, channel_missing_cutoff=-np.inf, warn_missing=True):
        self.cytoconfig = adatas[0].uns["cytoconfig"] # just use the first one, channels have to be the same everywhere anyway
        self.channels = self.cytoconfig.channels_fluorescence
        self.rboe = rboe
        self.adatas = adatas
        
        self.Nsubsample = Nsubsample
        self.device = device
        self.batch_size = batch_size
        self.data_on_device = data_on_device
        self.channel_missing_cutoff = channel_missing_cutoff
        self.warn_missing = warn_missing
        
        self.check_GMM_setup(adatas)
        self.check_type_Npeak_consistency(adatas)
        self.setup_data()
        if self.data_on_device:
            self.shift_data_to_device()
    
    @classmethod
    def check_GMM_setup(self, adatas):
        """Raise CalibrationModuleException if some adatas lack ``obs['GMM_label']`` or ``uns['GMM_means']``.

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Rainbow bead measurements.
        """
        if np.any([("GMM_label" not in ad.obs) or ("GMM_means" not in ad.uns) for ad in adatas]):
            raise CalibrationModuleException("Some of the adatas were not fitted with the simple GMM model!")
    
    @classmethod
    def check_type_Npeak_consistency(cls, adatas, prior={}):
        """Raise CalibrationModuleException if the number of GMM peaks differs within a rainbow type.

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Rainbow bead measurements after `fit_rainbow_GMM`.
        prior : dict of str to int, default {}
            Known number of peaks per type; a type in `prior` must also match it.
        """
        for tp in np.unique([ad.uns["rainbow_type"] for ad in adatas]):
            uN = np.unique([ad.uns["GMM_means"].shape[0] for ad in adatas if ad.uns["rainbow_type"]==tp])
            if len(uN)>1 or (uN[0]!=prior[tp] if tp in prior else False):
                raise CalibrationModuleException(f"Number of rainbow peaks not consistent for rainbow type {tp}!")
    
    @classmethod
    def get_type_Npeak(cls, adatas):
        """Number of GMM peaks per rainbow type.

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Rainbow bead measurements after `fit_rainbow_GMM`.

        Returns
        -------
        dict of str to int
            Number of peaks (rows of ``uns['GMM_means']``) per type.

        Raises
        ------
        CalibrationModuleException
            If the number differs within a type.
        """
        cls.check_type_Npeak_consistency(adatas)
        return {tp: np.unique([ad.uns["GMM_means"].shape[0]
                               for ad in adatas if ad.uns["rainbow_type"]==tp])[0]
                for tp in np.unique([ad.uns["rainbow_type"] for ad in adatas])}
    
    def check_batch_training_missing(self, warn=True):
        """Find the (batch, channel) factors that cannot be trained and store them in `df_channel_train`.

        A factor cannot be trained if no measurement of the batch has a usable peak in the channel (all peaks above the
        linear range, or the channel missing, see `channel_missing_cutoff`). The trainer needs `df_channel_train`.

        Parameters
        ----------
        warn : bool, default True
            Issue a `CalibrationModuleWarning` (with counts) if some channel cannot be trained in some batch.

        Returns
        -------
        None
        """
        batch_idx = {k:self.peak_shares_batchidx[k].cpu().numpy() for k in self.present_types}
        train_mask = {k:np.any(self.peak_usemask[k].cpu().numpy(), axis=1) for k in self.present_types}

        unique_batchidx = np.unique(np.concatenate(list(batch_idx.values())))
        df_train = pd.DataFrame(False, index=unique_batchidx, columns=self.channels)
        for k in self.present_types:
            for i, bidx in enumerate(batch_idx[k]): # loop in case batches are present more than once
                df_train.loc[bidx] = df_train.loc[bidx] | train_mask[k][i]
        df_train.index = self.rboe.oes["rainbow_batch"].labels[df_train.index]

        self.df_channel_train = df_train

        if warn and (~self.df_channel_train.to_numpy()).sum()>0:
            warnstr = "For some batches, some channels will not get trained! Make sure this is intentional, and not caused by wrong settings for 'max_linear_range' or 'channel_missing_cutoff'."
            n_channel, Nc = ((~self.df_channel_train.to_numpy()).sum(0)>0).sum(), self.df_channel_train.shape[1]
            n_batch, Nb = ((~self.df_channel_train.to_numpy()).sum(1)>0).sum(), self.df_channel_train.shape[0]
            warnstr+= f"\n{n_channel} (out of {Nc}) channels are not trainable in some batches, {n_batch} (out of {Nb}) batches have some not trainable channels."
            warnstr+= f"\nA full accounting of all untrained channels can be found at 'dataset.df_channel_train'."
            warnings.warn(warnstr, CalibrationModuleWarning)
    
    def setup_data(self):
        """Collect data, indices, usable-peak masks, initial peak shares and the subsamplers from `adatas` (by type).

        Called by the constructor; also sets `df_channel_train` (see `check_batch_training_missing`).

        Raises
        ------
        CalibrationModuleException
            If a batch, type or uid of `adatas` is not in `rboe`.
        """
        self.present_types = np.unique([ad.uns["rainbow_type"] for ad in self.adatas])
        self.adatas_tp = {tp:[ad for ad in self.adatas if ad.uns["rainbow_type"]==tp] for tp in self.present_types}
        
        # intensity data and GMM index
        self.data = {tp: torch.cat([torch.as_tensor(ad[:,self.channels].layers["raw"], dtype=torch.float32) for ad in adatas])
                     for tp, adatas in self.adatas_tp.items()}
        self.data_GMM_index = {tp: torch.tensor(np.concatenate([ad.obs["GMM_label"] for ad in adatas]))
                               for tp, adatas in self.adatas_tp.items()}
        
        # Index of batch, type etc. for every event
        fct_single_index = lambda ad, key: torch.tensor(self.rboe.oes[key].transform(ad.uns[key]), dtype=torch.long).repeat(ad.shape[0])
        
        self.data_batch_index = {tp: torch.cat([fct_single_index(ad, "rainbow_batch") for ad in adatas])
                                 for tp, adatas in self.adatas_tp.items()}
        if sum([(v==-1).sum() for v in self.data_batch_index.values()]).item()>0:
            raise CalibrationModuleException("Some of the rainbow batches in adatas were not set up in the ordinal encoder!")
        
        self.data_type_index = {tp: torch.cat([fct_single_index(ad, "rainbow_type") for ad in adatas])
                                 for tp, adatas in self.adatas_tp.items()}
        if sum([(v==-1).sum() for v in self.data_type_index.values()]).item()>0:
            raise CalibrationModuleException("Some of the rainbow types in adatas were not set up in the ordinal encoder!")
        
        self.data_uid_index = {tp: torch.cat([fct_single_index(ad, "uid") for ad in adatas])
                                 for tp, adatas in self.adatas_tp.items()}
        if sum([(v==-1).sum() for v in self.data_uid_index.values()]).item()>0:
            raise CalibrationModuleException("Some of the unique identifiers in adatas were not set up in the ordinal encoder!")
        
        # GMM result, mask non-linear ones
        def get_type_means(tp):
            return np.asarray([ad.uns["GMM_means"] for ad in self.adatas if ad.uns["rainbow_type"]==tp])
        self.GMM_means = {tp:get_type_means(tp) for tp in self.present_types}
        
        def get_type_usemask(tp):
            maxlin = np.asarray([ad.uns["cytoconfig"].max_linear_range for ad in self.adatas_tp[tp]])
            return torch.tensor(self.GMM_means[tp] < maxlin[:,None,None])
        def get_type_channelnotmissingmask(tp):
            return torch.tensor(~np.all(self.GMM_means[tp] < self.channel_missing_cutoff, axis=1, keepdims=True))
        self.peak_usemask = {tp:get_type_usemask(tp) & get_type_channelnotmissingmask(tp) for tp in self.present_types}
        
        self.data_intype_index = {tp: torch.tensor(np.concatenate([[i]*ad.shape[0] for i, ad in enumerate(adatas)]))
                               for tp, adatas in self.adatas_tp.items()}
        
        # filter out beads where every single peak is off-scale
        self.data_usemask = {tp: torch.any(self.peak_usemask[tp][self.data_intype_index[tp], self.data_GMM_index[tp]], dim=-1)
                               for tp, adatas in self.adatas_tp.items()}
        
        # Stuff to initialise means in the module
        self.peak_shares = {tp: _intype_GMM_crosstab_shares(self.data_intype_index[tp][self.data_usemask[tp]],
                                                            self.data_GMM_index[tp][self.data_usemask[tp]],
                                                            len(self.adatas_tp[tp]), self.peak_usemask[tp].shape[1]) for tp in self.present_types}
        self.peak_shares_uididx = {tp: torch.tensor(self.rboe.oes["uid"].transform([ad.uns["uid"] for ad in adatas]), dtype=torch.long)
                                   for tp, adatas in self.adatas_tp.items()}
        self.peak_shares_batchidx = {tp: torch.tensor(self.rboe.oes["rainbow_batch"].transform([ad.uns["rainbow_batch"] for ad in adatas]), dtype=torch.long)
                                     for tp, adatas in self.adatas_tp.items()}
        
        # batched indices
        def fct_inds_tp(self, tp):
            fullind = torch.arange(self.data_GMM_index[tp].shape[0])
            uids = torch.tensor(self.rboe.oes["uid"].transform([ad.uns["uid"] for ad in self.adatas_tp[tp]]))
            return [fullind[(self.data_usemask[tp]) & (self.data_uid_index[tp]==uid)] for uid in uids]
        self.data_indices_batched = {tp: fct_inds_tp(self, tp) for tp in self.present_types}
        
        # subsample class
        self.batch_index_sampler = {tp:Sampler(self.data_indices_batched[tp], self.Nsubsample, self.device) for tp in self.data_indices_batched}
        
        # not trainable channels per batch (df_channel_train, needed by the trainer); warn only if warn_missing
        self.check_batch_training_missing(warn=self.warn_missing)
    
    def shift_data_to_device(self):
        """Move the event data, index tensors and usable-peak masks to `device`."""
        keys_to_device = ["data", "data_uid_index", "data_batch_index", "data_intype_index", "peak_usemask"]
        for key in keys_to_device:
            setattr(self, key, {tp:getattr(self, key)[tp].to(self.device) for tp in getattr(self, key)})

    def get_sample_indices(self):
        """Draw a new subsample of at most `Nsubsample` usable beads per measurement.

        Returns
        -------
        dict of str to torch.Tensor
            Per rainbow type the event indices of the subsample (CPU), shape (events,).
        """
        return {tp:self.batch_index_sampler[tp].sample() for tp in self.batch_index_sampler}
    
    def get_batch(self, tp, sample_indices):
        """Data and indices of the given events of one rainbow type, on `device`.

        Parameters
        ----------
        tp : str
            Rainbow type.
        sample_indices : torch.Tensor
            Event indices into the type's data, shape (events,).

        Returns
        -------
        dict
            'data' (raw fluorescence, shape (events, channel)), 'uid_idx' and 'batch_idx' (encoded uid and batch,
            shape (events,)), 'peak_islinear' (usable-peak mask of each event's measurement, shape
            (events, peak, channel)).
        """
        if self.data_on_device:
            sample_indices = sample_indices.to(self.device)
        data = self.data[tp][sample_indices].to(self.device)
        uid_idx = self.data_uid_index[tp][sample_indices].to(self.device)
        batch_idx = self.data_batch_index[tp][sample_indices].to(self.device)
        type_peak_usemask = self.peak_usemask[tp].to(self.device)[self.data_intype_index[tp][sample_indices].to(self.device)]
        return {"data":data, "uid_idx":uid_idx, "batch_idx":batch_idx, "peak_islinear":type_peak_usemask}
    
    def sample_loaders(self, batch_size_override=None):
        """Draw a new subsample (`get_sample_indices`) and return one loader per rainbow type.

        Parameters
        ----------
        batch_size_override : int, optional
            Events per batch. Default: `batch_size`.

        Returns
        -------
        dict of str to DataLoader
            Loader per rainbow type, yielding dicts as returned by `get_batch`.
        """
        sample_indices = self.get_sample_indices()
        
        loaders = {tp: DataLoader(self, tp, sample_indices[tp], self.batch_size if batch_size_override is None else batch_size_override)
                   for tp in self.present_types}
        
        return loaders
    
    def sample(self):
        """Draw a new subsample and return all of its data at once, per rainbow type.

        Returns
        -------
        dict
            Keys as in `get_batch` ('data', 'uid_idx', 'batch_idx', 'peak_islinear'), each a dict of rainbow type to
            tensor on `device`.
        """
        sample_indices = self.get_sample_indices()
        
        data, uid_idx, batch_idx, intype_idx, type_peak_usemask = {}, {}, {}, {}, {}
        for tp in self.present_types:
            indices = sample_indices[tp].to(self.device) if self.data_on_device else sample_indices[tp]
            data[tp] = self.data[tp][sample_indices[tp]].to(self.device)
            uid_idx[tp] = self.data_uid_index[tp][sample_indices[tp]].to(self.device)
            batch_idx[tp] = self.data_batch_index[tp][sample_indices[tp]].to(self.device)
            type_peak_usemask[tp] = self.peak_usemask[tp].to(self.device)[self.data_intype_index[tp][sample_indices[tp]].to(self.device)]
        
        return {"data":data, "uid_idx":uid_idx, "batch_idx":batch_idx, "peak_islinear":type_peak_usemask}
