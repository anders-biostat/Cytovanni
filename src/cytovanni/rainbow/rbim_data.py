import numpy as np
import pandas as pd
import torch
import warnings
import itertools

from ..exceptions import IntegrationModuleException, IntegrationModuleWarning

class DataLoader():
    """ Simple dataloader from RainbowChannelGMMIntegrationDataset.
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

def _intype_GMM_crosstab_shares(index, label):
    """ Effectively the same as doing pd.crosstab(index, label), but much faster.
    """
    count = torch.zeros((index.max()+1, label.max()+1), dtype=torch.float32)
    for lb in torch.unique(label):
        u, c = torch.unique(index[label==lb], return_counts=True)
        count[u,lb] = c.to(torch.float32)
    return count / count.sum(-1, keepdims=True)

class Sampler():
    """ Sampler to get subsample indices.
        Faster on gpu by >10x
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
        """ Get one big permutation on gpu, much faster than many small ones.
            Argsort, add shift such that each group is in its own range, argsort again.
            Finally choose Nsubsample for each group and turn into indices in dataset.
        """
        full_permutation = torch.randperm(self.total, device=self.device, dtype=torch.int64)
        argsort_permutation = torch.argsort(full_permutation) + self.indices_group_add
        permutation = torch.argsort(argsort_permutation)
        subset_permutation = permutation[self.indices_choose]
        subsample_indices = self.indices[subset_permutation].cpu()
        return subsample_indices
    
    def permutation_cpu(self):
        """ Slower fallback on cpu.
        """
        get_permutation = lambda inds: inds[torch.randperm(len(inds), device=self.device)[:self.Nsubsample].cpu()]
        return torch.cat([get_permutation(inds) for inds in self.indices_list])
    
    def sample(self, split=False):
        """ If not split, returns the indices as one long tensor.
            If split, splits them according to the group they belong to.
        """
        if self.on_gpu:
            inds = self.permutation_cuda()
        else:
            inds = self.permutation_cpu()
        if not split:
            return inds
        
        groupinds = self.indices_group[inds]
        splitpos = torch.where(groupinds[1:] != groupinds[:-1])[0]+1
        splitinds = torch.tensor_split(inds, splitpos)
        return splitinds, groupinds[torch.cat([splitpos-1, torch.tensor([-1])])]

class RainbowFluorescenceGMMIntegrationDataset():
    """ Dataset for rainbow channel GMM integration.
    """
    def __init__(self, rboe, adatas, device="cpu", Nsubsample=1000, batch_size=20000, data_on_device=True, channel_missing_cutoff=-np.inf, warn_missing=True):
        """ 
            
            :param rboe: RainbowBatchOrdinalEncoder. Batch encoder, already setup.
            
            :param adatas: iterable. List of adatas to include in training.
            
            :param device: str. CUDA device to use, usually either 'cpu' or 'cuda:0'.
            
            :param Nsubsample: int. Subsample each sample to at most Nsubsample beads every time .sample() is used.
            
            :param batch_size: int. Batch size of the dataloaders returned by .sample_loaders().
            
            :param data_on_device: bool. Whether data should be shifted to device immediately, usually the best choice since it isn't that big anyway.
            
            :param channel_missing_cutoff: float. Usually, all channels need to be present for the integration to work. But when looking at calibration data etc. there may be rainbow bead measurements where a laser was turned off etc. Setting this to a finite value will treat all channels where every peak (in a given sample) is below this value as not fittable to avoid trying to fit pure noise. This is done by simply treating all peaks in that channel as nonlinear in the same way as for peaks that are off-scale.
            
            :param warn_missing: bool. Whether to warn about missing channels that will not get trained.
        """
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
        """ Make sure simple GMM was run on all samples.
        """
        if np.any([("GMM_label" not in ad.obs) or ("GMM_means" not in ad.uns) for ad in adatas]):
            raise IntegrationModuleException("Some of the adatas were not fitted with the simple GMM model!")
    
    @classmethod
    def check_type_Npeak_consistency(cls, adatas, prior={}):
        """ Mare sure that the number of peaks is consistent for every different rainbow bead type.
            Optionally also tests if it is the same as previously set.
        """
        for tp in np.unique([ad.uns["rainbow_type"] for ad in adatas]):
            uN = np.unique([ad.uns["GMM_means"].shape[0] for ad in adatas if ad.uns["rainbow_type"]==tp])
            if len(uN)>1 or (uN[0]==prior[tp] if tp in prior else False):
                IntegrationModuleException(f"Number of rainbow peaks not consistent for rainbow type {tp}!")
    
    @classmethod
    def get_type_Npeak(cls, adatas):
        """ Get number of peaks for every rainbow bead type.
        """
        cls.check_type_Npeak_consistency(adatas)
        return {tp: np.unique([ad.uns["GMM_means"].shape[0]
                               for ad in adatas if ad.uns["rainbow_type"]==tp])[0]
                for tp in np.unique([ad.uns["rainbow_type"] for ad in adatas])}
    
    def check_batch_training_missing(self):
        """ Check for batches where some parameters are not getting trained,
            either because all peaks are in the non-linear range, or because all peaks are too low.
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

        if (~self.df_channel_train.to_numpy()).sum()>0:
            warnstr = "For some batches, some channels will not get trained! Make sure this is intentional, and not caused by wrong settings for 'max_linear_range' or 'channel_missing_cutoff'."
            n_channel, Nc = ((~self.df_channel_train.to_numpy()).sum(0)>0).sum(), self.df_channel_train.shape[1]
            n_batch, Nb = ((~self.df_channel_train.to_numpy()).sum(1)>0).sum(), self.df_channel_train.shape[0]
            warnstr+= f"\n{n_channel} (out of {Nc}) channels are not trainable in some batches, {n_batch} (out of {Nb}) batches have some not trainable channels."
            warnstr+= f"\nA full accounting of all untrained channels can be found at 'dataset.df_channel_train'."
            warnings.warn(warnstr, IntegrationModuleWarning)
    
    def setup_data(self):
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
            raise IntegrationModuleException("Some of the rainbow batches in adatas were not set up in the ordinal encoder!")
        
        self.data_type_index = {tp: torch.cat([fct_single_index(ad, "rainbow_type") for ad in adatas])
                                 for tp, adatas in self.adatas_tp.items()}
        if sum([(v==-1).sum() for v in self.data_type_index.values()]).item()>0:
            raise IntegrationModuleException("Some of the rainbow types in adatas were not set up in the ordinal encoder!")
        
        self.data_uid_index = {tp: torch.cat([fct_single_index(ad, "uid") for ad in adatas])
                                 for tp, adatas in self.adatas_tp.items()}
        if sum([(v==-1).sum() for v in self.data_uid_index.values()]).item()>0:
            raise IntegrationModuleException("Some of the unique identifiers in adatas were not set up in the ordinal encoder!")
        
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
                                                            self.data_GMM_index[tp][self.data_usemask[tp]]) for tp in self.present_types}
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
        
        # check for not trainable channels in batches
        if self.warn_missing:
            self.check_batch_training_missing()
    
    def shift_data_to_device(self):
        """ Shift all relevant data to self.device.
        """
        keys_to_device = ["data", "data_uid_index", "data_batch_index", "data_intype_index", "peak_usemask"]
        for key in keys_to_device:
            setattr(self, key, {tp:getattr(self, key)[tp].to(self.device) for tp in getattr(self, key)})

    def get_sample_indices(self):
        """ Get sampled indices, subsample to N beads per rainbow bead sample.
        """
        return {tp:self.batch_index_sampler[tp].sample() for tp in self.batch_index_sampler}
    
    def get_batch(self, tp, sample_indices):
        """ Get a single batch from indices.
        """
        if self.data_on_device:
            sample_indices = sample_indices.to(self.device)
        data = self.data[tp][sample_indices].to(self.device)
        uid_idx = self.data_uid_index[tp][sample_indices].to(self.device)
        batch_idx = self.data_batch_index[tp][sample_indices].to(self.device)
        type_peak_usemask = self.peak_usemask[tp].to(self.device)[self.data_intype_index[tp][sample_indices].to(self.device)]
        return {"data":data, "uid_idx":uid_idx, "batch_idx":batch_idx, "peak_islinear":type_peak_usemask}
    
    def sample_loaders(self, batch_size_override=None):
        """ Subsample to N beads per rainbow bead sample, return dataloader for each bead type.
        """
        sample_indices = self.get_sample_indices()
        
        loaders = {tp: DataLoader(self, tp, sample_indices[tp], self.batch_size if batch_size_override is None else batch_size_override)
                   for tp in self.present_types}
        
        return loaders
    
    def sample(self):
        """ Subsample to N beads per rainbow bead sample, return their data.
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
