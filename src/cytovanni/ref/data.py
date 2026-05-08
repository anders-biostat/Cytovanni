import torch
import numpy as np
from torch.nn import Module, Parameter
import torch.nn.functional as F
import pandas as pd
import warnings

from ..utils import CustomOrdinalEncoder
from ..rainbow.rbim_data import Sampler


import sklearn
def graph_shortest_path(graph, filldiag=True):
    """ Shortest path in symmetric graph
        filldiag: if True, fill diag with 1
    """
    dist = np.zeros(graph.shape)
    for i in range(graph.shape[0]):
        dct = sklearn.utils.graph.single_source_shortest_path_length(graph, i)
        arr = np.asarray([(k,v) for k,v in dct.items()])
        dist[arr[:,0], i] = arr[:,1]
        if filldiag: dist[i,i] = 1
    return dist

import itertools
def generate_unique_connections(*inputs):
    """ https://stackoverflow.com/questions/65058154/how-to-generate-itertools-product-without-duplicates
    """
    seen = set()
    for prod in itertools.product(*inputs):
        prod_set = frozenset(prod)
        if len(prod_set) == 1:  # all items are the same
            continue
        if prod_set not in seen:
            seen.add(prod_set)
            yield prod

def idx_to_connections(left, right):
    return np.vstack([left, right])
def unique_connections(connections):
    return np.unique(np.sort(connections, axis=0), axis=1)
class ConnectionSampler():
    """ Class to sample overlap connections.
        
        Only use connections that are not in the same batch.
        For every connection, only trains the side that is further away from an anchor.
    """
    def __init__(self, idx_id, idx_batch, is_anchor=None, Nmax_persample=2, connect_train_within_anchors=False, max_anchor_dist=1, block_within_batch=True, train_higher=True):
        """ 
            
            :param idx_id: iterable. Sample id, connects entries where this is the same.
            
            :param idx_batch: iterable. Batch index, connections with same batch index are blocked, can disable this with block_within_batch.
            
            :param is_anchor: None, iterable. If none, sample all connections. If boolean mask, only sample connections that contain anchors.
            
            :param Nmax_persample: int. Maximum number of connections to sample per Sample.
            
            :param connect_train_within_anchors: bool. Whether to also allow connections between two anchors.
            
            :param max_anchor_dist: int. Only include samples that are at most max_anchor_dist connections away from an anchor. Set to one to only sample connections with anchors.
            
            :param block_within_batch: bool. Whether to block connections within the same batch.
            
            :param train_higher: bool. Whether to also train the side of connections that is closer to an anchor.
        """
        self.Nmax_persample = Nmax_persample
        self.connect_train_within_anchors = connect_train_within_anchors
        self.max_anchor_dist = max_anchor_dist
        self.train_higher = train_higher
        
        # data
        self.idx_id = np.asarray(idx_id)
        self.idx_batch = np.asarray(idx_batch)
        self.is_anchor = is_anchor if is_anchor is None else np.asarray(is_anchor)
        
        # basic comparisons
        condition_id = self.idx_id[None] == self.idx_id[:,None]
        condition_batch = self.idx_batch[None] != self.idx_batch[:,None]
        if not block_within_batch:
            condition_batch[...] = True
        if self.is_anchor is None or not self.connect_train_within_anchors: condition_anchor = False
        else: condition_anchor = self.is_anchor[None,:] & self.is_anchor[:,None]
        self.connections_naive = condition_id
        self.connections_diffbatch = (condition_id & (condition_batch | condition_anchor))
        
        connections_batch = np.diag(np.ones(self.idx_batch.max()+1)) #np.zeros([self.idx_batch.max()+1]*2)
        idxleft, idxright = self.connections_diffbatch.nonzero()
        connections_batch[self.idx_batch[idxleft], self.idx_batch[idxright]] = 1.
        self.batch_connections_diffbatch = connections_batch
        
        # deal with anchors
        if self.is_anchor is None:
            # no anchors, train both on all connections. not recommended if using panel embedding, otherwise fine
            warnings.warn("No anchor batches were given, this is only recommended if no panel embeddings are being fit.")
            self.connections = self.connections_diffbatch
            
        else:
            batch_is_anchor = np.zeros(self.idx_batch.max()+1)
            batch_is_anchor[self.idx_batch[self.is_anchor]] += 1
            self.batch_is_anchor = batch_is_anchor>0
            
            shortest_connection = graph_shortest_path(self.batch_connections_diffbatch)
            shortest_connection[self.batch_is_anchor, self.batch_is_anchor] = 0.
            self.batch_anchor_distance = shortest_connection[self.batch_is_anchor].min(0)
            self.idx_anchor_distance = self.batch_anchor_distance[self.idx_batch]
            
            connections = (self.connections_diffbatch>0) & ((self.idx_anchor_distance[None]==0) | (self.idx_anchor_distance[:,None]==0))

            for i in list(range(1, self.max_anchor_dist+1)):

                batch_connections = np.zeros(self.idx_batch.max()+1)
                csum = connections.sum(0)
                batch_connections[self.idx_batch[csum>0]] =1
                is_connected = (batch_connections>0)[self.idx_batch]

                condition_next = is_connected[None] & (self.idx_anchor_distance[None]==i) & (self.idx_anchor_distance[None] != self.idx_anchor_distance[:,None])
                condition_next = condition_next | condition_next.T
                connections = connections | (self.connections_diffbatch & condition_next)
            
            self.connections = connections
        
        np.fill_diagonal(self.connections, False)
        
        # find connections
        self.cidx_left, self.cidx_right = self.connections.nonzero()
        
        # split connections into fixed if not many, und sampled if many
        u, c = np.unique(self.cidx_left, return_counts=True)
        count = np.zeros(self.cidx_left.max()+1)
        count[u] = c
        cidx_count = count[self.cidx_left]
        
        # get trained in every step
        self.connections_fixed = idx_to_connections(self.cidx_left [cidx_count<=self.Nmax_persample],
                                                      self.cidx_right[cidx_count<=self.Nmax_persample])
        self.connections_fixed = unique_connections(self.connections_fixed)
        # get trained randomly
        sc_left, sc_right = self.cidx_left[cidx_count>self.Nmax_persample], self.cidx_right[cidx_count>self.Nmax_persample]
        self.connections_sampledct = {c:sc_right[sc_left==c] for c in np.unique(sc_left)}
    
    def sample_from_sampledct(self):
        if len(self.connections_sampledct)==0: return np.zeros((2,0), dtype=int)
        sample = lambda v: list(np.random.choice(v, size=self.Nmax_persample, replace=False))
        return np.hstack([[[k]*self.Nmax_persample, sample(v)] for k, v in self.connections_sampledct.items()])
    
    def sample(self):
        connections = unique_connections(np.hstack([self.connections_fixed, unique_connections(self.sample_from_sampledct())]))
        
        if self.train_higher or (self.is_anchor is None):
            ctrain = np.full(connections.shape, True)
        else:
            # only train side with larger distance to anchors
            cond1 = np.vstack([self.idx_anchor_distance[connections[0]] > self.idx_anchor_distance[connections[1]],
                               self.idx_anchor_distance[connections[0]] < self.idx_anchor_distance[connections[1]]])
            # also train anchors
            cond2 = (self.idx_anchor_distance[connections[0]]==0) & (self.idx_anchor_distance[connections[1]]==0)
            cond2 = np.vstack([cond2, cond2])
            # combine
            ctrain = cond1 | cond2
        
        return connections, ctrain
    
    def get_df_sample_connections(self):
        connections, ctrain = self.sample()
        
        df = pd.DataFrame(np.vstack([ctrain, self.idx_anchor_distance[connections]]), index=["train_left","train_right","adist_left","adist_right"]).T
        df[["train_left","train_right"]] = df[["train_left","train_right"]].astype(bool)
        df[["idx_left","idx_right"]] = connections.T
        return df
    
    def find_nontraining_batches(self):
        """ Return indices for all batches that have no connections.
        """
        batch_connections = np.zeros(self.idx_batch.max()+1)
        csum = self.connections.sum(0)
        batch_connections[self.idx_batch[csum>0]]=1
        return np.where(batch_connections==0)[0]


class SampleOverlapDataset():
    """ 
    """
    def __init__(self, adatas, moe, channels, key_layer="calibrated", load_obsm=False, key_batch_sample="batch", data_isanchor=None, key_id="type", key_uid="uid",
                       keys_sample_covariate=[], sN_cells=2000, sN_persample=2, device="cpu", data_on_device=True, min_sample_cells=1000):
        """ 
            
            :param adatas: itarable. List of the adatas to include.
            
            :param moe: MultiOrdinalEncoder. Encoder for the batches.
            
            :param channels: iterable. List of the channels to include.
            
            :param key_layer: str. Layer key to use for data.

            :param load_obsm: bool. Normally, loads data from adata.layers[key_layer], setting this to True switches the dataset to loading from adata.obsm[key_layer]
            
            :param key_batch_sample: str. Batch key to use when blocking connections withint batches for sampling. If set to None, uses all possible connections.
            
            :param data_isanchor: None, iterable. If none, just sample connections normally. If iterable, should be boolean that indicates for every adata whether it should be treated as a standardisation anchor. Then, only connections that include at least one anchor are used for training.
            
            :param key_id: str. Key of sample id in adata.uns that is used to generate connections.
            
            :param key_uid: str. Key to unique identifier of every single measurement of a sample.
            
            :param keys_sample_covariate: iterable. Other covariate keys in adata.uns to include when sampling data.
            
            :param sN_cells: int. Random subset size per sample.
            
            :param sN_persample: int. Number of connections to use per sample.
            
            :param device: str. Automatically transfer data to CUDA device, or keep on 'cpu'.
            
            :param data_on_device: bool. By default, sends the data to the device directly instead of only after sampling. Faster, but can be deactivated if GPU memory is too small.

            :param min_sample_cells: int. Minimal number of cells to allow per sample, throws an error if any have less than that.
        """
        self.adatas = adatas
        small_adatas = [ad.shape[0]<min_sample_cells for ad in adatas]
        if sum(small_adatas)>0:
            raise ValueError(f"There are {sum(small_adatas)} samples in the training data that have less than {min_sample_cells} cells! Remove empty or very small samples from the fit.")
        self.moe = moe
        self.channels = channels
        self.Nsample = len(adatas)
        self.Nchannel = len(self.channels)
        
        self.key_layer = key_layer
        self.load_obsm = load_obsm
        self.key_batch_sample = key_batch_sample
        self.key_id = key_id
        self.key_uid = key_uid
        self.keys_sample_covariate = keys_sample_covariate
        
        self.sN_cells = sN_cells
        self.sN_persample = sN_persample
        
        self.device = device
        self.data_on_device = data_on_device
        
        self.data_isanchor = None if data_isanchor is None else np.asarray(data_isanchor)
        
        self.setup()
    
    def setup(self):
        meta_keys = list( set(self.moe.necessary_keys) | set([self.key_batch_sample, self.key_id, self.key_uid]) )
        self.meta = pd.DataFrame({k:[ad.uns[k] for ad in self.adatas]
                                  for k in meta_keys})
        self.meta["_N"] = [ad.shape[0] for ad in self.adatas]
        
        self.oe_id = CustomOrdinalEncoder.from_labels(self.meta[self.key_id])
        self.Nid = len(self.oe_id)

        u, c = np.unique(self.meta[self.key_id], return_counts=True)
        if (c<2).sum()>0:
            warnings.warn("Some samples do not have any other matching samples!")

        def load_from_ad(ad):
            if self.load_obsm:
                return ad.obsm[self.key_layer][self.channels].to_numpy()
            else:
                return ad[:,self.channels].layers[self.key_layer]
        self.data = torch.cat([torch.as_tensor(load_from_ad(ad), dtype=torch.float32) for ad in self.adatas])
        if self.data_on_device:
            self.data = self.data.to(self.device)
        
        self.data_idx_batch = self.moe.transform_df(self.meta)
        self.data_idx_id = torch.tensor(self.oe_id.transform(self.meta[self.key_id]), dtype=torch.long)
        
        cNs = np.cumsum([0]+[ad.shape[0] for ad in self.adatas])
        self.data_split_indices = [torch.arange(cNs[i], cNs[i+1]) for i in range(len(cNs)-1)]
        
        # stick with numpy for covariates, torch tensors cannot handle strings etc.
        self.data_sample_covariate = {k:self.meta[k].to_numpy() for k in self.keys_sample_covariate}
        
        self.cellsampler = Sampler(self.data_split_indices, self.sN_cells, device=self.device)
        
        idx_batch = self.data_idx_id.numpy() if self.key_batch_sample is None else self.data_idx_batch[self.key_batch_sample].numpy()
        self.csampler = ConnectionSampler(self.data_idx_id.numpy(), idx_batch,
                                          is_anchor=self.data_isanchor, Nmax_persample=self.sN_persample,
                                          connect_train_within_anchors=False, max_anchor_dist=1, block_within_batch=True, train_higher=True)
    
    def sample_single(self, idx, sampinds=None, expand_idx=True):
        """ Sample from single sample idx.
            Can generate sample indices, but faster to generate them all at once using Sampler.
            Can either expand the batch indices to the same shape as the events, or leave them as single numbers.
        """
        if sampinds is None:
            inds = self.data_split_indices[idx]
            sampinds = inds[torch.randperm(inds.shape[0])[:self.sN_cells]]
        x = self.data[sampinds]
        fct_expand = lambda i: i.expand(x.shape[0]) if expand_idx else i[None]
        return {"x":x.to(self.device),
                "idx_batch":{k:fct_expand(data_idx_batch[idx]).to(self.device) for k, data_idx_batch in self.data_idx_batch.items()},
                "idx_id":fct_expand(self.data_idx_id[idx]).to(self.device),
                "samp_cov":{k:v[idx] for k, v in self.data_sample_covariate.items()}
               }
    
    def sample(self, expand_idx=True):
        sample_inds, idxs = self.cellsampler.sample(split=True)
        sample_data = [self.sample_single(idxs[i], sample_inds[i], expand_idx=expand_idx) for i in range(len(sample_inds))]
        
        connections, ctrain = self.csampler.sample()
        cidx_left, cidx_right = connections[0], connections[1]
        
        return {"data":sample_data,
                
                "cidx_left":cidx_left,
                "cidx_right":cidx_right,
                "dotr_left":ctrain[0],
                "dotr_right":ctrain[1],
                }
    
    def __len__(self):
        return self.Nsample
    
    def __getitem__(self, idx):
        return self.sample_single(idx)

    def make_dummy_spectra(self):
        return pd.DataFrame(np.identity(len(self.channels)), index=self.channels, columns=self.channels)

def adatas_process_anchors(adatas, anchors):
    """ Anchors should be dictionary of batch_key:iterable,
        returns True for every adata where any batch matches one of the anchor batches.
    """
    return np.any(np.asarray([[np.isin(ad.uns[k], anchors[k]) for ad in adatas] for k,v in anchors.items()]), 0)
