"""Training data of the overlap fit: samples measured in several batches and the connections between them."""
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
    """Compute the shortest path lengths (number of edges) between all nodes of a symmetric graph.

    Parameters
    ----------
    graph : numpy.ndarray of shape (node, node)
        Adjacency matrix (non-zero = edge), symmetric.
    filldiag : bool, default True
        Set the diagonal (distance of a node to itself) to 1 instead of 0.

    Returns
    -------
    numpy.ndarray of shape (node, node)
        Path lengths, ``numpy.inf`` for pairs without a connecting path.
    """
    dist = np.full(graph.shape, np.inf)
    for i in range(graph.shape[0]):
        dct = sklearn.utils.graph.single_source_shortest_path_length(graph, i)
        arr = np.asarray([(k,v) for k,v in dct.items()])
        dist[arr[:,0], i] = arr[:,1]
        if filldiag: dist[i,i] = 1
    return dist

import itertools
def generate_unique_connections(*inputs):
    """Yield the tuples of ``itertools.product(*inputs)`` without repeats and without all-equal tuples.

    Tuples that contain the same set of items are yielded only once (the first one), tuples whose items are all
    equal are skipped. Adapted from https://stackoverflow.com/questions/65058154.

    Parameters
    ----------
    *inputs : iterable
        Iterables passed to ``itertools.product``; items must be hashable.

    Yields
    ------
    tuple
        One tuple per distinct set of items.
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
    """Stack left and right sample indices into a connection array of shape (2, connections)."""
    return np.vstack([left, right])
def unique_connections(connections):
    """Sort each connection (column) of a (2, connections) array and drop duplicates, so (i, j) and (j, i) merge."""
    return np.unique(np.sort(connections, axis=0), axis=1)
class ConnectionSampler():
    """Sample the connections (pairs of samples) that the overlap fit compares in each step.

    A connection joins two measurements of the same sample (same `idx_id`) in different batches. Without anchors,
    all such connections are used. With anchors, connections are grown outwards from the anchor batches: first all
    connections that contain a sample of an anchor batch, then, for d = 1 .. ``max_anchor_dist - 1``, connections
    from already connected batches at anchor distance d to batches at another distance. So batches up to
    `max_anchor_dist` connections away from an anchor batch are trained; batches further away, or without any path to
    an anchor batch, are not (a warning names them, see `find_nontraining_batches`).

    The anchor distance of a batch is the number of steps in the batch graph (two batches are linked if they share a
    sample) to the nearest batch that contains an anchor sample. Every sample of such a batch has distance 0, also
    the samples not marked as anchors themselves.

    Each step uses all partners of a sample that has at most `Nmax_persample` of them, and a random choice of
    `Nmax_persample` partners (numpy's global random state) for the others. Connections are undirected and merged, so
    a sample can appear in more connections than `Nmax_persample` when other samples choose it.

    Parameters
    ----------
    idx_id : array-like of shape (sample,)
        Sample identity: entries with the same value are measurements of the same sample and get connected.
    idx_batch : array-like of int, shape (sample,)
        Batch index (0-based integer) of every entry. Connections within the same batch are blocked, unless
        `block_within_batch` is False.
    is_anchor : array-like of bool, shape (sample,), optional
        Marks the samples of the anchor batches (their batch is treated as an anchor batch). Default: no anchors,
        all connections are used (warning: recommended only if no panel embeddings are fitted).
    Nmax_persample : int, default 2
        Number of partners drawn per sample and step (see above).
    connect_train_within_anchors : bool, default False
        Also connect two anchor samples with the same `idx_id`, even within the same batch.
    max_anchor_dist : int, default 1
        With anchors: train batches up to this anchor distance. 1 uses only connections that contain an anchor-batch
        sample. Must be at least 1.
    block_within_batch : bool, default True
        Block connections between entries with the same `idx_batch`.
    train_higher : bool, default True
        Train both sides of every connection. If False (with anchors), only the side further away from an anchor is
        trained, and both sides of connections between two anchor-distance-0 samples.

    Raises
    ------
    ValueError
        If `max_anchor_dist` is below 1, or there is no connection to train on.

    Warns
    -----
    UserWarning
        Without anchors; for batches without a path to an anchor batch; for batches further than `max_anchor_dist`
        from an anchor batch.

    Notes
    -----
    Which samples are anchors only decides which connections are compared. Keeping the anchor batches' parameters
    fixed is done separately, by the `anchor_bidx` of the transformer and the panel embedding module.
    """
    def __init__(self, idx_id, idx_batch, is_anchor=None, Nmax_persample=2, connect_train_within_anchors=False, max_anchor_dist=1, block_within_batch=True, train_higher=True):
        if max_anchor_dist<1:
            raise ValueError(f"max_anchor_dist must be at least 1, got {max_anchor_dist}!")
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
            self.idx_anchor_distance = np.full(len(self.idx_batch), np.nan) # no anchors, no distance
            
        else:
            batch_is_anchor = np.zeros(self.idx_batch.max()+1)
            batch_is_anchor[self.idx_batch[self.is_anchor]] += 1
            self.batch_is_anchor = batch_is_anchor>0
            
            shortest_connection = graph_shortest_path(self.batch_connections_diffbatch)
            shortest_connection[self.batch_is_anchor, self.batch_is_anchor] = 0.
            self.batch_anchor_distance = shortest_connection[self.batch_is_anchor].min(0)
            unreachable = np.setdiff1d(np.where(np.isinf(self.batch_anchor_distance))[0], np.where(self.batch_is_anchor)[0])
            unreachable = np.intersect1d(unreachable, self.idx_batch)
            if len(unreachable)>0:
                warnings.warn(f"Batches {unreachable.tolist()} have no connection to any anchor batch and are not trained (see find_nontraining_batches)!")
            self.idx_anchor_distance = self.batch_anchor_distance[self.idx_batch]
            
            connections = (self.connections_diffbatch>0) & ((self.idx_anchor_distance[None]==0) | (self.idx_anchor_distance[:,None]==0))

            for i in range(1, self.max_anchor_dist):

                batch_connections = np.zeros(self.idx_batch.max()+1)
                csum = connections.sum(0)
                batch_connections[self.idx_batch[csum>0]] =1
                is_connected = (batch_connections>0)[self.idx_batch]

                condition_next = is_connected[None] & (self.idx_anchor_distance[None]==i) & (self.idx_anchor_distance[None] != self.idx_anchor_distance[:,None])
                condition_next = condition_next | condition_next.T
                connections = connections | (self.connections_diffbatch & condition_next)
            
            self.connections = connections
        
        np.fill_diagonal(self.connections, False)
        
        if self.is_anchor is not None:
            nontraining = np.intersect1d(self.find_nontraining_batches(), self.idx_batch)
            beyond = nontraining[np.isfinite(self.batch_anchor_distance[nontraining]) & ~self.batch_is_anchor[nontraining]]
            if len(beyond)>0:
                warnings.warn(f"Batches {beyond.tolist()} (anchor distances {self.batch_anchor_distance[beyond].astype(int).tolist()}) are connected to an anchor batch, "
                              f"but further than max_anchor_dist={self.max_anchor_dist} away, and are not trained (see find_nontraining_batches)!")
        
        # find connections
        self.cidx_left, self.cidx_right = self.connections.nonzero()
        if len(self.cidx_left)==0:
            raise ValueError("No connections between samples of different batches (and, with anchors, within max_anchor_dist of an anchor), nothing to train on!")
        
        # split connections into fixed if not many, and sampled if many
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
        """Draw `Nmax_persample` random partners for every sample with more partners than that.

        Returns
        -------
        numpy.ndarray of int, shape (2, connections)
            Sample indices (row 0: the sample, row 1: the drawn partner); empty (2, 0) if no sample has more partners.
        """
        if len(self.connections_sampledct)==0: return np.zeros((2,0), dtype=int)
        sample = lambda v: list(np.random.choice(v, size=self.Nmax_persample, replace=False))
        return np.hstack([[[k]*self.Nmax_persample, sample(v)] for k, v in self.connections_sampledct.items()])
    
    def sample(self):
        """Draw the connections for one training step.

        Returns
        -------
        connections : numpy.ndarray of int, shape (2, connections)
            Sample indices of both sides; every connection once, with the smaller index in row 0.
        ctrain : numpy.ndarray of bool, shape (2, connections)
            Whether each side is trained (gradient passed through). All True if `train_higher` or without anchors.
        """
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
        """Draw one set of connections (`sample`) and return it as a table, to inspect the sampling.

        Returns
        -------
        pandas.DataFrame
            One row per connection, columns ``train_left``, ``train_right`` (bool, side is trained), ``adist_left``,
            ``adist_right`` (anchor distance of each side, NaN without anchors) and ``idx_left``, ``idx_right``
            (sample indices).
        """
        connections, ctrain = self.sample()
        
        df = pd.DataFrame(np.vstack([ctrain, self.idx_anchor_distance[connections]]), index=["train_left","train_right","adist_left","adist_right"]).T
        df[["train_left","train_right"]] = df[["train_left","train_right"]].astype(bool)
        df[["idx_left","idx_right"]] = connections.T
        return df
    
    def find_nontraining_batches(self):
        """Return the batch indices that take part in no connection.

        Returns
        -------
        numpy.ndarray of int
            Batch indices from 0 to ``max(idx_batch)`` without any connection, e.g. batches without a path to an
            anchor batch, further than `max_anchor_dist` from one, or without a shared sample. Can include indices
            that do not occur in `idx_batch`.
        """
        batch_connections = np.zeros(self.idx_batch.max()+1)
        csum = self.connections.sum(0)
        batch_connections[self.idx_batch[csum>0]]=1
        return np.where(batch_connections==0)[0]


class SampleOverlapDataset():
    """Training data for `OverlapFitModel`: samples measured in several batches, and the pairs to compare.

    Holds the events of all samples (one AnnData per measurement) as one tensor, draws a random subsample of
    `sN_cells` events per sample in every step, and the connections (pairs of measurements of the same sample in
    different batches) via a `ConnectionSampler`. The metadata (sample identity, batch labels, covariates) is read
    from ``adata.uns`` of every sample.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        One AnnData per measurement. ``adata.uns`` must contain the keys of `moe` (``moe.necessary_keys``),
        `key_id`, `key_uid`, `keys_sample_covariate` and `key_batch_sample`.
    moe : MultiOrdinalEncoder
        Batch encoder for all batch keys of the model.
    channels : list of str
        Channels to load, in the order of the transformer's channels (`OverlapFitModel` checks this).
    key_layer : str, default "calibrated"
        Layer with the data, ``adata.layers[key_layer]`` of shape (events, channel), usually rainbow-calibrated
        intensities (linear scale).
    load_obsm : bool, default False
        Load the data from ``adata.obsm[key_layer][channels]`` (a DataFrame) instead of the layer, e.g. already
        unmixed data for a `ScalingTransformer` (see `make_dummy_spectra`).
    key_batch_sample : str or None, default "batch"
        Batch key (one of the keys of `moe`) whose batches block connections: two measurements in the same batch are
        not compared. None: no blocking; with anchors, every sample then counts as its own batch for the anchor
        distances, so these count connections between samples.
    data_isanchor : array-like of bool, shape (len(adatas),), optional
        Marks the measurements of the anchor batches, e.g. from `adatas_process_anchors`. With anchors, only
        connections within `max_anchor_dist` of an anchor batch are used (see `ConnectionSampler`). Default: no
        anchors, all connections are used. The anchor parameters are fixed separately (``anchor_bidx`` of the
        transformer).
    key_id : str, default "type"
        Key in ``adata.uns`` of the sample identity; measurements with the same value are connected.
    key_uid : str, default "uid"
        Key in ``adata.uns`` of a unique identifier of every measurement; labels the per-connection loss matrices of
        `OverlapFitModel.get_DDIST_from_dataset`.
    keys_sample_covariate : list of str, default []
        Further keys in ``adata.uns`` passed on per sample (``samp_cov`` in `sample_single`) to the marker weighting.
    sN_cells : int, default 2000
        Events drawn per sample in every step (all events if a sample has fewer).
    sN_persample : int, default 2
        Number of partners drawn per sample and step (`Nmax_persample` of `ConnectionSampler`).
    max_anchor_dist : int, default 2
        With anchors: train batches up to this many connections away from an anchor batch (see `ConnectionSampler`);
        1 uses only connections that contain an anchor-batch sample.
    device : str, default "cpu"
        Torch device the samples are returned on, e.g. ``"cuda:0"``.
    data_on_device : bool, default True
        Move all data to `device` at construction (faster sampling). False keeps the data on the CPU and moves only
        the drawn events, for data that does not fit into GPU memory.
    min_sample_cells : int, default 1000
        Minimum number of events per sample.

    Raises
    ------
    ValueError
        If a sample has fewer than `min_sample_cells` events; from `ConnectionSampler` if there is nothing to train on.

    Warns
    -----
    UserWarning
        If some samples have no matching sample (same `key_id`) in the data; the `ConnectionSampler` warnings.
    """
    def __init__(self, adatas, moe, channels, key_layer="calibrated", load_obsm=False, key_batch_sample="batch", data_isanchor=None, key_id="type", key_uid="uid",
                       keys_sample_covariate=[], sN_cells=2000, sN_persample=2, max_anchor_dist=2, device="cpu", data_on_device=True, min_sample_cells=1000):
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
        self.max_anchor_dist = max_anchor_dist
        
        self.device = device
        self.data_on_device = data_on_device
        
        self.data_isanchor = None if data_isanchor is None else np.asarray(data_isanchor)
        
        self.setup()
    
    def setup(self):
        """Load the metadata and data and build the event and connection samplers (called by the constructor).

        Sets ``meta`` (DataFrame, one row per sample: the ``adata.uns`` keys used and ``_N``, the number of events),
        ``data`` (torch.Tensor of shape (events, channel), all samples concatenated), ``data_idx_batch`` (dict of batch
        key to torch.Tensor of shape (sample,), batch indices from `moe`), ``data_idx_id``, ``cellsampler`` and
        ``csampler`` (`ConnectionSampler`).
        """
        meta_keys = set(self.moe.necessary_keys) | set([self.key_id, self.key_uid]) | set(self.keys_sample_covariate)
        if self.key_batch_sample is not None:
            meta_keys.add(self.key_batch_sample)
        meta_keys = list(meta_keys)
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
        
        # without key_batch_sample: no blocking, all connections allowed; every sample is its own "batch" for the
        # anchor distances, so these count connections between samples
        if self.key_batch_sample is None:
            idx_batch, block_within_batch = np.arange(self.Nsample), False
        else:
            idx_batch, block_within_batch = self.data_idx_batch[self.key_batch_sample].numpy(), True
        self.csampler = ConnectionSampler(self.data_idx_id.numpy(), idx_batch,
                                          is_anchor=self.data_isanchor, Nmax_persample=self.sN_persample,
                                          connect_train_within_anchors=False, max_anchor_dist=self.max_anchor_dist,
                                          block_within_batch=block_within_batch, train_higher=True)
    
    def sample_single(self, idx, sampinds=None, expand_idx=True):
        """Return a random subsample of one sample, with its batch indices.

        Parameters
        ----------
        idx : int
            Index of the sample (position in `adatas`).
        sampinds : torch.Tensor of int, optional
            Event indices into ``data`` to use. Default: draw up to `sN_cells` events of the sample (`sample`
            draws them for all samples at once, which is faster).
        expand_idx : bool, default True
            Repeat the batch indices for every event (shape (events,)); if False, keep one entry (shape (1,)).

        Returns
        -------
        dict
            ``'x'``: torch.Tensor of shape (events, channel), on `key_layer`'s scale; ``'idx_batch'``: dict of batch
            key to torch.Tensor of int; ``'idx_id'``: torch.Tensor of int (sample identity index); ``'samp_cov'``:
            dict of covariate key to value. Tensors are on `device`.
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
        """Draw the data for one training step: a subsample of every sample, and the connections to compare.

        Parameters
        ----------
        expand_idx : bool, default True
            Passed to `sample_single`.

        Returns
        -------
        dict
            ``'data'``: list with the `sample_single` output of every sample (in the order of `adatas`);
            ``'cidx_left'``, ``'cidx_right'``: numpy.ndarray of int, sample indices of the two sides of each
            connection; ``'dotr_left'``, ``'dotr_right'``: numpy.ndarray of bool, whether each side is trained.
        """
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
        """Number of samples (AnnDatas)."""
        return self.Nsample
    
    def __getitem__(self, idx):
        """Random subsample of sample `idx`, see `sample_single`."""
        return self.sample_single(idx)

    def make_dummy_spectra(self):
        """Return identity spectra for the dataset's channels.

        For a `ScalingTransformer` on data that is already unmixed (``load_obsm=True``): with identity spectra the
        unmixing does nothing and only the stain scaling is fitted.

        Returns
        -------
        pandas.DataFrame of shape (channel, channel)
            Identity matrix, index and columns the dataset's channels (read as (stain, channel)).
        """
        return pd.DataFrame(np.identity(len(self.channels)), index=self.channels, columns=self.channels)

def adatas_process_anchors(adatas, anchors):
    """Mark the AnnDatas that belong to an anchor batch.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        Samples; the batch labels are read from ``adata.uns``.
    anchors : dict of str to list
        Anchor batch labels per batch key, e.g. ``{"batch": ["18-APR-2024"]}``.

    Returns
    -------
    numpy.ndarray of bool, shape (len(adatas),)
        True for every AnnData whose label matches an anchor label for any of the keys; use as `data_isanchor` of
        `SampleOverlapDataset`.

    Examples
    --------
    >>> anchors = {"batch": ["18-APR-2024"]}
    >>> isanchor = adatas_process_anchors(adatas, anchors)
    >>> anchor_bidx = moe.transform_dict_simple(anchors)  # the matching anchor_bidx for the transformer
    """
    return np.any(np.asarray([[np.isin(ad.uns[k], anchors[k]) for ad in adatas] for k,v in anchors.items()]), 0)
