"""Encoding of the batch labels of several batch keys into the batch indices of the model parameters."""
import torch
import numpy as np
import pandas as pd

from ..utils import CustomOrdinalEncoder
from ..torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict

class MultiOrdinalEncoder():
    """Encode the labels of several batch keys into integer batch indices at once.

    Every batch key (e.g. measurement day, plate, reference aliquot) gets its own ordinal encoder; the model
    parameters are indexed by these batch indices. Create it with `from_data` or `from_adatas` (or `from_exported`);
    the constructor creates an empty object.

    Nested batches allow imputation for batches not seen during the fit. With ``nested={"a": ["b", "b_order"]}``,
    batch "a" is assumed to be nested within batch "b", with "b_order" a numeric order within "b" (e.g. the plate
    number within a day). A sample whose value of "a" is unknown, but whose "b" is known, gets the index of the known
    "a" batch within the same "b" whose "b_order" is closest; for equal distances the lower "b_order" wins.

    Attributes
    ----------
    keys_batch : list of str
        The encoded batch keys.
    labels : dict of str to numpy.ndarray
        Known labels per batch key (as str), in the order of their indices.
    Nbatch : dict of str to int
        Number of known labels per batch key.
    necessary_keys : list of str
        All keys a sample needs in its metadata (batch keys plus the keys of `nested`).

    Examples
    --------
    >>> moe = MultiOrdinalEncoder.from_adatas(adatas, keys_batch=["batch", "batch_ref"],
    ...                                       nested={"batch_ref": ["batch", "order_batch_ref"]})
    >>> bidx = moe.transform_ad(adata)  # {"batch": tensor([3]), "batch_ref": tensor([7]), "batch_ref_imputed": ...}
    """
    def __init__(self):
        pass
    
    @classmethod
    def from_data(cls, meta, keys_batch, nested={}):
        """Create the encoder from a metadata table.

        Parameters
        ----------
        meta : pandas.DataFrame
            One row per sample, with a column for every batch key and for the keys used in `nested`.
        keys_batch : list of str
            Batch keys to encode. Labels are converted to str and sorted.
        nested : dict of str to list of str, default {}
            ``{key: [major_key, order_key]}`` for nested batch keys (see the class description). Only nested keys
            accept unknown labels in the transforms; the nested key must also be in `keys_batch`.

        Returns
        -------
        MultiOrdinalEncoder
        """
        moe = cls()
        moe.keys_batch = keys_batch
        moe.oe_batches = {key:CustomOrdinalEncoder.from_labels(meta[key], allow_unknown=key in nested) for key in moe.keys_batch}
        moe.Nbatch = {k:len(oe) for k, oe in moe.oe_batches.items()}
        moe.labels = {k:oe.labels for k, oe in moe.oe_batches.items()}
        
        moe.nested = nested
        moe.nested_keys_batch    = [k    for k, v in moe.nested.items()]
        moe.nested_keys_supbatch = [v[0] for k, v in moe.nested.items()]
        moe.nested_keys_order    = [v[1] for k, v in moe.nested.items()]
        moe.nested_data = {k:meta[[v[0], k, v[1]]].sort_values([v[0], v[1]]) for k, v in moe.nested.items()}
        for k, df in moe.nested_data.items():
            df.columns = ["major","minor","order"]
            df["minor_idx"] = moe.oe_batches[k].transform(df["minor"])
            moe.nested_data[k] = moe.nested_data[k][~moe.nested_data[k].duplicated()].reset_index(drop=True)
        
        moe.necessary_keys = list( set(moe.keys_batch) | set(moe.nested_keys_batch) | set(moe.nested_keys_supbatch) | set(moe.nested_keys_order) )
        #moe.data_idx_batch = moe.transform_df(meta)
        
        return moe
    
    @classmethod
    def from_adatas(self, adatas, keys_batch, nested={}):
        """Create the encoder from the ``adata.uns`` entries of a list of samples.

        Parameters
        ----------
        adatas : list of anndata.AnnData
            Samples; every ``adata.uns`` must contain the keys of `keys_batch` and `nested`.
        keys_batch : list of str
            Batch keys to encode.
        nested : dict of str to list of str, default {}
            Nested batch keys, see `from_data`.

        Returns
        -------
        MultiOrdinalEncoder
        """
        use_keys = list(set(keys_batch) | set(nested.keys()) | set([e for k,v in nested.items() for e in v]))
        meta = pd.DataFrame({k:[ad.uns[k] for ad in adatas] for k in use_keys})
        return MultiOrdinalEncoder.from_data(meta, keys_batch, nested=nested)
    
    
    @classmethod
    def from_exported(cls, dct):
        """Restore an encoder from the output of `export`.

        Parameters
        ----------
        dct : dict
            Output of `export`.

        Returns
        -------
        MultiOrdinalEncoder
        """
        moe = cls()
        moe.keys_batch = dct["keys_batch"]
        moe.oe_batches = {key:CustomOrdinalEncoder.from_old(dct["labels"][key], dct["allow_unknown"][key]) for key in moe.keys_batch}
        moe.Nbatch = dct["Nbatch"]
        moe.labels = dct["labels"]
        
        moe.nested = dct["nested"]
        moe.nested_keys_batch    = [k    for k, v in moe.nested.items()]
        moe.nested_keys_supbatch = [v[0] for k, v in moe.nested.items()]
        moe.nested_keys_order    = [v[1] for k, v in moe.nested.items()]
        moe.nested_data = pdsave_collect_dict(dct["nested_data"])
        for k, df in moe.nested_data.items():
            df["minor_idx"] = df["minor_idx"].astype(int)
        
        moe.necessary_keys = list( set(moe.keys_batch) | set(moe.nested_keys_batch) | set(moe.nested_keys_supbatch) | set(moe.nested_keys_order) )
        
        return moe
    
    def export(self):
        """Export the encoder to a dict (labels, nesting), for saving with a model.

        Returns
        -------
        dict
            Restore with `from_exported`.
        """
        dct = {"keys_batch":self.keys_batch,
               "Nbatch":self.Nbatch,
               "labels":self.labels,
               "allow_unknown":{k:self.oe_batches[k].allow_unknown for k in self.keys()},
               "nested":self.nested,
               "nested_data":pdsave_expand_dict(self.nested_data),
               }
        return dct
    
    
    def extend(self, meta):
        """Add all labels of `meta` that are not known yet; known labels keep their indices.

        New labels get the next free indices (sorted as str). Use the result to extend a transformer
        (``transformer.extend(moe, new_ords)``).

        Parameters
        ----------
        meta : pandas.DataFrame
            One row per sample, with a column for every batch key and for the keys used in `nested`.

        Returns
        -------
        dict of str to list of int
            Newly added indices per batch key (empty lists for keys without new labels).
        """
        new_ords = {k:oe.fit_extend(meta[k]) for k, oe in self.oe_batches.items()}
        self.Nbatch = {k:len(oe) for k, oe in self.oe_batches.items()}
        self.labels = {k:oe.labels for k, oe in self.oe_batches.items()}
        
        nested_data = {k:meta[[v[0], k, v[1]]].sort_values([v[0], v[1]]) for k, v in self.nested.items()}
        for k, df in nested_data.items():
            df.columns = ["major","minor","order"]
            df["minor_idx"] = self.oe_batches[k].transform(df["minor"])
            self.nested_data[k] = pd.concat([self.nested_data[k], df]).sort_values(["major", "order"])
            self.nested_data[k] = self.nested_data[k][~self.nested_data[k].duplicated()].reset_index(drop=True)
        
        return new_ords
    
    
    def keys(self):
        """Batch keys, as in a dict."""
        return self.oe_batches.keys()
    
    def items(self):
        """Pairs of batch key and its `cytovanni.utils.CustomOrdinalEncoder`."""
        return self.oe_batches.items()
    
    def __getitem__(self, key):
        """`cytovanni.utils.CustomOrdinalEncoder` of batch key `key` (``len()`` gives its number of batches)."""
        return self.oe_batches[key]
    
    
    def transform_df(self, df, astorch=True):
        """Encode the batch labels of a table into batch indices, imputing unknown nested batches.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per sample, with columns for all of `necessary_keys`.
        astorch : bool, default True
            Return torch tensors (dtype long) instead of numpy arrays.

        Returns
        -------
        dict of str to torch.Tensor
            Batch indices of shape (rows,) per batch key. For every nested key ``k`` also ``k + "_imputed"`` (bool
            torch.Tensor, always torch): True where the label of ``k`` was unknown and imputed from the nearest known
            batch in the same major batch (see the class description).

        Raises
        ------
        ValueError
            If a row with an unknown nested label has a major batch without any known nested batch to impute from,
            or (from the encoder) a label of a non-nested key is unknown.
        """
        outdct = {}
        fct_torch = (lambda x: torch.tensor(x, dtype=torch.long)) if astorch else (lambda x: x)
        for k, oe in self.oe_batches.items():
            if k in self.nested_keys_batch:
                trfd = oe.transform(df[k])
                impmask = trfd==-1
                if impmask.sum()>0:
                    refdata = self.nested_data[k]
                    curdata = df[[self.nested[k][0], k, self.nested[k][1]]].copy()
                    curdata.columns = ["major", "minor", "order"]
                    mask_major = refdata["major"].to_numpy()[None]==curdata["major"].to_numpy()[:,None]
                    diff = np.abs(refdata["order"].to_numpy()[None]-curdata["order"].to_numpy()[:,None]).astype(float)
                    diff[~mask_major] = np.inf
                    no_ref = impmask & ~mask_major.any(axis=1)
                    if no_ref.any():
                        unknown = curdata["major"].to_numpy()[no_ref]
                        raise ValueError(f"Cannot impute nested batch '{k}': major batch values {sorted(set(unknown.tolist()), key=str)} "
                                         f"(key '{self.nested[k][0]}') of rows with an unknown '{k}' have no known '{k}' to impute from.")
                    # impute only the unknown rows, known rows keep their own label
                    imputed = np.where(impmask, refdata["minor_idx"].to_numpy()[np.argmin(diff, axis=1)], trfd)
                    outdct[k] = fct_torch(imputed)
                else:
                    outdct[k] = fct_torch(trfd)
                outdct[k+"_imputed"] = torch.tensor(trfd==-1)
            else:
                outdct[k] = fct_torch(oe.transform(df[k]))
        
        return outdct
    
    def extract_ad(self, ad):
        """Return the values of `necessary_keys` from ``ad.uns``.

        All keys needed for a possible imputation must be present, even if every batch of `ad` is known.

        Parameters
        ----------
        ad : anndata.AnnData
            One sample.

        Returns
        -------
        dict of str to object
        """
        return {k:ad.uns[k] for k in self.necessary_keys}
    
    def transform_dict(self, dct):
        """Encode one sample's batch labels, given as a dict, like a one-row `transform_df`.

        Parameters
        ----------
        dct : dict of str to object
            One label per key of `necessary_keys`.

        Returns
        -------
        dict of str to torch.Tensor
            Batch indices of shape (1,) per batch key, plus the ``_imputed`` flags of nested keys.
        """
        return self.transform_df(pd.DataFrame(dct, index=[0]))
    
    def transform_dict_simple(self, dct):
        """Encode labels per batch key on their own, without imputation; used to build ``anchor_bidx``.

        Parameters
        ----------
        dct : dict of str to label or list of labels
            Labels per batch key, e.g. ``{"batch": ["18-APR-2024"]}``. Keys that are not batch keys are dropped.

        Returns
        -------
        dict of str to numpy.ndarray of int (or int for a single label)
            Batch indices per batch key, e.g. as ``anchor_bidx`` of the transformers and `PanelEmbeddingModule`.
        """
        return {k:self.oe_batches[k].transform(v) for k,v in dct.items() if k in self.oe_batches}
    
    def transform_ad(self, ad):
        """Encode the batch labels in ``ad.uns`` into batch indices (`extract_ad` then `transform_dict`).

        All keys needed for a possible imputation must be in ``ad.uns``, even if every batch of `ad` is known.

        Parameters
        ----------
        ad : anndata.AnnData
            One sample.

        Returns
        -------
        dict of str to torch.Tensor
            Batch indices of shape (1,) per batch key, plus the ``_imputed`` flags of nested keys.
        """
        return self.transform_dict(self.extract_ad(ad))
    
    
class OLD_MultiOrdinalEncoder():
    """Unused leftover methods of an earlier encoder (not functional on their own); use `MultiOrdinalEncoder`."""
    def transform_dict_tobidx(self, dct, x=None):
        """Encode the labels in `dct`, optionally expanded to the events of `x` (unused)."""
        fct = (lambda bidx: bidx) if x is None else (lambda bidx: bidx.expand(x.shape[0]).to(x.device))
        return {k:fct(v) for k, v in self.transform_df(pd.DataFrame({k:[v] for k, v in dct.items()})).items()}
    
    def transform_df_naive(self, df, astorch=True):
        """Encode the table `df` without nested imputation, as torch tensors if `astorch` (unused)."""
        fct_torch = (lambda x: torch.tensor(x, dtype=torch.long)) if astorch else (lambda x: x)
        return {k:fct_torch(oe.transform(df[k])) for k, oe in self.oe_batches.items()}
    
    def get_batches_mapping(self, k1, k2):
        """Map the batches of `k1` to those of `k2`; only accurate if `k1` is nested in `k2` (unused)."""
        v1, v2 = self.data_idx_batch[k1], self.data_idx_batch[k2]
        cons = torch.zeros((v1.max()+1, v2.max()+1))
        cons[v1, v2] = 1
        mapping = torch.argmax(cons, axis=1).numpy()
        assert (mapping==-1).sum()==0
        return mapping
    
    def extract_ad(self, ad):
        """Return the batch and order keys from ``ad.uns`` (unused)."""
        return {k:ad.uns[k] for k in self.keys()} | {k:ad.uns[k] for k in self.nested_keys_order}
