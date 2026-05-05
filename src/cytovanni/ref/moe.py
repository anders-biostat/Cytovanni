import torch
import numpy as np
import pandas as pd

from ..utils import CustomOrdinalEncoder
from ..torch import pdsave_expand, pdsave_collect, pdsave_expand_dict, pdsave_collect_dict

class MultiOrdinalEncoder():
    """ Encodes multiple batches simultaneously.
        
        Allows nested batches for imputation. Example: nested={"a":["b","b_order"]}
        This assumes that batch "a" is a subset of "b", the numerical order within "b" is "b_order"
        Encoding a sample where the value for "b" is present, but the value for "a" was not part of the setup
        then returns the batch index for "a" that is closest to the true one according to "b_order", biased towards
        lower "b_order" if otherwise equidistant.
    """
    def __init__(self):
        pass
    
    @classmethod
    def from_data(cls, meta, keys_batch, nested={}):
        """ 
            
            :param meta: pd.DataFrame. Dataframe containing the data to initialise with.
            
            :param keys_batch: iterable. List of the batch keys that should be used.
            
            :param nested: dict. Dictionary containing information about nested batches, see example in class description.
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
        """ 
            
            :param adatas: iterable of AnnDatas. adatas to use, extracts all relevant data from their ad.uns.
            
            :param keys_batch: iterable. List of the batch keys that should be used.
            
            :param nested: dict. Dictionary containing information about nested batches, see example in class description.
        """
        use_keys = list(set(keys_batch) | set(nested.keys()) | set([e for k,v in nested.items() for e in v]))
        meta = pd.DataFrame({k:[ad.uns[k] for ad in adatas] for k in use_keys})
        return MultiOrdinalEncoder.from_data(meta, keys_batch, nested=nested)
    
    
    @classmethod
    def from_exported(cls, dct):
        """ Initialize from exported data.
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
        """ Export data.
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
        """ Extend by all unseen labels present in meta.
            Returns a dict with all newly added ordinals.
            
            :param meta: pd.DataFrame. Dataframe containing the data to extend with.
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
        return self.oe_batches.keys()
    
    def items(self):
        return self.oe_batches.items()
    
    def __getitem__(self, key):
        return self.oe_batches[key]
    
    
    def transform_df(self, df, astorch=True):
        """ Transform full dataframe, taking into account nested batches.
            Needs to contain all relevant batches etc. as columns.
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
                    curdata["minor_idx"] = refdata["minor_idx"].to_numpy()[np.argmin(diff, axis=1)]
                    outdct[k] = fct_torch(curdata["minor_idx"].to_numpy())
                else:
                    outdct[k] = fct_torch(trfd)
                outdct[k+"_imputed"] = torch.tensor(trfd==-1)
            else:
                outdct[k] = fct_torch(oe.transform(df[k]))
        
        return outdct
    
    def extract_ad(self, ad):
        """ Extract all relevant keys from adata.
            For simplicity, requires all keys for possible imputation to be present in ad.uns,
            even when all batches are present in the training data.
        """
        return {k:ad.uns[k] for k in self.necessary_keys}
    
    def transform_dict(self, dct):
        """ Transform dictionary to batch indices.
            Output is the same as for transform_df with a single row.
        """
        return self.transform_df(pd.DataFrame(dct, index=[0]))
    
    def transform_dict_simple(self, dct):
        """ Transform simple dictionary, no imputation etc., just every batch transformed on its own.
            Output only includes available batch keys, everything else in dct is discarded.
        """
        return {k:self.oe_batches[k].transform(v) for k,v in dct.items() if k in self.oe_batches}
    
    def transform_ad(self, ad):
        """ Transform adata to batch indices, using values from ad.uns.
            For simplicity, requires all keys for possible imputation to be present in ad.uns,
            even when all batches are present in the training data.
        """
        return self.transform_dict(self.extract_ad(ad))
    
    
class OLD_MultiOrdinalEncoder():
    def transform_dict_tobidx(self, dct, x=None):
        fct = (lambda bidx: bidx) if x is None else (lambda bidx: bidx.expand(x.shape[0]).to(x.device))
        return {k:fct(v) for k, v in self.transform_df(pd.DataFrame({k:[v] for k, v in dct.items()})).items()}
    
    def transform_df_naive(self, df, astorch=True):
        """ Transform full dataframe, naively without nested.
        """
        fct_torch = (lambda x: torch.tensor(x, dtype=torch.long)) if astorch else (lambda x: x)
        return {k:fct_torch(oe.transform(df[k])) for k, oe in self.oe_batches.items()}
    
    def get_batches_mapping(self, k1, k2):
        """ Only accurate if connections from k2 -> k1 is a tree!
        """
        v1, v2 = self.data_idx_batch[k1], self.data_idx_batch[k2]
        cons = torch.zeros((v1.max()+1, v2.max()+1))
        cons[v1, v2] = 1
        mapping = torch.argmax(cons, axis=1).numpy()
        assert (mapping==-1).sum()==0
        return mapping
    
    def extract_ad(self, ad):
        return {k:ad.uns[k] for k in self.keys()} | {k:ad.uns[k] for k in self.nested_keys_order}
