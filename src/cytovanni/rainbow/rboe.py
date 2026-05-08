import numpy as np
import pandas as pd
import warnings

from ..exceptions import AutoBatchException, InconsistentBatchException, RepeatedBatchException, OrdinalEncoderWarning, OrdinalEncoderException, AutoBatchWarning
from ..utils import CustomOrdinalEncoder

class RainbowBatchOrdinalEncoder():
    """ Ordinal encoder for the rainbow bead sample metadata.
    """
    encode_keys = ["cytometer", "rainbow_batch", "rainbow_type", "uid"]
    WARN_UTC_DIFFERENCE = 2*60*60
    
    def __init__(self):
        pass
    
    def _process_df(self, df):
        """ Process df. Keep encode_keys, try adding UTC and date.
        """
        data_df = df[self.encode_keys].copy()
        data_df["set_hash"] = df["set_hash"]
        # Try adding UTC
        if "UTC" in df.columns: data_df["UTC"] = df["UTC"]
        else:                   data_df["UTC"] = np.nan
        data_df["UTC"] = data_df["UTC"].astype(float)
        # Try adding date
        if "date" in df.columns: data_df["date"] = df["date"]
        else:                    data_df["date"] = np.nan
        
        return data_df
    
    def _ensure_batch_consistency(self, add=""):
        """ Ensure that all samples within a batch have the same settings, and that batch names are only present on one cytometer each.
        """
        for cyt in np.unique(self.data_df["cytometer"]):
            for batch in np.unique(self.data_df["rainbow_batch"][self.data_df["cytometer"]==cyt]):
                df = self.data_df.loc[(self.data_df["rainbow_batch"]==batch) & (self.data_df["cytometer"]==cyt)]
                if np.unique(df["set_hash"]).shape[0]>1:
                    raise InconsistentBatchException(f"Batch {batch} on cytometer {cyt} contains multiple different settings{add}!")
        
        for batch in np.unique(self.data_df["rainbow_batch"]):
            cytos = self.data_df.loc[self.data_df["rainbow_batch"]==batch, "cytometer"].unique().tolist()
            if len(cytos)>1:
                raise RepeatedBatchException(f"Found rainbow batch {batch} on multiple cytometers ({cytos}){add}, make sure batch names are not shared across cytometers!")
    
    def _set_data_df_types(self):
        """ Ensure proper types for some of the metadata.
        """
        self.data_df["UTC"] = self.data_df["UTC"].astype(float)
        #self.data_df["set_hash"] = self.data_df["set_hash"].astype(str)
    
    @classmethod
    def from_data(cls, df):
        """ Initialise from data.
            Requires:
                'cytometer': name of the cytometer
                'rainbow_batch': batch name
                'rainbow_type': type of the rainbow bead (lot etc.)
                'uid': rainbow bead sample unique identifier
                'set_hash': hash of relevant cytometer settings (voltages etc.) to ensure no changes in batches
            to be present in df.
            If available also uses 'UTC' and 'date'.
            
            :param df: DataFrame. Dataframe containing the data to initialise with.
        """
        rboe = cls()
        rboe.data_df = rboe._process_df(df)
        rboe._set_data_df_types()
        rboe._ensure_batch_consistency()
        rboe.oes = {k:CustomOrdinalEncoder.from_labels(rboe.data_df[k], allow_unknown=True) for k in rboe.encode_keys}
        return rboe
    
    def export(self):
        """ Export self to dictionary.
        """
        exported = {}
        exported["labels"] = {k:v.labels for k, v in self.oes.items()}
        exported["data_df"] = {"columns":self.data_df.columns.to_numpy(), "data":self.data_df.to_numpy()}
        return exported
    
    @classmethod
    def from_exported(cls, exported):
        """ Initialises from ouput of .export().
        """
        rboe = cls()
        rboe.data_df = pd.DataFrame(exported["data_df"]["data"], columns=exported["data_df"]["columns"])
        rboe._ensure_batch_consistency()
        rboe._set_data_df_types()
        rboe.oes = {k:CustomOrdinalEncoder.from_old(exported["labels"][k], allow_unknown=True) for k in rboe.encode_keys}
        return rboe
    
    def extend(self, df):
        """ Same as from_data, but extend by all unseen labels present in df.
            Returns a dict with all newly added ordinals.
        """
        self._ensure_batch_consistency()
        self.data_df = pd.concat([self.data_df, self._process_df(df)])
        self._set_data_df_types()
        self._ensure_batch_consistency(" after extending the data")
        new_ords = {k:oe.fit_extend(self.data_df[k]) for k, oe in self.oes.items()}
        return new_ords
    
    def find_closest_batch(self, adata, fct_hash_settings=None, max_hour_difference_across_days=4):
        """ Attempt to find the closest rainbow batch for adata, using 'cytometer', 'date' and 'UTC' from .uns.
            Requires using the same settings hash function fct_hash_settings as for the training data.
            Throws an error if unsuccessful.
            
            Would also be possible to relax this assumption, but probably better this way?
            
            :param fct_hash_settings: function, None. Function to get settings hash, if None expects 'set_hash' in adata.uns.
        """
        if not ("date" in adata.uns and "UTC" in adata.uns and "cytometer" in adata.uns):
            raise AutoBatchException("Automatically inferring rainbow batch requires 'date', 'UTC', and 'cytometer' to be present in adata.uns!")
        date, utc, cytometer = adata.uns["date"], adata.uns["UTC"], adata.uns["cytometer"]
        
        if fct_hash_settings is None and not ("set_hash" in adata.uns):
            raise AutoBatchException("Automatically inferring rainbow batch requires either fct_hash_settings or 'set_hash' in adata.uns!")
        set_hash = fct_hash_settings(adata) if fct_hash_settings is not None else adata.uns["set_hash"]
        
        possible = self.data_df[(
                                    (self.data_df["date"]==date) | ((self.data_df["UTC"]-utc).abs()<max_hour_difference_across_days*60*60)
                                ) & (self.data_df["cytometer"]==cytometer)]
        if possible.shape[0]==0:
            raise AutoBatchException(f"Could not infer rainbow batch automatically, date {date} on cytometer {cytometer} was not present in training data!")
        
        possible = possible[possible["set_hash"]==set_hash]
        if possible.shape[0]==0:
            raise AutoBatchException(f"Could not infer rainbow batch automatically, date {date} has no batches with same settings!")
        
        if not np.any(~np.isnan(possible["UTC"].to_numpy())):
            raise AutoBatchException(f"Could not infer rainbow batch automatically, training data for date {date} has no UTC!")
        if np.isnan(utc):
            raise AutoBatchException(f"Could not infer rainbow batch automatically, UTC {utc} is not valid!")
        utcdiff = np.abs(possible["UTC"] - utc)
        
        batch = possible["rainbow_batch"].iloc[np.argmin(utcdiff)]
        
        if np.min(utcdiff)>self.WARN_UTC_DIFFERENCE:
            warnstr = f"Sample that was measured at {utc} (UTC) on {date} has more than {self.WARN_UTC_DIFFERENCE/60/60:.2f} hours difference to the closest rainbow beads!"
            warnstr += " Ideally, rainbow beads should be measured at least every hour or so."
            warnings.warn(warnstr, AutoBatchWarning)
        
        return batch
    
    def transform_adata_batch(self, adata=None, rainbow_batch=None, fct_hash_settings=None):
        """ Encode rainbow batch of adata.
            If present, prefers explicit rainbow_batch, if not tries adata.uns['rainbow_batch'],
            finally tries to infer batch from training data and current UTC etc. with .find_closest_batch.
        """
        if (adata is None) and (rainbow_batch is None):
            raise OrdinalEncoderException(f"Getting scatter calibration factor requires at least either adata or explicit rainbow_batch!")
        
        # First try explicit batch
        if rainbow_batch is not None:
            batchidx = self.oes["rainbow_batch"].transform(rainbow_batch)
            if batchidx!=-1:
                return batchidx
            if adata is None:
                raise OrdinalEncoderException(f"Did not find provided rainbow batch '{rainbow_batch}' and no adata was given!")
            else:
                warnings.warn(f"Did not find provided rainbow batch '{rainbow_batch}'! Trying to get it from adata instead.", OrdinalEncoderWarning)
        
        # Then try adata.uns
        if "rainbow_batch" in adata.uns:
            batchidx = self.oes["rainbow_batch"].transform(adata.uns["rainbow_batch"])
            if batchidx!=-1:
                return batchidx
            warnings.warn(f"Did not find rainbow batch '{adata.uns['rainbow_batch']}' from adata.uns! Trying to automatically infer it instead.", OrdinalEncoderWarning)
        
        # Then try automatically inferring it
        rainbow_batch = self.find_closest_batch(adata,
                                fct_hash_settings=fct_hash_settings if fct_hash_settings is not None else adata.uns["cytoconfig"].fct_hash_settings)
        batchidx = self.oes["rainbow_batch"].transform(rainbow_batch)
        return batchidx
    
    def transform_adata(self, adata):
        """ Encode single adata, defaults to -1 if not present in ordinal encoders, or not present in adata.uns
        """
        return {k:(self.oes[k].transform(adata.uns[k]) if k in adata.uns else -1) for k in self.oes}
