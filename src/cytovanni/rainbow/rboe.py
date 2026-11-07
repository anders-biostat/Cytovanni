"""Encoder of the rainbow bead metadata (cytometer, batch, bead type, measurement) shared by the calibration modules."""
import numpy as np
import pandas as pd
import warnings

from ..exceptions import AutoBatchException, InconsistentBatchException, RepeatedBatchException, OrdinalEncoderWarning, OrdinalEncoderException, AutoBatchWarning, CalibrationModuleException
from ..utils import CustomOrdinalEncoder

class RainbowBatchOrdinalEncoder():
    """Ordinal encoder of the rainbow bead metadata, and assignment of samples to rainbow batches.

    Holds one `CustomOrdinalEncoder` per key of `encode_keys` ('cytometer', 'rainbow_batch', 'rainbow_type', 'uid')
    and the metadata table of all training measurements. Ensures that every batch has one settings hash and lives on
    one cytometer. Unknown labels encode to -1. Construct it with `from_data` or `from_exported`.

    Attributes
    ----------
    oes : dict of str to CustomOrdinalEncoder
        Encoder per key; ``oes[key].labels`` are the known labels in ordinal order.
    data_df : pandas.DataFrame
        Metadata of all training measurements (encode_keys, 'set_hash', 'UTC' (float, seconds), 'date').
    WARN_UTC_DIFFERENCE : float
        Time difference (seconds, 2 h) to the closest rainbow measurement above which `find_closest_batch` warns.
    """
    encode_keys = ["cytometer", "rainbow_batch", "rainbow_type", "uid"]
    WARN_UTC_DIFFERENCE = 2*60*60
    
    def __init__(self):
        pass
    
    def _process_df(self, df):
        """Copy of the encode_keys and 'set_hash' columns of df, with 'UTC' (float) and 'date' (NaN if absent)."""
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
        """Raise InconsistentBatchException/RepeatedBatchException unless each batch has one hash and one cytometer."""
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
        """Cast data_df['UTC'] to float."""
        self.data_df["UTC"] = self.data_df["UTC"].astype(float)
        #self.data_df["set_hash"] = self.data_df["set_hash"].astype(str)
    
    @classmethod
    def from_data(cls, df):
        """Build the encoder from the metadata of the training measurements.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per rainbow bead measurement, with the columns 'cytometer' (name of the cytometer),
            'rainbow_batch' (batch name), 'rainbow_type' (bead type, e.g. lot), 'uid' (unique measurement id) and
            'set_hash' (hash of the cytometer settings, e.g. voltages); optional 'UTC' (seconds) and 'date', used
            by `find_closest_batch`.

        Returns
        -------
        RainbowBatchOrdinalEncoder

        Raises
        ------
        InconsistentBatchException
            If a batch contains measurements with different settings hashes.
        RepeatedBatchException
            If a batch name occurs on several cytometers.
        """
        rboe = cls()
        rboe.data_df = rboe._process_df(df)
        rboe._set_data_df_types()
        rboe._ensure_batch_consistency()
        rboe.oes = {k:CustomOrdinalEncoder.from_labels(rboe.data_df[k], allow_unknown=True) for k in rboe.encode_keys}
        return rboe
    
    def export(self):
        """Export the labels of all encoders and the metadata table as a dict for `from_exported`."""
        exported = {}
        exported["labels"] = {k:v.labels for k, v in self.oes.items()}
        exported["data_df"] = {"columns":self.data_df.columns.to_numpy(), "data":self.data_df.to_numpy()}
        return exported
    
    @classmethod
    def from_exported(cls, exported):
        """Rebuild an encoder from the output of `export` (label order is preserved).

        Parameters
        ----------
        exported : dict
            Output of `export`.

        Returns
        -------
        RainbowBatchOrdinalEncoder
        """
        rboe = cls()
        rboe.data_df = pd.DataFrame(exported["data_df"]["data"], columns=exported["data_df"]["columns"])
        rboe._ensure_batch_consistency()
        rboe._set_data_df_types()
        rboe.oes = {k:CustomOrdinalEncoder.from_old(exported["labels"][k], allow_unknown=True) for k in rboe.encode_keys}
        return rboe
    
    def check_orphan_types(self, df):
        """Check that every new rainbow type in df is linked to the known data; call before `extend`.

        A new rainbow type can only be calibrated consistently with the known types if it was measured in a known
        batch, or in a new batch in which a known type was measured as well (in `df`). Otherwise the parameters of
        the new type and its batches are degenerate (scatter and fluorescence alike) and the calibration factors of
        its batches are arbitrary.

        Parameters
        ----------
        df : pandas.DataFrame
            New metadata (columns 'rainbow_type', 'rainbow_batch'), as passed to `extend`.

        Returns
        -------
        None

        Raises
        ------
        CalibrationModuleException
            If a new type has no such link.
        """
        initial_types, initial_batches = self.oes["rainbow_type"].labels, self.oes["rainbow_batch"].labels
        new_orphan_types = []
        for tp in set(df["rainbow_type"].unique())-set(initial_types):
            batches = df.loc[df["rainbow_type"]==tp, "rainbow_batch"].unique()
            # overlap with a trained batch, or a trained type measured in one of the new type's batches
            if len(set(batches) & set(initial_batches))==0:
                otheroverlap = (np.isin(df["rainbow_type"], initial_types) & np.isin(df["rainbow_batch"], batches)).sum()
                if otheroverlap==0:
                    new_orphan_types.append(tp)
        if len(new_orphan_types)>0:
            raise CalibrationModuleException(f"Newly added rainbow types {sorted(new_orphan_types)} have no overlap with previously trained on data! Final calibration for these types will not be consistent with the others, aborting.")

    def extend(self, df):
        """Add the measurements of df to the metadata and their unseen labels to the encoders.

        Known labels keep their ordinals; new ones are appended.

        Parameters
        ----------
        df : pandas.DataFrame
            Metadata of the new measurements, as for `from_data`.

        Returns
        -------
        dict of str to list of int
            Newly added ordinals per key of `encode_keys`.

        Raises
        ------
        InconsistentBatchException, RepeatedBatchException
            As in `from_data`, for the combined metadata (the metadata table is already extended when this is raised).
        """
        self._ensure_batch_consistency()
        self.data_df = pd.concat([self.data_df, self._process_df(df)])
        self._set_data_df_types()
        self._ensure_batch_consistency(" after extending the data")
        new_ords = {k:oe.fit_extend(self.data_df[k]) for k, oe in self.oes.items()}
        return new_ords
    
    def find_closest_batch(self, adata, fct_hash_settings=None, max_hour_difference_across_days=4):
        """Find the rainbow batch of a sample from its cytometer, measurement time and settings.

        Candidates are the training measurements on the same cytometer, from the same date or less than
        `max_hour_difference_across_days` hours apart, with the same settings hash as the sample. The batch of the
        candidate closest in time (UTC) is returned.

        Parameters
        ----------
        adata : anndata.AnnData
            Sample with 'cytometer', 'date' and 'UTC' (seconds) in ``uns``.
        fct_hash_settings : callable, optional
            Function ``fct_hash_settings(adata)`` giving the settings hash; it must be the one used for the training
            data. Default: ``adata.uns['set_hash']``.
        max_hour_difference_across_days : float, default 4
            Also consider measurements from other dates up to this many hours apart (e.g. a sample measured shortly
            after midnight).

        Returns
        -------
        str
            Batch name.

        Raises
        ------
        AutoBatchException
            If 'date', 'UTC' or 'cytometer' is missing, there is no settings hash, no candidate is found (none in the
            time window or none with the same settings), or the UTC of the sample or of all candidates is NaN.

        Warns
        -----
        AutoBatchWarning
            If the closest rainbow measurement is more than `WARN_UTC_DIFFERENCE` (2 h) away.
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
        """Ordinal of the rainbow batch of a sample.

        Tries in this order: `rainbow_batch`, ``adata.uns['rainbow_batch']``, and `find_closest_batch` (date, time,
        settings); a given but unknown batch falls through to the next with an `OrdinalEncoderWarning`.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Sample.
        rainbow_batch : str, optional
            Batch name.
        fct_hash_settings : callable, optional
            Settings hash function for `find_closest_batch`. Default: ``adata.uns['cytoconfig'].fct_hash_settings``.

        Returns
        -------
        int
            Batch ordinal.

        Raises
        ------
        OrdinalEncoderException
            If neither `adata` nor `rainbow_batch` is given, or `rainbow_batch` is unknown and there is no `adata`.
        AutoBatchException
            If the batch has to be inferred and `find_closest_batch` fails.
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
        """Ordinals of the encode_keys of one AnnData.

        Parameters
        ----------
        adata : anndata.AnnData
            Measurement with the encode_keys in ``uns``.

        Returns
        -------
        dict of str to int
            Ordinal per key; -1 if the label is unknown or the key is missing in ``adata.uns``.
        """
        return {k:(self.oes[k].transform(adata.uns[k]) if k in adata.uns else -1) for k in self.oes}
