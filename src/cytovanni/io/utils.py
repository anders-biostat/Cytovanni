import numpy as np
import pandas as pd
import flowkit as fk
import flowio
import flowutils
import os
import anndata
import dateutil.parser
import warnings
import hashlib

from ..exceptions import ChannelMismatchException, RainbowSpilloverWarning, ChannelMismatchWarning

def _access_lowercase_df(df, keys):
    cols = {c.lower(): c for c in df.columns}
    return df[[cols[k] for k in keys]]
def _access_lowercase_df_single(df, key):
    cols = {c.lower(): c for c in df.columns}
    return df[cols[key]]
def _access_lowercase_dict(dct, key):
    cols = {c.lower(): c for c in dct.keys()}
    return dct[cols[key]]

def readfcs_available_channels(filepath, simple_flowio=True):
    """ Returns all available channels in .fcs file at filepath.
        
        :param filepath: str. Path to .fcs file.
        
        :param simple_flowio: bool. Whether to use flowio instead of flowkit, maybe a bit more likely to break through an update of flowio at some point than the flowkit approach.
    """
    if simple_flowio:
        data = flowio.FlowData(filepath, only_text=True)
        channels = np.asarray([_access_lowercase_dict(data.channels[k], "pnn") for k in data.channels])
        return channels
    else:
        sample = fk.Sample(filepath)
        sample_df = sample.as_dataframe(source="raw")
        return np.asarray(sample_df.columns.get_level_values(0))

def readfcs_channel_settings_hash(filepath, cytoconfig):
    """ Custom function to read only channel settings hash, to avoid having to read all the data only to get the settings and nothing else.
    """
    data = flowio.FlowData(filepath, only_text=True)
    
    var = pd.DataFrame([[k for k in data.channels],
                        [_access_lowercase_dict(data.channels[k], "pnn") for k in data.channels]],
                       index=["channel_number", "pnn"]
                      ).T
    var["pnn"] = var["pnn"].apply(lambda channel: _replace_channel_name(channel, cytoconfig))
    var.index = var["pnn"]

    load_voltage = lambda i: float(data.text[f"p{i}v"]) if f"p{i}v" in data.text else np.nan
    var["pnv"] = var["channel_number"].apply(load_voltage)

    load_gain = lambda i: float(data.text[f"p{i}g"]) if f"p{i}g" in data.text else np.nan
    var["png"] = var["channel_number"].apply(load_gain)

    adata = anndata.AnnData(var=var)
    adata.uns["cytoconfig"] = cytoconfig
    
    return cytoconfig.get_settings_hash(adata)

def sample_has_nonzero_spill(sample, manual=False):
    """ Returns False if sample either doesn't contain a spillover matrix, or if its entries are only 0 or 1, True otherwise.
        Tests $SPILL and $SPILLOVER.
        My implementation if manual, else wraps flowutils.
    """
    if "spill" in sample.metadata:
        spillstr = sample.metadata["spill"]
    elif "spillover" in sample.metadata:
        spillstr = sample.metadata["spillover"]
    else:
        return False
    
    if manual: # my simple implementation
        spill = spillstr.split(",")
        spillentry = np.asarray(spill[int(spill[0])+1:], dtype=float)
    else: # uses flowutils
        spillentry = flowutils.compensate.get_spill(spillstr)[0]

    return not np.all(np.isclose(spillentry, 0.) | np.isclose(spillentry, 1.))

def readfcs_spill(filepath):
    """ Reads spillover matrix from either $SPILL or $SPILLOVER.
        Returns pd.DataFrame of shape (stain, channel)

        Currently only works properly for conventional flow with a square matrix,
        deal with spectral somewhere else!
    """
    sample = flowio.FlowData(filepath, only_text=True)
    
    if "spill" in sample.text:       spillstr = sample.text["spill"]
    elif "spillover" in sample.text: spillstr = sample.text["spillover"]
    else:                            return None
    
    spillentry = flowutils.compensate.get_spill(spillstr)
    spill = pd.DataFrame(spillentry[0], columns=spillentry[1])

    return spill

def readfcs_metadata(filepath):
    """ Reads metadata of .fcs as dict
    """
    sample = flowio.FlowData(filepath, only_text=True)
    return sample.text

def _spectra_from_autospectralstr(s):
    """
    Reconstruct a pandas DataFrame from a flattened CSV-style string
    that encodes:
        n_rows, n_cols, [row_names...], [col_names...], [flattened_values...]
    """
    # Split and clean
    tokens = [x.strip() for x in s.strip().split(',') if x.strip()]
    
    # Extract metadata
    n_rows = int(tokens[0])
    n_cols = int(tokens[1])
    
    # Extract row and column labels
    row_names = tokens[2:2 + n_rows]
    col_names = tokens[2 + n_rows : 2 + n_rows + n_cols]
    
    # Extract numeric values
    values = np.array(tokens[2 + n_rows + n_cols:], dtype=float)
    if values.size != n_rows * n_cols:
        raise ValueError(f"Data size mismatch: expected {n_rows*n_cols}, got {values.size}")
    
    # Reshape and construct DataFrame
    df = pd.DataFrame(values.reshape((n_rows, n_cols)), index=row_names, columns=col_names)
    return df

def readfcs_autospectral_spill(filepath):
    """ Reads spectra matrix from a .fcs file as written by AutoSpectral
    """
    sample = flowio.FlowData(filepath, only_text=True)
    
    if "spectra" in sample.text:
        return _spectra_from_autospectralstr(sample.text["spectra"])
    else:
        return None


def _replace_channel_name(channel, cytoconfig):
    return cytoconfig.replace_channelname_dict[channel] if channel in cytoconfig.replace_channelname_dict else channel

def readfcs_sample(filepath, cytoconfig, metadct={}, gates=[], print_gates=False, warn_spillover=False, warn_spillover_message="Non-trivial spillover matrix!", fct_fix_df=lambda df: df, allow_missing_scatter=True):
    """ Read a single sample from an .fcs file into an AnnData object.
        Based on FlowKit so we don't need to maintain reading code.
        
        :param filepath: str. Path to .fcs with sample.
        
        :param cytoconfig: CytometerConfiguration. Cytometer configuration for sample.
        
        :param metadct: dict. Optional additional metadata to add to .uns.
        
        :param gates: list. Optional, gates that should be immediately applied.

        :param print_gates: bool. If True, print how many events are excluded by the gates.
        
        :param warn_spillover: bool. If True, throws a warning if sample contains a non-trivial spillover matrix.
        
        :param fct_fix_df: function.

        :param allow_missing_scatter: bool. If True, only warn if scatter channels are missing instead of throwing an error.
    """
    sample = fk.Sample(filepath, preprocess=False)
    sample.channels["pnn"] = _access_lowercase_df_single(sample.channels, "pnn").apply(lambda channel: _replace_channel_name(channel, cytoconfig))
    
    if warn_spillover and sample_has_nonzero_spill(sample):
        warnings.warn(warn_spillover_message, RainbowSpilloverWarning)
    
    # FlowKit doesn't read the voltages, do that manually
    load_voltage = lambda i: float(sample.metadata[f"p{i}v"]) if f"p{i}v" in sample.metadata else np.nan
    sample.channels["pnv"] = sample.channels["channel_number"].apply(load_voltage)
    
    # Transfer data
    sample_df = sample.as_dataframe(source="raw")
    sample_df.columns = sample_df.columns.get_level_values(0)
    sample_df.columns = [_replace_channel_name(channel, cytoconfig) for channel in sample_df.columns]
    
    # if necessary, apply bug fix etc.
    sample_df = fct_fix_df(sample_df)
    
    # test channels
    if set(sample_df.columns) | set(cytoconfig.channels_fluorescence) != set(sample_df.columns):
        total = list(sample_df.columns)
        missing = list(set(cytoconfig.channels_fluorescence) - set(sample_df.columns))
        raise ChannelMismatchException(f"Could not find fluorescence channels {missing} in file {filepath}!\nAvailable channels are: {total}")
    
    if set(sample_df.columns) | set(cytoconfig.channels_scatter) != set(sample_df.columns):
        total = list(sample_df.columns)
        missing = list(set(cytoconfig.channels_scatter) - set(sample_df.columns))
        if not allow_missing_scatter:
            raise ChannelMismatchException(f"Could not find scatter channels {missing} in file {filepath}!\nAvailable channels are: {total}")
        else:
            warnings.warn(f"Could not find scatter channels {missing} in file {filepath}!", ChannelMismatchWarning)
            channels_scatter_available = [c for c in cytoconfig.channels_scatter if (c in sample_df.columns)]
    else:
        channels_scatter_available = cytoconfig.channels_scatter
    
    channels_HW = [c for c in cytoconfig.channels_fluorescence_addHW if c in sample_df.columns]
    channels = channels_scatter_available + cytoconfig.channels_fluorescence + channels_HW
    
    ## var data, channel config
    var = _access_lowercase_df(sample.channels, ["pnn","pns","png","pnv","pnr"])
    var.index = var["pnn"].tolist()
    var = var.loc[channels]
    
    ## obs data, only channels_meta
    obs = sample_df[cytoconfig.channels_meta]
    obs.index = obs.index.astype(str)
    
    ## raw data
    X = sample_df[channels]
    
    ## AnnData
    adata = anndata.AnnData(obs=obs, var=var)
    adata.layers["raw"] = X
    
    # Add date and UTC of beginning of recording if available and nicely formatted
    if "date" in sample.metadata and "btim" in sample.metadata:
        try: # Try harmonizing date
            #adata.uns["date"] = str(dateutil.parser.parse(adata.uns["date"]).date()) # bug
            adata.uns["date"] = dateutil.parser.parse(sample.metadata["date"]).date().strftime('%d-%b-%Y').upper()
        except Exception:
            adata.uns["date"] = sample.metadata["date"]
        try: # Try getting UTC
            adata.uns["UTC"] = dateutil.parser.parse(sample.metadata["date"]+"-"+sample.metadata["btim"]).timestamp()
        except Exception:
            adata.uns["UTC"] = np.nan
    else:
        adata.uns["date"], adata.uns["UTC"] = None, np.nan
    
    adata.uns["cytometer"] = cytoconfig.cytometer
    uid = int(adata.uns["UTC"]) if (adata.uns["UTC"]==adata.uns["UTC"]) else hashlib.sha256(filepath.encode('utf-8')).hexdigest()
    adata.uns["uid"] = adata.uns["cytometer"] +"-"+ str(uid)
    
    adata.uns["cytoconfig"] = cytoconfig
    adata.uns["set_hash"] = cytoconfig.get_settings_hash(adata)
    adata.uns["Nevents"] = adata.shape[0]
    
    for k, v in metadct.items():
        adata.uns[k] = v
    
    if len(gates)>0:
        keepmask = np.all(np.vstack([gate.apply(adata) for gate in gates]), axis=0)
        adata = adata[keepmask].copy()
        if print_gates:
            outstr = f"Gating {os.path.basename(filepath)} - "
            outstr += f"dropped {(1-keepmask.mean())*100:.1f}% ({len(keepmask)-keepmask.sum()} / {len(keepmask)})"
            print(outstr)
    
    return adata


def read_fcs_timestamp(filepath):
    """ Read date and recording time from .fcs file if available.
    """
    data = flowio.FlowData(filepath, only_text=True)
    date = data.text["date"] if "date" in data.text else ""
    btim = data.text["btim"] if "btim" in data.text else np.nan
    etim = data.text["etim"] if "etim" in data.text else np.nan
    utc = dateutil.parser.parse(date+"-"+btim).timestamp() if (date and btim==btim) else np.nan
    return pd.Series([date, btim, etim, utc], index=["date", "begin_record", "end_record", "UTC"])

def read_folder_timestamps(folder, cytoconfig=None):
    """ Read date and recording time for all .fcs files in folder.
        If cytoconfig is given, also reads the settings hash.
    """
    files = list(filter(lambda x: x.endswith(".fcs"), os.listdir(folder)))
    tps = pd.DataFrame([read_fcs_timestamp(os.path.join(folder, file)) for file in files], index=files)
    if cytoconfig is not None:
        def safe_hash(file):
            try:
                return readfcs_channel_settings_hash(os.path.join(folder, file), cytoconfig)
            except Exception as e:
                return np.nan
        tps["set_hash"] = [safe_hash(file) for file in files]
    if tps["UTC"].isna().sum():
        return tps.sort_values(["date","begin_record"])
    else:
        return tps.sort_values(["UTC"])


def fix_A3_HW_bug(df):
    """ Fix weird bug of A3, to be used as fct_fix_df for readfcs_sample.
        If 'Time' is the first channel, the -H and -W channels are switched for some reason.
        Reverse this here.
    """
    if df.columns[0]=="Time":
        df["FSC-H"], df["FSC-W"] = df["FSC-W"], df["FSC-H"]
        df["SSC-H"], df["SSC-W"] = df["SSC-W"], df["SSC-H"]
    return df


def writefcs(filepath, data_df, sample_id="", add_metadata={}, overwrite_pns=True, pns=None):
    """ Export pd.DataFrame as .fcs file. Automatically creates the necessary folder structure.

        :param filepath: str. Path to newly created file, should end in .fcs.

        :param data_df: pd.DataFrame. The data to be written to the file.

        :param add_metadata: dict. Additional metadata keys to be added to the file.
    """
    from pathlib import Path
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
    
    sample = fk.Sample(data_df, sample_id=sample_id)
    sample.metadata.update(add_metadata)
    if overwrite_pns:
        sample.pns_labels = sample.pnn_labels if pns is None else pns
    sample.export(filepath, source="raw", include_metadata=True)

def fix_sample_bug(filepath, filepath_new, fct_bugfix, delete_spill=True):
    sample = fk.Sample(filepath, preprocess=False)
    data_df = sample.as_dataframe(source="raw")
    data_df.columns = data_df.columns.get_level_values(0)
    data_df = fct_bugfix(data_df)
    add_metadata = sample.metadata
    if delete_spill:
        if "spill" in add_metadata:
            del add_metadata["spill"]
    writefcs(filepath_new, data_df, sample_id="", add_metadata=add_metadata, overwrite_pns=False)

def readfcs_dataframe(filepath):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = fk.Sample(filepath, ignore_offset_error=True, preprocess=False)

    data = sample.as_dataframe(source="raw")
    cols = data.columns.get_level_values(1).to_numpy()
    mask = cols==""
    cols[mask] = data.columns.get_level_values(0)[mask]
    data.columns = cols

    return data

