"""Reading and writing .fcs files: samples as AnnData, metadata, spillover matrices, time stamps."""
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
from zoneinfo import ZoneInfo

from ..utils.config import DEFAULT_TIMEZONE
from ..exceptions import ChannelMismatchException, RainbowSpilloverWarning, ChannelMismatchWarning

def _access_lowercase_df(df, keys):
    """Columns `keys` (lower case) of `df`, whatever the case of its column names."""
    cols = {c.lower(): c for c in df.columns}
    return df[[cols[k] for k in keys]]
def _access_lowercase_df_single(df, key):
    """Column `key` (lower case) of `df`, whatever the case of its column names."""
    cols = {c.lower(): c for c in df.columns}
    return df[cols[key]]
def _access_lowercase_dict(dct, key):
    """Entry `key` (lower case) of `dct`, whatever the case of its keys."""
    cols = {c.lower(): c for c in dct.keys()}
    return dct[cols[key]]

def readfcs_available_channels(filepath, simple_flowio=True):
    """List the channels ($PnN names) of an .fcs file.

    Channel names are returned as stored in the file, without the renaming of a cytometer configuration.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.
    simple_flowio : bool, default True
        Read only the text segment with flowio (fast). False reads the whole sample with flowkit (slower, but
        perhaps a bit less likely to break with a future flowio update).

    Returns
    -------
    numpy.ndarray
        Channel names ($PnN), in file order.
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
    """Settings hash of an .fcs file, from its text segment only (without reading the events).

    Builds a minimal AnnData (``var`` with 'pnn' after the renaming of `cytoconfig`, voltages 'pnv' and gains 'png')
    and passes it to ``cytoconfig.get_settings_hash``. Meant to give the same hash as ``uns['set_hash']`` of
    `readfcs_sample`; this requires a hash function that uses only these columns, such as the default.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Cytometer configuration (channel renaming and settings-hash function).

    Returns
    -------
    object
        The hash, as returned by the hash function of `cytoconfig`.

    Raises
    ------
    SettingsHashException
        If the hash function fails, e.g. because it needs other ``var`` columns.
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
    """Check whether a sample carries a non-trivial spillover matrix ($SPILL or $SPILLOVER).

    Parameters
    ----------
    sample : flowkit.Sample
        The loaded sample (not a path).
    manual : bool, default False
        Parse the matrix with a simple own parser instead of ``flowutils.compensate.get_spill``.

    Returns
    -------
    bool
        False if there is no spillover keyword or all its entries are 0 or 1 (identity-like), True otherwise.
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
    """Read the spillover matrix ($SPILL or $SPILLOVER) of an .fcs file.

    Only for conventional (compensation) matrices, which are square; spectral spillover matrices are read by
    `readfcs_autospectral_spill` or `read_spill_S8`.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.

    Returns
    -------
    pandas.DataFrame or None
        Matrix of shape (stain, channel): columns are the channel names listed in the keyword, the rows are in the
        same order but carry an unlabelled RangeIndex. None if the file has no spillover keyword.
    """
    sample = flowio.FlowData(filepath, only_text=True)
    
    if "spill" in sample.text:       spillstr = sample.text["spill"]
    elif "spillover" in sample.text: spillstr = sample.text["spillover"]
    else:                            return None
    
    spillentry = flowutils.compensate.get_spill(spillstr)
    spill = pd.DataFrame(spillentry[0], columns=spillentry[1])

    return spill

def readfcs_metadata(filepath):
    """Read the text segment (metadata keywords) of an .fcs file.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.

    Returns
    -------
    dict of str to str
        Keywords as parsed by flowio (lower case, without the leading '$'), e.g. 'date', 'btim', 'p1n'.
    """
    sample = flowio.FlowData(filepath, only_text=True)
    return sample.text

def _spectra_from_autospectralstr(s):
    """DataFrame from a flat string ``n_rows, n_cols, row names..., column names..., values (row by row)...``.

    Raises ValueError if the number of values is not ``n_rows * n_cols``.
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
    """Read the spectra matrix that AutoSpectral writes into an .fcs file (keyword 'spectra').

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.

    Returns
    -------
    pandas.DataFrame or None
        The matrix with the row and column names stored in the keyword. None if the file has no 'spectra' keyword.

    Raises
    ------
    ValueError
        If the number of values does not match the stated matrix size.
    """
    sample = flowio.FlowData(filepath, only_text=True)
    
    if "spectra" in sample.text:
        return _spectra_from_autospectralstr(sample.text["spectra"])
    else:
        return None


def _fcs_utc(date, btim, timezone):
    """Seconds since the epoch (true UTC) of the .fcs $DATE and $BTIM, which are local time of the instrument.

    They are interpreted in `timezone` (IANA name), independent of the time zone of the reading machine; a time zone
    given in the strings themselves takes precedence.
    """
    dt = dateutil.parser.parse(date+"-"+btim)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(timezone))
    return dt.timestamp()

def _replace_channel_name(channel, cytoconfig):
    """`channel` renamed with ``cytoconfig.replace_channelname_dict``, or unchanged if it is not a key there."""
    return cytoconfig.replace_channelname_dict[channel] if channel in cytoconfig.replace_channelname_dict else channel

def readfcs_sample(filepath, cytoconfig, metadct=None, gates=None, print_gates=False, warn_spillover=False, warn_spillover_message="Non-trivial spillover matrix!", fct_fix_df=lambda df: df, allow_missing_scatter=True):
    """Read a single sample from an .fcs file into an AnnData object.

    Based on flowkit, so that no own .fcs parsing code needs to be maintained. Channels are renamed with
    ``cytoconfig.replace_channelname_dict`` right after reading.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Cytometer configuration: channels to load, renaming, time zone and settings-hash function.
    metadct : dict, optional
        Additional metadata, added to ``uns`` (overwrites entries of the same name).
    gates : list, optional
        Gates applied right after reading (``gate.apply(adata)`` on the new AnnData, so gates on ``layers['raw']``
        such as `cytovanni.gating.MinMaxGate`); only events kept by all gates are returned. Default: no gating.
    print_gates : bool, default False
        Print how many events the gates removed.
    warn_spillover : bool, default False
        Warn (`RainbowSpilloverWarning`, with `warn_spillover_message`) if the file contains a non-trivial spillover
        matrix.
    warn_spillover_message : str, default "Non-trivial spillover matrix!"
        Text of that warning.
    fct_fix_df : callable, default identity
        Function applied to the raw event DataFrame (channels as columns, already renamed) before the AnnData is
        built, to repair instrument bugs, e.g. `fix_A3_HW_bug`.
    allow_missing_scatter : bool, default True
        If True, missing scatter channels only give a `ChannelMismatchWarning` and are left out; if False, they raise.

    Returns
    -------
    anndata.AnnData
        Events x channels, with:

        - ``layers['raw']``: raw intensities of the scatter channels (those available), the fluorescence channels
          and the optional -H/-W fluorescence channels present (``cytoconfig.channels_fluorescence_addHW``), in this
          order; ``X`` is empty.
        - ``var``: 'pnn', 'pns', 'png', 'pnv' (voltage, NaN if not in the file), 'pnr'; index: channel names.
        - ``obs``: the meta channels (``cytoconfig.channels_meta``, e.g. 'Time'); index: event numbers as str.
        - ``uns``: 'date' ($DATE as 'DD-MON-YYYY' if it can be parsed, else as in the file; None without $DATE/$BTIM),
          'UTC' (start of the recording, $DATE and $BTIM read as local time in ``cytoconfig.timezone``, in seconds
          since the epoch; NaN if not available), 'cytometer', 'uid' (``'<cytometer>-<int(UTC)>'``, or with the
          SHA-256 hash of `filepath` instead of the time if it is not available), 'cytoconfig', 'set_hash' (settings
          hash of `cytoconfig`), 'Nevents' (number of events in the file, before `gates`), and the entries of
          `metadct`.

    Raises
    ------
    ChannelMismatchException
        If a fluorescence or meta channel of `cytoconfig` is missing in the file, or a scatter channel with
        ``allow_missing_scatter=False``; the message lists the available channels.
    SettingsHashException
        If the settings hash cannot be computed.

    Examples
    --------
    >>> gates = [MinMaxGate(["FSC-A", "FSC-H", "SSC-A", "SSC-H"], 0, 4e5), FluorescenceMinMaxGate(cytoconfig)]
    >>> adata = readfcs_sample(path, cytoconfig, gates=gates, metadct={"patient": "P01"},
    ...                        fct_fix_df=fix_A3_HW_bug)
    """
    metadct = {} if metadct is None else metadct
    gates = [] if gates is None else gates
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
    
    missing_meta = [c for c in cytoconfig.channels_meta if c not in sample_df.columns]
    if missing_meta:
        raise ChannelMismatchException(f"Could not find meta channels {missing_meta} in file {filepath}!\nAvailable channels are: {list(sample_df.columns)}")

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
            adata.uns["UTC"] = _fcs_utc(sample.metadata["date"], sample.metadata["btim"], cytoconfig.timezone)
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


def read_fcs_timestamp(filepath, timezone=DEFAULT_TIMEZONE):
    """Read the date and recording times of an .fcs file from its text segment.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.
    timezone : str, default "Europe/Berlin"
        IANA time zone of the instrument clock, used to convert $DATE/$BTIM to 'UTC' (see
        ``CytometerConfiguration.timezone``).

    Returns
    -------
    pandas.Series
        'date' ($DATE as in the file, "" if missing), 'begin_record' ($BTIM), 'end_record' ($ETIM; both NaN if
        missing) and 'UTC' (start of the recording in seconds since the epoch; NaN without $DATE or $BTIM).

    Raises
    ------
    ValueError
        If $DATE and $BTIM are present but cannot be parsed.
    """
    data = flowio.FlowData(filepath, only_text=True)
    date = data.text["date"] if "date" in data.text else ""
    btim = data.text["btim"] if "btim" in data.text else np.nan
    etim = data.text["etim"] if "etim" in data.text else np.nan
    utc = _fcs_utc(date, btim, timezone) if (date and btim==btim) else np.nan
    return pd.Series([date, btim, etim, utc], index=["date", "begin_record", "end_record", "UTC"])

def read_folder_timestamps(folder, cytoconfig=None, timezone=None):
    """Read the date and recording times of all .fcs files in a folder (not recursive), sorted by time.

    Only the text segments are read, so this is fast also for large files.

    Parameters
    ----------
    folder : str
        Folder with the .fcs files (file names ending in ".fcs").
    cytoconfig : cytovanni.utils.CytometerConfiguration, optional
        If given, also read the settings hash of every file (see `readfcs_sample`, ``uns['set_hash']``).
    timezone : str, optional
        IANA time zone of the instrument clock. Default: ``cytoconfig.timezone`` if `cytoconfig` is given, else
        "Europe/Berlin".

    Returns
    -------
    pandas.DataFrame
        One row per file (index: file name), columns as in `read_fcs_timestamp` ('date', 'begin_record',
        'end_record', 'UTC') and, with `cytoconfig`, 'set_hash' (NaN where it could not be computed). Sorted by
        'UTC'; files without 'UTC' (no $DATE or $BTIM) come last, ordered by the 'date' and 'begin_record' strings.

    Raises
    ------
    ValueError
        If a file has a $DATE/$BTIM that cannot be parsed.

    Examples
    --------
    >>> tps = read_folder_timestamps("data/rainbow/", cytoconfig=cytoconfig)
    """
    if timezone is None:
        timezone = cytoconfig.timezone if cytoconfig is not None else DEFAULT_TIMEZONE
    files = list(filter(lambda x: x.endswith(".fcs"), os.listdir(folder)))
    tps = pd.DataFrame([read_fcs_timestamp(os.path.join(folder, file), timezone=timezone) for file in files], index=files,
                       columns=["date", "begin_record", "end_record", "UTC"])
    if cytoconfig is not None:
        def safe_hash(file):
            try:
                return readfcs_channel_settings_hash(os.path.join(folder, file), cytoconfig)
            except Exception as e:
                return np.nan
        tps["set_hash"] = [safe_hash(file) for file in files]
    return tps.sort_values(["UTC", "date", "begin_record"], na_position="last")


def fix_A3_HW_bug(df):
    """Repair swapped scatter -H and -W channels of files from our FACSymphony A3 (for `readfcs_sample`).

    In some files of that instrument, those where 'Time' is the first channel, FSC-H/FSC-W and SSC-H/SSC-W are
    swapped. This function swaps them back if the first column is 'Time' and leaves the data unchanged otherwise.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw events with the channel names as columns (needs FSC-H, FSC-W, SSC-H, SSC-W if 'Time' is first).

    Returns
    -------
    pandas.DataFrame
        `df`, changed in place.

    Examples
    --------
    >>> adata = readfcs_sample(path, cytoconfig, fct_fix_df=fix_A3_HW_bug)
    """
    if df.columns[0]=="Time":
        df["FSC-H"], df["FSC-W"] = df["FSC-W"], df["FSC-H"]
        df["SSC-H"], df["SSC-W"] = df["SSC-W"], df["SSC-H"]
    return df


def writefcs(filepath, data_df, sample_id="", add_metadata=None, overwrite_pns=True, pns=None):
    """Write a DataFrame of events as .fcs file (via flowkit); missing parent folders are created.

    Parameters
    ----------
    filepath : str
        Path of the new file, should end in .fcs.
    data_df : pandas.DataFrame
        Events x channels; the column names become the channel names ($PnN).
    sample_id : str, default ""
        Sample ID passed to ``flowkit.Sample``.
    add_metadata : dict, optional
        Additional metadata keywords written to the file (they update the metadata flowkit creates).
    overwrite_pns : bool, default True
        Set the channel labels ($PnS) to `pns`. If False, this function sets no $PnS labels.
    pns : list of str, optional
        $PnS labels, one per column. Default: the channel names ($PnN).

    Examples
    --------
    >>> writefcs("out/sample_calibrated.fcs", df, add_metadata={"note": "calibrated"})
    """
    from pathlib import Path
    Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
    
    sample = fk.Sample(data_df, sample_id=sample_id)
    sample.metadata.update({} if add_metadata is None else add_metadata)
    if overwrite_pns:
        sample.pns_labels = sample.pnn_labels if pns is None else pns
    sample.export(filepath, source="raw", include_metadata=True)

def fix_sample_bug(filepath, filepath_new, fct_bugfix, delete_spill=True):
    """Write a repaired copy of an .fcs file: read the raw events, apply `fct_bugfix`, write them to `filepath_new`.

    The metadata of the original file are passed on to `writefcs` (``add_metadata``), and each channel keeps its $PnS
    label from the original ("" where it had none).

    Parameters
    ----------
    filepath : str
        Path to the original .fcs file.
    filepath_new : str
        Path of the repaired file (parent folders are created).
    fct_bugfix : callable
        Function of the raw event DataFrame (columns: $PnN as in the file, not renamed) that returns the repaired
        DataFrame, e.g. `fix_A3_HW_bug`.
    delete_spill : bool, default True
        Drop the spillover keywords ($SPILL and $SPILLOVER) from the copied metadata.
    """
    sample = fk.Sample(filepath, preprocess=False)
    data_df = sample.as_dataframe(source="raw")
    data_df.columns = data_df.columns.get_level_values(0)
    data_df = fct_bugfix(data_df)
    add_metadata = sample.metadata
    if delete_spill: # flowkit stores $SPILL as 'spill' and $SPILLOVER as 'spillover'
        for key in ["spill", "spillover"]:
            add_metadata.pop(key, None)
    pns_original = dict(zip(sample.pnn_labels, sample.pns_labels))
    pns = [pns_original.get(channel, "") for channel in data_df.columns]
    writefcs(filepath_new, data_df, sample_id="", add_metadata=add_metadata, overwrite_pns=True, pns=pns)

def readfcs_dataframe(filepath):
    """Read the raw events of an .fcs file into a DataFrame, labelled with the channel labels.

    Unlike `readfcs_sample`, no cytometer configuration is used: no renaming, no channel checks, no metadata.
    flowkit's warnings are suppressed and offset errors in the file are ignored.

    Parameters
    ----------
    filepath : str
        Path to the .fcs file.

    Returns
    -------
    pandas.DataFrame
        Events x channels, raw values; column names are the $PnS labels where set, else the $PnN names.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample = fk.Sample(filepath, ignore_offset_error=True, preprocess=False)

    data = sample.as_dataframe(source="raw")
    cols = data.columns.get_level_values(1).to_numpy()
    mask = cols==""
    cols[mask] = data.columns.get_level_values(0)[mask]
    data.columns = cols

    return data

