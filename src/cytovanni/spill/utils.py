"""Collect single-stain controls: find the .fcs files, gate them and write them as .h5ad files with a metadata table."""
import numpy as np
import pandas as pd
import os
import re
import warnings
from tqdm import tqdm
from pathlib import Path
import warnings

from ..exceptions import SpilloverFitException, SpilloverFitWarning
from ..gating import gate_scatter_bead_sample
from ..io import readfcs_sample, write_bead_h5ad, read_bead_h5ad


def _find_single_stain(single_stain_path, filename_regex, channel, warn_missing=True):
    """Find the single-stain file matching `filename_regex`; return (file name, antigen) or ("", "").

    The antigen is the stripped first regex group. Raises `SpilloverFitException` if the match is ambiguous; warns
    (`SpilloverFitWarning`, if `warn_missing`) and returns empty strings if no file matches.
    """
    files = list(filter(lambda x: x.endswith(".fcs"), os.listdir(single_stain_path)))
    matches = [re.findall(filename_regex, filename) for filename in files]
    
    # Fatal errors
    if max([len(m) for m in matches], default=0)>1:
        raise SpilloverFitException(f"Something went wrong when searching for the file for channel {channel}, more than one regex match per file!")
    if sum([len(m) for m in matches])>1:
        raise SpilloverFitException(f"Something went wrong when searching for the file for channel {channel}, found multiple possible matches:\n{[f for f,m in zip(files, matches) if len(m)>0]}")
    
    # if none found
    if sum([len(m) for m in matches])==0:
        if warn_missing:
            warnings.warn(f"Found no availiable single stain for channel {channel}!", SpilloverFitWarning)
        return "", ""
    
    # if found
    matchind = np.argmax([len(m) for m in matches])
    marker = matches[matchind][0].strip()
    file = files[matchind]
    return file, marker

def _get_AutoSpill_script_path():
    """Path of the bundled AutoSpill R script ``AutoSpill/run_AutoSpil.R``."""
    return os.path.join(os.path.dirname(__file__), "AutoSpill", "run_AutoSpil.R")


def collect_bead_controls_regularflow(single_stain_path, cytoconfig, collect_data_path, warn_missing=True, drop=[], verbose=True):
    """Collect the bead single-stain controls of one conventional-flow measurement as recorded by FACSDiva.

    For every fluorescence channel of `cytoconfig`, finds the single-stain file
    ``'Compensation Controls_(antigen ){dye} Stained Control.fcs'`` (`dye` is the channel name without ``-A``,
    `antigen` may be empty), reads it with `cytovanni.io.readfcs_sample`, gates it on bead singlets with
    `cytovanni.gating.gate_scatter_bead_sample` and writes it as .h5ad. The results are written into
    `collect_data_path`:

    - ``data/Compensation_Control_{channel}.h5ad``: the gated events (raw intensities in ``layers['raw']``), with
      ``obs['dye']``, ``obs['channel']``, ``uns['dye']`` and ``uns['channel']`` set. ``uns['channel']`` marks the
      main channel of the dye for the single-stain fitters (conventional-flow normalisation).
    - ``gating/scattergate_{dye}.jpg``: plot of the scatter gate.
    - ``panel_meta.csv``: one row per fluorescence channel (integer index) with columns ``channel``, ``dye``,
      ``antigen`` (from the file name), ``datafile`` (empty if no single stain was found) and ``set_hash`` (cytometer
      settings hash, NaN if no file). This is the metadata table `SpilloverMeasurement.from_custom` reads.

    Parameters
    ----------
    single_stain_path : str
        Folder with the single-stain .fcs files.
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Cytometer configuration of the measurement.
    collect_data_path : str
        Output folder (created if needed).
    warn_missing : bool, default True
        Warn (`SpilloverFitWarning`) for every channel without a single stain.
    drop : list of str, default []
        Dyes (channel names without ``-A``) to skip, e.g. single stains that were not recorded on beads.
    verbose : bool, default True
        Show a progress bar.

    Raises
    ------
    SpilloverFitException
        If the file search for a channel is ambiguous (several matching files).

    Warns
    -----
    SpilloverFitWarning
        If the single stains were recorded at different cytometer settings (different ``set_hash``).

    See Also
    --------
    collect_bead_controls_spectralflow : the same for spectral flow, dyes taken from the file names.
    """
    fcs_control = pd.DataFrame([cytoconfig.channels_fluorescence], index=["channel"]).T
    fcs_control["dye"] = fcs_control["channel"].apply(lambda x: x[:-2] if x.endswith("-A") else x)
    fcs_control["filename_regex"] = fcs_control["dye"].apply(lambda c: rf'Compensation Controls_(.*){re.escape(c)} Stained Control\.fcs$')
    
    fcs_control[["filename", "antigen"]] = ""
    for name, row in fcs_control.iterrows():
        if row["dye"] not in drop:
            fcs_control.loc[name, ["filename", "antigen"]] = _find_single_stain(single_stain_path, row["filename_regex"], row["dye"], warn_missing=warn_missing)
    
    Path(os.path.dirname(collect_data_path)).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "gating", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "data", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    
    fcs_control = fcs_control.drop("filename_regex", axis=1)
    fcs_control["datafile"] = ""
    fcs_control["set_hash"] = pd.Series(np.nan, index=fcs_control.index, dtype=object) # object column: holds str hashes, NaN where no file
    pbar = tqdm(fcs_control.iterrows(), total=fcs_control.shape[0]) if verbose else fcs_control.iterrows()
    for name, row in pbar:
        if row["filename"]:
            adata = readfcs_sample(os.path.join(single_stain_path, row["filename"]), cytoconfig)
            adata = gate_scatter_bead_sample(adata, plot=False, titlefct=lambda x: row['dye'],
                                             plotpath=os.path.join(collect_data_path, "gating", f"scattergate_{row['dye']}.jpg"))
            adata.obs["dye"] = row["dye"]
            adata.obs["channel"] = row["channel"]
            adata.uns["dye"] = row["dye"]
            adata.uns["channel"] = row["channel"]
            
            fcs_control.loc[name, "set_hash"] = adata.uns["set_hash"]
            
            datafile = f"Compensation_Control_{row['channel']}.h5ad"
            fcs_control.loc[name, "datafile"] = datafile
            write_bead_h5ad(adata, os.path.join(collect_data_path, "data", datafile))
    
    if np.unique(fcs_control["set_hash"][fcs_control["set_hash"]==fcs_control["set_hash"]]).shape[0]>1:
        warnings.warn("Found multiple different cytometer settings in single stains!", SpilloverFitWarning)
            
            
    fcs_control = fcs_control.drop("filename", axis=1)
    fcs_control.to_csv(os.path.join(collect_data_path, "panel_meta.csv"))

def collect_bead_controls_spectralflow(single_stain_path, cytoconfig, collect_data_path, verbose=True, filename_regex=r'Compensation Controls_(.*)\.fcs$'):
    """Collect the bead single-stain controls of one spectral-flow measurement.

    Reads every file in `single_stain_path` that matches `filename_regex` (the dye name is the first regex group),
    gates it on bead singlets with `cytovanni.gating.gate_scatter_bead_sample` and writes the results into
    `collect_data_path`:

    - ``data/Compensation_Control_{dye}.h5ad``: the gated events (raw intensities in ``layers['raw']``), with
      ``obs['dye']`` and ``uns['dye']`` set. No ``uns['channel']`` is set, so the single-stain fitters use the
      channel with the highest intensity as main channel (spectral-flow normalisation).
    - ``gating/scattergate_{dye}.jpg``: plot of the scatter gate.
    - ``panel_meta.csv``: one row per file (integer index, sorted by dye) with columns ``dye``, ``datafile`` and
      ``set_hash`` (cytometer settings hash).

    Parameters
    ----------
    single_stain_path : str
        Folder with the single-stain .fcs files.
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Cytometer configuration of the measurement.
    collect_data_path : str
        Output folder (created if needed).
    verbose : bool, default True
        Show a progress bar.
    filename_regex : str, default r'Compensation Controls_(.*)\\.fcs$'
        Regular expression for the single-stain file names, with one group that matches the dye name.

    Warns
    -----
    SpilloverFitWarning
        If the single stains were recorded at different cytometer settings (different ``set_hash``).

    See Also
    --------
    collect_bead_controls_regularflow : the same for conventional flow (FACSDiva file names, one dye per channel).
    """
    files = os.listdir(single_stain_path)
    files = [f for f in files if len(re.findall(filename_regex, f))>0]

    fcs_control = pd.DataFrame([files], index=["filename"]).T
    fcs_control["dye"] = [re.findall(filename_regex, f)[0] for f in files]
    fcs_control = fcs_control.sort_values("dye").reset_index(drop=True)
    
    Path(os.path.dirname(collect_data_path)).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "gating", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "data", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    
    fcs_control["datafile"] = ""
    fcs_control["set_hash"] = pd.Series(np.nan, index=fcs_control.index, dtype=object) # object column: holds str hashes, NaN where no file
    pbar = tqdm(fcs_control.iterrows(), total=fcs_control.shape[0]) if verbose else fcs_control.iterrows()
    for name, row in pbar:
        if row["filename"]:
            adata = readfcs_sample(os.path.join(single_stain_path, row["filename"]), cytoconfig)
            adata = gate_scatter_bead_sample(adata, plot=False, titlefct=lambda x: row['dye'],
                                             plotpath=os.path.join(collect_data_path, "gating", f"scattergate_{row['dye']}.jpg"))
            adata.obs["dye"] = row["dye"]
            adata.uns["dye"] = row["dye"]
            
            fcs_control.loc[name, "set_hash"] = adata.uns["set_hash"]
            
            datafile = f"Compensation_Control_{row['dye']}.h5ad"
            fcs_control.loc[name, "datafile"] = datafile
            write_bead_h5ad(adata, os.path.join(collect_data_path, "data", datafile))
    
    if np.unique(fcs_control["set_hash"][fcs_control["set_hash"]==fcs_control["set_hash"]]).shape[0]>1:
        warnings.warn("Found multiple different cytometer settings in single stains!", SpilloverFitWarning)
    
    fcs_control = fcs_control.drop("filename", axis=1)
    fcs_control.to_csv(os.path.join(collect_data_path, "panel_meta.csv"))
