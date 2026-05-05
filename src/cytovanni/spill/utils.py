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
    """ Find file for specific single stain.
    """
    files = list(filter(lambda x: x.endswith(".fcs"), os.listdir(single_stain_path)))
    matches = [re.findall(filename_regex, filename) for filename in files]
    
    # Fatal errors
    if max([len(m) for m in matches])>1:
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
    """ Get path to script file run_AutoSpil.R
    """
    return os.path.join(os.path.dirname(__file__), "AutoSpill", "run_AutoSpil.R")


def collect_bead_controls_regularflow(single_stain_path, cytoconfig, collect_data_path, warn_missing=True, drop=[], verbose=True):
    """ Collect bead single stain controls as produced by FACSDiva.
        Also does the gating, produces .h5ad files for data and plots for gating.
    
        Throws a warning if some channels don't have a single stain associated with them.
        
        Expects single stains to conform to the naming as produces by FACSDiva,
            'Compensation Controls_(antigen ){channel} Stained Control.fcs'
        where 'antigen' can be empty, and 'channel' should be the channel name without -A.
        
        :param single_stain_path: str. Path to the single stain .fcs files.
        
        :param cytoconfig: CytometerConfiguration. Cytometer configuration for measurement series.
        
        :param collect_data_path: str. Folder where data should be collected.
        
        :param warn_missing: bool. Whether to warn in case of missing single stains.
        
        :param drop: iterable. Single stains to ignore because they are not beads etc.
        
        :param verbose: bool. Whether to show progress bar.
    """
    fcs_control = pd.DataFrame([cytoconfig.channels_fluorescence], index=["channel"]).T
    fcs_control["dye"] = fcs_control["channel"].apply(lambda x: x[:-2] if x.endswith("-A") else x)
    fcs_control["filename_regex"] = fcs_control["dye"].apply(lambda c: f'Compensation Controls_(.*){c} Stained Control.fcs')
    
    fcs_control[["filename", "antigen"]] = ""
    for name, row in fcs_control.iterrows():
        if row["dye"] not in drop:
            fcs_control.loc[name, ["filename", "antigen"]] = _find_single_stain(single_stain_path, row["filename_regex"], row["dye"], warn_missing=True)
    
    Path(os.path.dirname(collect_data_path)).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "gating", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "data", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    
    fcs_control = fcs_control.drop("filename_regex", axis=1)
    fcs_control["datafile"] = ""
    fcs_control["set_hash"] = np.nan
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

def collect_bead_controls_spectralflow(single_stain_path, cytoconfig, collect_data_path, verbose=True, filename_regex=f'Compensation Controls_(.*).fcs'):
    """ Collect spectral single stain data.
        
        :param single_stain_path: str. Path to the single stain .fcs files.
        
        :param cytoconfig: CytometerConfiguration. Cytometer configuration for measurement series.
        
        :param collect_data_path: str. Folder where data should be collected.
        
        :param verbose: bool. Whether to show progress bar.
        
        :param filename_regex: str. Regex for single stain file name, should contain one group that matches the dye name.
    """
    files = os.listdir(single_stain_path)
    filename_regex = f'Compensation Controls_(.*).fcs'
    files = [f for f in files if len(re.findall(filename_regex, f))>0]

    fcs_control = pd.DataFrame([files], index=["filename"]).T
    fcs_control["dye"] = [re.findall(filename_regex, f)[0] for f in files]
    fcs_control = fcs_control.sort_values("dye").reset_index(drop=True)
    
    Path(os.path.dirname(collect_data_path)).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "gating", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    Path(os.path.dirname(os.path.join(collect_data_path, "data", ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
    
    fcs_control["datafile"] = ""
    fcs_control["set_hash"] = np.nan
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
