import numpy as np
import pandas as pd
import os
import warnings

from ..exceptions import SpilloverFitException, SpilloverFitWarning

from .utils import _find_single_stain

def make_AutoSpill_fcs_config(single_stain_path, cytoconfig, warn_missing=True, pad_missing=True):
    """ Search for single stains, make fcs_control DataFrame that is needed to run AutoSpill.
    
        Throws a warning if some channels don't have a single stain associated with them.
        
        Since AutoSpill needs this (it only includes channels in the fit that have a single stain),
        use random other single stain for empty channels and later remember that they are not actually that channel.
        Their antigen is set as '_dummy_pad'.
        
        Expects single stains to conform to the naming as produces by FACSDiva,
            'Compensation Controls_(antigen ){channel} Stained Control.fcs'
        where 'antigen' can be empty, and 'channel' should be the channel name without -A.
        
        :param single_stain_path: str. Folder that contains all single stains.
        
        :param cytoconfig: CytometerConfiguration. Cytometer configuration for measurement series.
        
        :param warn_missing: bool. Whether to warn in case of missing single stains.
        
        :param pad_missing: bool. Whether to fill missing stains by random one to force AutoSpill to fit all channels.
    """
    fcs_control = pd.DataFrame([cytoconfig.channels_fluorescence], index=["dye"]).T
    fcs_control["wavelength"] = fcs_control.index+1
    fcs_control["channnel_name"] = fcs_control["dye"].apply(lambda x: x[:-2] if x.endswith("-A") else x)
    fcs_control["filename_regex"] = fcs_control["channnel_name"].apply(lambda c: f'Compensation Controls_(.*){c} Stained Control.fcs')

    fcs_control[["filename", "antigen"]] = ""
    for name, row in fcs_control.iterrows():
        fcs_control.loc[name, ["filename", "antigen"]] = _find_single_stain(single_stain_path, row["filename_regex"], row["channnel_name"], warn_missing=warn_missing)

    if fcs_control["filename"].apply(lambda x: len(x)>0).sum()==0:
        raise SpilloverFitException(f"Could not find any single stains in {single_stain_path}! Make sure they conform to the expected naming convention!")
    
    fcs_control["filename_source"] = fcs_control["filename"]
    if pad_missing:
        pad_file = [f for f in fcs_control["filename"] if len(f)>0][0]
        i = 0
        for name, row in fcs_control.iterrows():
            if len(row["filename"])==0:
                fcs_control.loc[name, ["filename_source", "filename", "antigen"]] = pad_file, pad_file.replace(".fcs",f"_{i}.fcs"), "_dummy_pad"
    
    return fcs_control[["filename_source","filename","dye","antigen","wavelength"]]
