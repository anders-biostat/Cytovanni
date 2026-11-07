"""Prepare the single-stain controls of one measurement for AutoSpill."""
import numpy as np
import pandas as pd
import os
import re
import warnings

from ..exceptions import SpilloverFitException, SpilloverFitWarning

from .utils import _find_single_stain

def make_AutoSpill_fcs_config(single_stain_path, cytoconfig, warn_missing=True, pad_missing=True):
    """Find the single-stain .fcs files of one measurement and build the ``fcs_control`` table for AutoSpill.

    For every fluorescence channel of `cytoconfig`, searches `single_stain_path` for the FACSDiva single-stain file
    ``'Compensation Controls_(antigen ){channel} Stained Control.fcs'``, where `antigen` may be empty and `channel`
    is the channel name without the ``-A`` suffix.

    AutoSpill only fits the channels that have a single stain. With `pad_missing`, every channel without a single
    stain is therefore assigned another single stain (the first one found) under a new file name, with the antigen
    ``'_dummy_pad'``. No file is copied here: ``filename_source`` names the existing file, ``filename`` the name
    under which AutoSpill expects it. The spillover AutoSpill fits for padded channels does not belong to them and
    has to be discarded later.

    Parameters
    ----------
    single_stain_path : str
        Folder that contains the single-stain .fcs files.
    cytoconfig : cytovanni.utils.CytometerConfiguration
        Cytometer configuration; its ``channels_fluorescence`` are the channels to search single stains for.
    warn_missing : bool, default True
        Warn (`SpilloverFitWarning`) for every channel without a single stain.
    pad_missing : bool, default True
        Assign another single stain to channels without one (see above).

    Returns
    -------
    pandas.DataFrame
        One row per fluorescence channel (integer index), columns ``filename_source`` (the existing file to use),
        ``filename`` (the name AutoSpill should see: equal to ``filename_source``, or for padded channels the source
        name with a suffix ``_<i>.fcs``; empty if missing and not padded), ``dye`` (the channel name, with ``-A``),
        ``antigen`` (from the file name; ``'_dummy_pad'`` for padded channels) and ``wavelength`` (a placeholder,
        the row number plus 1).

    Raises
    ------
    SpilloverFitException
        If no single stain is found at all, or the search for one channel is ambiguous (several matching files).
    """
    fcs_control = pd.DataFrame([cytoconfig.channels_fluorescence], index=["dye"]).T
    fcs_control["wavelength"] = fcs_control.index+1
    fcs_control["channel_name"] = fcs_control["dye"].apply(lambda x: x[:-2] if x.endswith("-A") else x)
    fcs_control["filename_regex"] = fcs_control["channel_name"].apply(lambda c: rf'Compensation Controls_(.*){re.escape(c)} Stained Control\.fcs$')

    fcs_control[["filename", "antigen"]] = ""
    for name, row in fcs_control.iterrows():
        fcs_control.loc[name, ["filename", "antigen"]] = _find_single_stain(single_stain_path, row["filename_regex"], row["channel_name"], warn_missing=warn_missing)

    if fcs_control["filename"].apply(lambda x: len(x)>0).sum()==0:
        raise SpilloverFitException(f"Could not find any single stains in {single_stain_path}! Make sure they conform to the expected naming convention!")
    
    fcs_control["filename_source"] = fcs_control["filename"]
    if pad_missing:
        pad_file = [f for f in fcs_control["filename"] if len(f)>0][0]
        i = 0
        for name, row in fcs_control.iterrows():
            if len(row["filename"])==0:
                fcs_control.loc[name, ["filename_source", "filename", "antigen"]] = pad_file, pad_file.replace(".fcs",f"_{i}.fcs"), "_dummy_pad"
                i += 1
    
    return fcs_control[["filename_source","filename","dye","antigen","wavelength"]]
