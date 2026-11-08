"""Semi-automated PMT voltage setting with rainbow beads: reproduce the detector settings of a reference measurement."""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io import readfcs_rainbow_sample
from ..gating import gate_scatter_bead_sample
from ..utils import textcolor
from ..exceptions import CytovanniWarning
from .gmm import fit_rainbow_GMM
from .plot import plot_rainbow_channel_GMM_single

def get_config_from_rainbow_reference(cytoconfig, rainbow_reference_path, set_channels, N_rainbow_peaks,
                                      safety_factor_linear=.75, use_highest_peak=True, PMT_exponent=7., plot=True):
    """Target intensities for setting PMT voltages, from a reference rainbow bead measurement.

    Reads the reference, gates it on scatter (`cytovanni.gating.gate_scatter_bead_sample`) and fits the peaks
    (`fit_rainbow_GMM`, default settings). For every fluorescence channel, the target is the raw GMM mean of the
    brightest peak below ``safety_factor_linear * cytoconfig.max_linear_range``; for scatter channels it is the
    median of the gated beads. Setting the voltages so that a new rainbow measurement reaches these targets
    (`get_voltages_from_rainbow`) reproduces the channel gains of the reference.

    Parameters
    ----------
    cytoconfig : CytometerConfiguration
        Configuration of the cytometer.
    rainbow_reference_path : str
        Path of the reference rainbow bead .fcs file.
    set_channels : list of str
        Channels whose voltages are set (scatter and fluorescence channels of `cytoconfig`).
    N_rainbow_peaks : int
        Number of peaks of the rainbow beads.
    safety_factor_linear : float, default 0.75
        Only peaks up to this fraction of the maximum linear range are used as targets.
    use_highest_peak : bool, default True
        Allow the brightest peak (last in GMM order) as target; turn off if it is unreliable.
    PMT_exponent : float or pandas.Series, default 7.0
        Assumed exponent of the gain in the PMT voltage (gain proportional to ``voltage ** exponent``): one value for
        all channels, or a Series with index `set_channels` (e.g. fitted with `PMTExponentFitter`).
    plot : bool, default True
        Show the scatter gate and the GMM fit (to check them).

    Returns
    -------
    pandas.DataFrame
        Index the channels (scatter channels first, then the fluorescence channels in the order of `set_channels`) plus
        a last row '_N_peak'. Columns 'target' (target intensity, raw units, rounded; nullable integer), 'peak'
        (1-based peak number in GMM order for fluorescence channels, NaN for scatter channels) and 'exponent'. Row
        '_N_peak' holds the number of peaks in 'peak' (target 0, exponent NaN). The input of `get_voltages_from_rainbow`; nothing is saved.

    Raises
    ------
    Exception
        If a channel of `set_channels` is neither a scatter nor a fluorescence channel of `cytoconfig`.

    See Also
    --------
    get_voltages_from_rainbow : voltages that reach the targets.

    Warns
    -----
    CytovanniWarning
        If no peak of a channel in `set_channels` is below the limit; its target is then NaN (``<NA>``, 'peak' 1),
        and `get_voltages_from_rainbow` gives NaN for its factor, error and target voltage.

    Examples
    --------
    >>> targetconfig = get_config_from_rainbow_reference(cytoconfig, "reference_rainbow.fcs", ["FSC-A", "SSC-A",
    ...                                                  "BV421-A"], N_rainbow_peaks=8, PMT_exponent=7.)
    >>> targetconfig.to_csv("targetconfig.csv")
    """
    
    adata = readfcs_rainbow_sample(rainbow_reference_path, cytoconfig, N_rainbow_peaks, name="Rainbow Reference")
    adata = gate_scatter_bead_sample(adata, plot=plot, plotpath="")
    if plot: plt.show()

    fit_rainbow_GMM(adata)
    if plot:
        plot_rainbow_channel_GMM_single(adata)
        plt.show()

    set_channels_scatter = [c for c in set_channels if c in cytoconfig.channels_scatter]
    set_channels_fluorescence = [c for c in set_channels if c in cytoconfig.channels_fluorescence]
    set_channels_leftover = set(set_channels) - set(set_channels_scatter) - set(set_channels_fluorescence)
    if len(set_channels_leftover)>0:
        raise Exception(f"Could not find channels {set_channels_leftover}!")

    df_target_scatter = pd.DataFrame(np.round(np.median(adata[:,set_channels_scatter].layers["raw"], axis=0)[:,None], 0).astype(int),
                                     columns=["target"], index=set_channels_scatter)
    df_target_scatter["peak"] = np.nan

    df_peakpos = pd.DataFrame(adata.uns["GMM_means"].copy(),
                              index=np.arange(adata.uns["N_rainbow_peak"]),
                              columns=cytoconfig.channels_fluorescence)
    df_peakpos[df_peakpos > cytoconfig.max_linear_range*safety_factor_linear] = 0.
    if not use_highest_peak:
        df_peakpos.iloc[-1] = 0.
    no_linear_peak = pd.Series((df_peakpos.to_numpy() <= 0).all(axis=0), index=cytoconfig.channels_fluorescence)
    if no_linear_peak[set_channels_fluorescence].any():
        warnings.warn(f"No rainbow peak below the linear limit for channels "
                      f"{no_linear_peak[set_channels_fluorescence][lambda x: x].index.tolist()}, their target is NaN; "
                      f"lower their voltage in the reference or raise safety_factor_linear.", CytovanniWarning)

    highest_linear_peak_ind = np.argmax(df_peakpos.to_numpy(), axis=0)
    highest_linear_peak_val = np.round(df_peakpos.to_numpy()[highest_linear_peak_ind, np.arange(len(highest_linear_peak_ind))], 0)
    highest_linear_peak_val[no_linear_peak.to_numpy()] = np.nan
    df_target_fluoro = pd.DataFrame([highest_linear_peak_val,
                                     1+highest_linear_peak_ind],
                                     index=["target", "peak"], columns=cytoconfig.channels_fluorescence).T
    df_target_fluoro["peak"] = df_target_fluoro["peak"].astype(int)

    targetconfig = pd.concat([df_target_scatter,
                              df_target_fluoro.loc[set_channels_fluorescence]])
    targetconfig["target"] = targetconfig["target"].astype("Int64")
    # Some playing around to get nicer formatting
    targetconfig["peak"] = np.array([int(x) if x==x else x for x in targetconfig["peak"]], dtype=object)
    if isinstance(PMT_exponent, pd.Series):
        targetconfig["exponent"] = PMT_exponent.loc[targetconfig.index]
    else:
        targetconfig["exponent"] = PMT_exponent
    
    targetconfig.loc["_N_peak", ["target", "peak"]] = [0, N_rainbow_peaks]

    return targetconfig

def get_voltages_from_rainbow(cytoconfig, targetconfig, rainbow_path, plot=True):
    """Estimate the PMT voltages that reproduce the targets of `get_config_from_rainbow_reference`.

    Reads a newly measured rainbow bead file, gates and fits it like the reference, and measures every channel's
    current value (scatter: median; fluorescence: GMM mean of the target peak). With the current voltage V (from
    ``adata.var['pnv']``) and the exponent e, the new voltage is ``V * (target / value) ** (1 / e)``. Iterate:
    measure rainbow beads, run this, set the voltages to 'voltage_target', until no 'error (%)' is above about
    1-2 %. How many iterations are needed depends mostly on how accurate the exponents are.

    Parameters
    ----------
    cytoconfig : CytometerConfiguration
        Configuration of the cytometer.
    targetconfig : pandas.DataFrame
        Output of `get_config_from_rainbow_reference` (or that table read back from .csv).
    rainbow_path : str
        Path of the new rainbow bead .fcs file.
    plot : bool, default True
        Show the scatter gate and the GMM fit.

    Returns
    -------
    pandas.DataFrame
        `targetconfig` without the '_N_peak' row, with the added columns 'value' (current intensity, raw units),
        'voltage' (current voltage), 'factor' (voltage factor, rounded to 4 digits), 'error (%)' (``(value -
        target) / target * 100``, rounded to 0.1) and 'voltage_target' (suggested voltage, rounded to an integer).

    Warns
    -----
    RainbowSpilloverWarning
        If the rainbow file contains a non-trivial spillover matrix.

    See Also
    --------
    color_voltageconfig : coloured table for printing.

    Examples
    --------
    >>> voltageconfig = get_voltages_from_rainbow(cytoconfig, targetconfig, "rainbow_today.fcs", plot=False)
    >>> print(color_voltageconfig(voltageconfig, errorlevel=1))
    """
    N_rainbow_peak = int(targetconfig.loc["_N_peak", "peak"])
    metainds = ["_N_peak"]
    targetconfig = targetconfig.iloc[~np.isin(targetconfig.index, metainds)].copy()

    adata = readfcs_rainbow_sample(rainbow_path, cytoconfig, N_rainbow_peak, name="Rainbow Sample", warn_spillover=True)
    adata = gate_scatter_bead_sample(adata, plot=plot, plotpath="")
    if plot: plt.show()

    fit_rainbow_GMM(adata)
    if plot:
        plot_rainbow_channel_GMM_single(adata)
        plt.show()

    set_channels_scatter = targetconfig[targetconfig["peak"]!=targetconfig["peak"]].index.tolist()
    set_channels_fluorescence = targetconfig[targetconfig["peak"]==targetconfig["peak"]].index.tolist()

    value_scatter = pd.Series(np.round(np.median(adata[:,set_channels_scatter].layers["raw"], axis=0), 0),
                                         index=set_channels_scatter)

    df_peakpos = pd.DataFrame(adata.uns["GMM_means"].copy(),
                                  index=np.arange(adata.uns["N_rainbow_peak"]),
                                  columns=cytoconfig.channels_fluorescence)
    value_peak = pd.Series(
        df_peakpos.to_numpy()[(targetconfig.loc[set_channels_fluorescence, "peak"].to_numpy()-1).astype(int),
                              df_peakpos.columns.get_indexer(set_channels_fluorescence)],
        index=set_channels_fluorescence)

    targetconfig["value"] = np.round(pd.concat([value_scatter, value_peak]), 0)

    targetconfig["voltage"] = adata.var.loc[targetconfig.index, "pnv"]

    targetconfig["factor"] = (targetconfig["target"]/targetconfig["value"]) ** (1/targetconfig["exponent"])
    targetconfig["error (%)"] = np.round((targetconfig["value"]-targetconfig["target"]) / targetconfig["target"] *100, 1)
    targetconfig["voltage_target"] = np.round(targetconfig["voltage"] * targetconfig["factor"], 0).astype("Int64")

    targetconfig["target"] = targetconfig["target"].astype("Int64")
    targetconfig["value"] = targetconfig["value"].astype(int)
    targetconfig["voltage"] = targetconfig["voltage"].astype(int)
    targetconfig["factor"] = np.round(targetconfig["factor"], 4)
    
    return targetconfig

def color_voltageconfig(voltageconfig, errorlevel=1, simplify=True):
    """Colour the output of `get_voltages_from_rainbow` for printing in a terminal or notebook.

    Parameters
    ----------
    voltageconfig : pandas.DataFrame
        Output of `get_voltages_from_rainbow`.
    errorlevel : float, default 1
        Channels with ``abs(error (%))`` above this value are coloured red (as needing adjustment), the others dark
        cyan (index, 'error (%)', 'voltage_target' and a last column repeating the channel name).
    simplify : bool, default True
        Keep only the columns 'target', 'peak', 'voltage', 'value', 'error (%)' and 'voltage_target'.

    Returns
    -------
    pandas.DataFrame
        Table of strings with ANSI colour codes (NaN shown as empty), for ``print``.
    """
    if simplify:
        voltageconfig = voltageconfig[["target","peak","voltage","value","error (%)","voltage_target"]]
    #voltageconfig["channel"] = voltageconfig.index
    voltageconfig = voltageconfig.astype(str).replace("nan","")
    
    needs_adjustment = voltageconfig["error (%)"].apply(lambda x: np.abs(float(x))>errorlevel)
    
    def adjust(ar, color="RED"):
        return [(textcolor.toCOLOR(i, color) if adj else textcolor.toCOLOR(i, "DARKCYAN")) for i, adj in zip(ar, needs_adjustment)]
    
    voltageconfig.index = adjust(voltageconfig.index)
    voltageconfig["error (%)"] = adjust(voltageconfig["error (%)"].tolist())
    voltageconfig["voltage_target"] = adjust(voltageconfig["voltage_target"].tolist())
    voltageconfig[""] = adjust(voltageconfig.index)
    
    return voltageconfig
