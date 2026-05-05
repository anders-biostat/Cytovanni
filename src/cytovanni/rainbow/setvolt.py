import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..io import readfcs_rainbow_sample
from ..gating import gate_scatter_bead_sample
from ..utils import textcolor
from .gmm import fit_rainbow_GMM
from .plot import plot_rainbow_channel_GMM_single

def get_config_from_rainbow_reference(cytoconfig, rainbow_reference_path, set_channels, N_rainbow_peaks,
                                      safety_factor_linear=.75, use_highest_peak=True, PMT_exponent=7., plot=True):
    """ Generate config file for semi-automated PMT voltage setting from a reference rainbow bead sample.
        Setting the voltages this way then reproduces the channel gains etc. from the reference.
        
        Reads sample, gates on scatter, fits peaks, uses highest peak per channel that is still in the (safe) linear range.
        Simple median instead of GMM fit for the scatter channels.
        
        :param cytoconfig: CytometerConfiguration. Cytometer configuration for the sample.
        
        :param rainbow_reference_path: str. Path to thr reference .fcs sample.
        
        :param set_channels: iterable. List of channels for which the voltages need to be set.
        
        :param N_rainbow_peaks: int. Number of peaks in the rainbow data.
        
        :param savepath: str. Where the final config .csv should be saved.
        
        :param safety_factor_linear: float. Takes max linear range from cytoconfig, uses peaks up to safety_factor_linear times the max linear range as a safety offset.
        
        :param use_highest_peak: bool. Sometimes the highest rainbow peak may have some issues, using it can be turned off here.
        
        :param PMT_exponent: float, pd.Series. PMT exponent that should be assumed for the detectors. Either single value for all, or pd.Series with index set_channels if different ones should be used for every channel.
        
        :param plot: bool. Whether gating and GMM fit should be plotted to make sure everything worked fine.
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

    highest_linear_peak_ind = np.argmax(df_peakpos.to_numpy(), axis=0)
    highest_linear_peak_val = np.round(df_peakpos.to_numpy()[highest_linear_peak_ind, np.arange(len(highest_linear_peak_ind))], 0).astype(int)
    df_target_fluoro = pd.DataFrame([highest_linear_peak_val,
                                     1+highest_linear_peak_ind],
                                     index=["target", "peak"], columns=cytoconfig.channels_fluorescence).T
    df_target_fluoro["peak"] = df_target_fluoro["peak"].astype(int)

    targetconfig = pd.concat([df_target_scatter,
                              df_target_fluoro.loc[set_channels_fluorescence]])
    targetconfig["target"] = targetconfig["target"].astype(int)
    # Some playing around to get nicer formatting
    targetconfig["peak"] = np.array([int(x) if x==x else x for x in targetconfig["peak"]], dtype=object)
    if isinstance(PMT_exponent, pd.Series):
        targetconfig["exponent"] = PMT_exponent.loc[targetconfig.index]
    else:
        targetconfig["exponent"] = PMT_exponent
    
    targetconfig.loc["_N_peak", ["target", "peak"]] = [0, N_rainbow_peaks]

    return targetconfig

def get_voltages_from_rainbow(cytoconfig, targetconfig, rainbow_path, plot=True):
    """ Use newly recorded rainbow bead sample and prior config to estimate the voltages that reproduce the old settings.
        Depending on the initial settings, a couple of iterations of measuring rainbow beads,
        running this, and setting the PMT voltages to 'voltage_target' will yield good detector settings.
        Convergence speed will depend on how accurate the guessed PMT exponents are.
        'error (%)' is the error of the peak positions in percent, stop iterating if no errors are above 1-2%.
        
        :param cytoconfig: CytometerConfiguration. Configuration of the cytometer.
        
        :param targetconfig: DataFrame. Old settings config to reproduce, produced by get_config_from_rainbow_reference.
        
        :param rainbow_path: str. Path to the newly measured rainbow beads.
        
        :param plot: bool. Whether to plot stuff.
    """
    N_rainbow_peak = targetconfig.loc["_N_peak", "peak"]
    metainds = ["_N_peak"]
    targetconfig = targetconfig.iloc[~np.in1d(targetconfig.index, metainds)].copy()

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
    targetconfig["voltage_target"] = np.round(targetconfig["voltage"] * targetconfig["factor"], 0).astype(int)

    targetconfig["target"] = targetconfig["target"].astype(int)
    targetconfig["value"] = targetconfig["value"].astype(int)
    targetconfig["voltage"] = targetconfig["voltage"].astype(int)
    targetconfig["factor"] = np.round(targetconfig["factor"], 4)
    
    return targetconfig

def color_voltageconfig(voltageconfig, errorlevel=1, simplify=True):
    """ Get simplified, colored version of voltageconfig for printing.
        Will color all channels with error larger than errorlevel.
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
