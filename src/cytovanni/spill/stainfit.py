"""Fit the spectrum of one dye from a single-stain control, with several methods, and collect the fits per method.

All fitters work on the raw intensities (``adata.layers['raw']``) of a gated single-stain .h5ad as written by
`collect_bead_controls_regularflow` / `collect_bead_controls_spectralflow` and read with
`cytovanni.io.read_bead_h5ad`. The spectrum is a slope per fluorescence channel relative to the main channel of the
dye (slope 1 in the main channel), plus an intercept per channel.
"""
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
import os
from pathlib import Path
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm

from ..exceptions import SpilloverFitWarning
from ..utils import unrolled_subplots, get_cmap, cmap_to_legendhandles

def _find_max_channel(ad):
    """Main channel of a single stain: the fluorescence channel with the highest 95th percentile of raw intensity."""
    return ad.uns["cytoconfig"].channels_fluorescence[np.argmax(np.percentile(ad[:,ad.uns["cytoconfig"].channels_fluorescence].layers["raw"],
                                                                             95, axis=0))]

class SingleStainSpectrumFitBase():
    """Base class of the single-stain spectrum fitters: data checks, main channel, plotting.

    The main channel (``stain``) is ``adata.uns['channel']`` if that is set and a channel of `adata` (conventional
    flow, set by `collect_bead_controls_regularflow`); otherwise the fluorescence channel with the highest 95th
    percentile of the raw intensity (spectral flow). The spectrum is expressed relative to this channel.

    On construction, events above the linear range are removed (`assure_linearity`). The data are kept as a gated copy
    in ``self.adata``; the input `adata` is not changed.

    Parameters
    ----------
    adata : anndata.AnnData
        Gated single-stain events with raw intensities in ``layers['raw']``, ``uns['cytoconfig']``
        (`cytovanni.utils.CytometerConfiguration`; its ``channels_fluorescence`` are fitted and its
        ``max_linear_range`` is the linear limit), ``uns['dye']`` and optionally ``uns['channel']``.
    linearity_tolerance_factor : float, default 2.0
        Events with any fluorescence channel above ``linearity_tolerance_factor * max_linear_range`` (raw) are
        dropped.
    asinh_cofactor : float, default 1500
        Cofactor (raw intensity units) of the arcsinh transform; only used by `PosNegSingleStainSpectrumFit`.
    """
    WARN_SHARE = .01
    def __init__(self, adata, linearity_tolerance_factor=2., asinh_cofactor=1500):
        self.adata = adata
        self.channels = adata.uns["cytoconfig"].channels_fluorescence
        self.dye = adata.uns["dye"]
        self.stain = adata.uns["channel"] if ("channel" in adata.uns) and (adata.uns["channel"] in adata.var.index) else _find_max_channel(adata)
        self.max_linear_range = adata.uns["cytoconfig"].max_linear_range
        self.linearity_tolerance_factor = linearity_tolerance_factor
        self.asinh_cofactor = asinh_cofactor
        
        self.assure_linearity()
        
    def assure_linearity(self):
        """Drop events outside the tolerated linear range (in place on ``self.adata``).

        Events with any fluorescence channel above ``linearity_tolerance_factor * max_linear_range`` are removed. Events
        between ``max_linear_range`` and that limit are kept. Warns (`SpilloverFitWarning`) if more than 1%
        (``WARN_SHARE``) of the events have a channel above ``max_linear_range``.
        """
        mask_linear = ~np.any(self.adata[:,self.channels].layers["raw"]>self.max_linear_range, axis=1)
        mask_linear_wtol = ~np.any(self.adata[:,self.channels].layers["raw"]>self.linearity_tolerance_factor*self.max_linear_range, axis=1)
        nonlinear_share = 1-mask_linear.mean()
        nonlinear_wtol_share = 1-mask_linear_wtol.mean()
        
        if nonlinear_share>self.WARN_SHARE:
            warnstr = f"Single stain for {self.dye} has {nonlinear_share*100:.1f}% in nonlinear range!"
            warnstr+= f" Dropping {nonlinear_wtol_share*100:.1f}% that is outside tolerance of linear range."
            warnings.warn(warnstr, SpilloverFitWarning)
        
        self.adata = self.adata[mask_linear_wtol].copy()
    
    def get_spectrum(self):
        """Return the fitted spectrum: pandas.DataFrame of shape (channel, 1), the slopes, column named by the dye."""
        spectrum = self.spectrum[["slope"]]
        spectrum.columns = [self.dye]
        return spectrum

    def plot(self, folder="", s=.1):
        """Plot every fluorescence channel against the main channel, with the fitted line and the compensated data.

        Each panel shows the raw data, the fitted line ``slope * x + intercept``, the compensated values
        ``y - slope * x`` and the intercept.

        Parameters
        ----------
        folder : str, default ""
            If given, save the figure as ``{folder}/{PLOTFOLDERNAME}/{dye}.jpg`` (folder created if needed), close it.
        s : float, default 0.1
            Marker size of the scatter plots.
        """

        fig, ax = unrolled_subplots(len(self.channels), Ncol=5, elsize=(8,5))
        fig.suptitle(self.dye, size=40)

        for i in range(len(self.channels)):
            ax[i].set_xlabel(self.stain, size=20)
            ax[i].set_ylabel(self.channels[i], size=20)

            x = self.adata[:,self.stain].layers["raw"][:,0].copy()
            y = self.adata[:,self.channels[i]].layers["raw"][:,0].copy()
            ax[i].scatter(x=x, y=y, s=s, label="raw")

            slope = self.spectrum.loc[self.channels[i], "slope"]
            intercept = self.spectrum.loc[self.channels[i], "intercept"]

            xord = np.sort(x)
            ax[i].plot(xord, xord * slope + intercept)

            y_comp = y - (x * slope)
            ax[i].scatter(x=x, y=y_comp, s=s, label="compensated")
            ax[i].plot(xord, np.repeat(intercept, len(xord)))
        if folder:
            savepath = os.path.join(folder, self.PLOTFOLDERNAME, f"{self.dye}.jpg")
            Path(os.path.dirname(savepath)).mkdir(parents=True, exist_ok=True) # ensure path exists
            plt.savefig(savepath, dpi=200)
            plt.close()


class LinearSingleStainSpectrumFit(SingleStainSpectrumFitBase):
    """Fit the spectrum of a single stain with a linear regression of every channel on the main channel.

    Each fluorescence channel is regressed on the main channel (raw intensities), by default with a robust linear
    model (``statsmodels.RLM``, Huber weights), like AutoSpill. The fit runs on construction unless `fit` is False.
    Written by `write` to the folder ``fit_linear``.

    Parameters
    ----------
    adata : anndata.AnnData
        Gated single-stain events, see `SingleStainSpectrumFitBase`.
    robust : bool, default True
        Use ``statsmodels.RLM``; otherwise ordinary least squares (``numpy.polyfit``, or ``scipy.optimize.curve_fit``
        without intercept).
    intercept : bool, default True
        Fit an intercept per channel; otherwise the intercept is 0.
    fit : bool, default True
        Run `fit` on construction.
    **kwargs
        Passed to `SingleStainSpectrumFitBase` (`linearity_tolerance_factor`, `asinh_cofactor`).

    Attributes
    ----------
    spectrum : pandas.DataFrame
        Shape (channel, 3), columns ``intercept`` (raw units), ``slope`` (relative to the main channel, 1 there) and
        ``r`` (Pearson correlation with the main channel). Set by `fit`.
    """
    PLOTFOLDERNAME = "plot_fit_linear"
    WRITEFOLDERNAME = "fit_linear"
    
    def __init__(self, adata, robust=True, intercept=True, fit=True, **kwargs):
        super().__init__(adata, **kwargs)
        self.robust = robust
        self.intercept = intercept
        
        if fit: self.fit()
    
    def fit_single_channel(self, channel, stain=None):
        """Fit one channel against the main channel (or against `stain`).

        Parameters
        ----------
        channel : str
            Channel to regress.
        stain : str, optional
            Channel to regress on. Default: the main channel.

        Returns
        -------
        list of float
            ``[intercept, slope, r]``; ``[0, 1, 1]`` if `channel` equals `stain`.
        """
        if stain is None: stain = self.stain
        if channel==stain: # Don't fit main channel, normalised anyway.
            return [0.,1.,1.]
        
        with np.errstate(invalid='ignore', divide='ignore'):
            x = self.adata[:,stain].layers["raw"][:,0].copy()
            y = self.adata[:,channel].layers["raw"][:,0].copy()
            
            if self.intercept:
                if self.robust: fit = list(sm.RLM(y, add_constant(x)).fit().params)
                else:           fit = list(np.polyfit(x, y, 1)[::-1])
                return fit + [np.corrcoef(x, y)[0,1]]
            else:
                if self.robust: fit = list(sm.RLM(y, x).fit().params)
                else:           fit = list(curve_fit(lambda x, a: a*x, x, y)[0])
                return [0.] + fit + [np.corrcoef(x, y)[0,1]]
    
    def fit(self):
        """Fit all fluorescence channels and set ``self.spectrum``."""
        self.spectrum = pd.DataFrame(np.nan, index=self.channels, columns=["intercept", "slope", "r"])
        for channel in self.channels:
            self.spectrum.loc[channel] = self.fit_single_channel(channel)

    def write(self, folder):
        """Write ``self.spectrum`` as ``{folder}/{WRITEFOLDERNAME}/{dye}.csv`` (folder created if needed).

        `collect_single_fits` collects these files into one spectra matrix.
        """
        Path(os.path.dirname(os.path.join(folder, self.WRITEFOLDERNAME, ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
        self.spectrum.to_csv(os.path.join(folder, self.WRITEFOLDERNAME, f"{self.dye}.csv"))
    
    
    def fit_NxN(self):
        """Fit every channel against every other channel; slow for many channels (spectral flow).

        Requires `fit` first. Sets pandas.DataFrame attributes of shape (channel, channel), entry ``[j, i]`` for channel
        j against channel i: ``NxN_slope`` and ``NxN_intercept`` (regression of j on i), ``NxN_slope_T`` and
        ``NxN_intercept_T`` (regression of i on j, inverted to a line of j against i), ``NxN_slope_imp`` and
        ``NxN_intercept_imp`` (the line implied by the fitted spectrum).
        """
        self.NxN_intercept = pd.DataFrame(np.nan, index=self.channels, columns=self.channels)
        self.NxN_slope = pd.DataFrame(np.nan, index=self.channels, columns=self.channels)
        
        for channeli in self.channels:
            for channelj in self.channels:
                meani = self.adata[:,channeli].layers["raw"].mean()
                meanj = self.adata[:,channelj].layers["raw"].mean()
                fit = self.fit_single_channel(channelj, channeli)
                self.NxN_intercept.loc[channelj, channeli] = fit[0]
                self.NxN_slope.loc[channelj, channeli] = fit[1]
        
        self.NxN_slope_T = 1/self.NxN_slope.T
        self.NxN_intercept_T = - self.NxN_intercept.T / self.NxN_slope.T
        
        slope, intercept = self.spectrum["slope"].to_numpy(), self.spectrum["intercept"].to_numpy()
        self.NxN_slope_imp = pd.DataFrame(slope[:,None] / slope[None,:],
                                          index=self.channels, columns=self.channels)
        self.NxN_intercept_imp = pd.DataFrame(- slope[:,None] / slope[None,:] * intercept[None,:] + intercept[:,None],
                                          index=self.channels, columns=self.channels)
    
    def plot_NxN(self, savepath="", clipNchannel=None, channels=None):
        """Plot every pair of channels with the three lines of `fit_NxN`.

        Requires `fit_NxN` first.

        Parameters
        ----------
        savepath : str, default ""
            If given, save the figure there and close it.
        clipNchannel : int, optional
            Only plot the first `clipNchannel` fluorescence channels. Ignored if `channels` is given.
        channels : list of str, optional
            Channels to plot (fluorescence channels, in any order). Default: all fluorescence channels (or the first
            `clipNchannel`).
        """
        if channels is None:
            channels = self.channels if clipNchannel is None else self.channels[:clipNchannel]
        N = len(channels)

        data = self.adata[:,list(channels)].layers["raw"]
        cmap = get_cmap(["fit","fit_T","fit_main"])

        fig = plt.figure(constrained_layout=True, figsize=(6*N, 6*N))
        spec = gridspec.GridSpec(ncols=N, nrows=N, figure=fig)
        for i in range(N):
            for j in range(N):
                ax = fig.add_subplot(spec[i, j])
                ax.grid()
                x, y = data[:,i], data[:,j]
                ax.scatter(x, y, s=.1, color="black")
                ax.set_xlabel(f"{channels[i]}", size=22)
                ax.set_ylabel(f"{channels[j]}", size=22)
                
                xlim, ylim = ax.get_xlim(), ax.get_ylim()

                slope = self.NxN_slope.loc[channels[j], channels[i]]
                intercept = self.NxN_intercept.loc[channels[j], channels[i]]
                xord = np.sort(x)
                ax.plot(xord, xord * slope + intercept, color=cmap["fit"], linewidth=4, alpha=.8)

                slope = self.NxN_slope_T.loc[channels[j], channels[i]]
                intercept = self.NxN_intercept_T.loc[channels[j], channels[i]]
                xord = np.sort(x)
                ax.plot(xord, xord * slope + intercept, color=cmap["fit_T"], linewidth=4, alpha=.8)

                slope = self.NxN_slope_imp.loc[channels[j], channels[i]]
                intercept = self.NxN_intercept_imp.loc[channels[j], channels[i]]
                xord = np.sort(x)
                ax.plot(xord, xord * slope + intercept, color=cmap["fit_main"], linewidth=4, alpha=.8)
                
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                if i==j:
                    ax.legend(handles=cmap_to_legendhandles(cmap), loc="upper left")

        if savepath:
            plt.savefig(savepath, dpi=200)
            plt.close()

class OutliersafeLinearSingleStainSpectrumFit(LinearSingleStainSpectrumFit):
    """`LinearSingleStainSpectrumFit` after removing outlier events with a local outlier factor.

    Before the fit, removes events that ``sklearn.neighbors.LocalOutlierFactor`` flags as outliers either in the
    other fluorescence channels (z-scored with the mean and standard deviation of the data clipped to their 5th-95th
    percentiles) or in the main channel alone (`filter_outlier`). Warns (`SpilloverFitWarning`) if more than 4%
    (``WARN_OUTLIER_SHARE``) of the events are removed. Written by `write` to the folder ``fit_linearos``.

    Parameters
    ----------
    adata : anndata.AnnData
        Gated single-stain events, see `SingleStainSpectrumFitBase`.
    neighbors_factor : int, default 100
        The local outlier factor uses ``max(N_events // neighbors_factor, 5)`` neighbours.
    **kwargs
        Passed to `LinearSingleStainSpectrumFit` (`robust`, `intercept`) and `SingleStainSpectrumFitBase`.
    """
    PLOTFOLDERNAME = "plot_fit_linearos"
    WRITEFOLDERNAME = "fit_linearos"
    WARN_OUTLIER_SHARE = .04
    
    def __init__(self, adata, neighbors_factor=100, **kwargs):
        super().__init__(adata, fit=False, **kwargs)
        self.neighbors_factor = neighbors_factor
        self.filter_outlier()
        self.fit()
    
    def _lof_n_neighbors(self):
        """Number of neighbours for LocalOutlierFactor: N_events // neighbors_factor, at least 5."""
        return max(self.adata.shape[0]//self.neighbors_factor, 5)
    
    def filter_outlier(self):
        """Remove outlier events from ``self.adata`` (in place), see the class docstring."""
        x = self.adata[:,[self.stain]].layers["raw"]
        y = self.adata[:,self.channels].layers["raw"]
        y = y[:,np.arange(y.shape[1])!=self.channels.index(self.stain)]
        y_clipped = np.clip(y, a_min=np.percentile(y, 5, axis=0, keepdims=True), a_max=np.percentile(y, 95, axis=0, keepdims=True))
        mean, std = y_clipped.mean(0, keepdims=True), y_clipped.std(0, keepdims=True)
        y = (y - mean) / std
        
        lof = LocalOutlierFactor(n_neighbors=self._lof_n_neighbors())
        mask = lof.fit_predict(y)==1
        lof = LocalOutlierFactor(n_neighbors=self._lof_n_neighbors())
        mask = mask & (lof.fit_predict(x)==1)
        if mask.mean()<1-self.WARN_OUTLIER_SHARE:
            warnings.warn(f"Detected outlier fraction is {1-mask.mean():.3f} for {self.dye}!", SpilloverFitWarning)
        self.adata = self.adata[mask].copy()
    
    def filter_outlier_residuals(self):
        """Remove events whose residuals to the current fit are outliers (local outlier factor); not used by the class.

        Requires ``self.spectrum``. Filtering on the residuals keeps the events close to the fitted line, so a fit
        repeated afterwards looks better than it is; `filter_outlier` is used instead.
        """
        x = self.adata[:,[self.stain]].layers["raw"]
        y = self.adata[:,self.channels].layers["raw"]
        residuals = y - (x * self.spectrum["slope"].to_numpy()[None] + self.spectrum["intercept"].to_numpy()[None])
        residuals = residuals[:,np.arange(residuals.shape[1])!=self.channels.index(self.stain)]
        residuals = (residuals - residuals.mean(0, keepdims=True)) / residuals.std(0, keepdims=True)
        
        lof = LocalOutlierFactor(n_neighbors=self._lof_n_neighbors())
        mask = lof.fit_predict(residuals)==1
        if mask.mean()<1-self.WARN_OUTLIER_SHARE:
            warnings.warn(f"Detected outlier fraction is {1-mask.mean():.3f} for {self.dye}!", SpilloverFitWarning)
        self.adata = self.adata[mask].copy()



class TorchSpectrumFitter(Module):
    """Fit a single-stain spectrum by gradient descent on a one-factor model of every event.

    Models every event as ``data[e, c] ~ staining[e] * spectrum[c] + offset[c]``: one staining intensity per event,
    a spectrum and an offset per channel, all fitted jointly. The spectrum is encoded as ``exp`` of its parameter
    (strictly positive); the staining and the offset as ``sinh`` of their parameters (any sign). The spectrum and the
    staining are only determined up to a common factor; `OutliersafeTorchSingleStainSpectrumFit` normalises the
    spectrum to the main channel.

    Initialisation: staining ``arcsinh`` of the data in the channel with the highest mean, spectrum and offset from
    a line fit of every channel against that channel (``numpy.polyfit``).

    Parameters
    ----------
    data : numpy.ndarray
        Raw intensities, shape (events, channel).
    channels : list of str
        Channel names of the columns of `data`.
    device : str, default "cpu"
        Torch device.
    criterion : {'mse', 'L1'}, default 'mse'
        Loss between the data and the reconstruction (``torch.nn.MSELoss`` or ``torch.nn.L1Loss``), on the raw scale.
    debug : bool, default False
        Show a progress bar in `fit` and record the spectrum and offset every 4 steps (for `plot_parameters`).

    Raises
    ------
    ValueError
        If `criterion` is not 'mse' or 'L1'.
    """
    def __init__(self, data, channels, device="cpu", criterion="mse", debug=False):
        super().__init__()
        self.device = device
        self.channels = list(channels)
        self.debug = debug
        
        self.data_input = data
        
        # data to fit
        self.data = torch.tensor(data, dtype=torch.float32, device=device)

        # spectrum
        self.spectrum_log = Parameter(torch.zeros(data.shape[1], dtype=torch.float32, device=device), requires_grad=True)
        # offset
        self.offset_asinh = Parameter(torch.zeros(data.shape[1], dtype=torch.float32, device=device, requires_grad=True))
        # per bead staining intensity
        #self.staining_log = Parameter(torch.tensor(np.log(np.clip(data.max(1), 0, None)), dtype=torch.float32, device=device, requires_grad=True))
        self.staining_log = Parameter(torch.tensor(np.arcsinh(data[:,data.mean(0).argmax()]), dtype=torch.float32, device=device, requires_grad=True))

        # use mean squared error loss
        if criterion=="mse":
            self.criterion = torch.nn.MSELoss()
        elif criterion=="L1":
            self.criterion = torch.nn.L1Loss()
        else:
            raise ValueError(f"Criterion {criterion} not available!")
        
        # initialise with simple polynomial fit for faster convergence
        self.init_polyfit()
    
    def init_polyfit(self):
        """Initialise spectrum and offset from a line fit of every channel against the channel with the highest mean."""
        slope, intercept = np.polyfit(self.data_input[:,self.data_input.mean(0).argmax()], self.data_input, 1)
        
        self.spectrum_log = Parameter(torch.tensor(np.log(np.clip(slope, 1e-14, None)), dtype=torch.float32, device=self.device), requires_grad=True)
        self.offset_asinh = Parameter(torch.tensor(np.arcsinh(intercept), dtype=torch.float32, device=self.device, requires_grad=True))
    
    @property
    def spectrum(self):
        """Spectrum, torch.Tensor of shape (channel,): ``exp`` of the parameter, strictly positive (not normalised)."""
        return self.spectrum_log.exp() #(self.spectrum_log-self.spectrum_log.max()).exp()
    
    @property
    def offset(self):
        """Offset per channel (raw units), torch.Tensor of shape (channel,): ``sinh`` of the parameter, any sign."""
        return (self.offset_asinh).sinh()
    
    @property
    def staining(self):
        """Staining intensity per event, torch.Tensor of shape (events,): ``sinh`` of the parameter, any sign."""
        return self.staining_log.sinh()
    
    @property
    def reconstruction(self, subset=slice(None)):
        """Reconstructed data ``staining * spectrum + offset``, torch.Tensor of shape (events, channel)."""
        return self.staining[:, None] * self.spectrum[None, :] + self.offset[None, :]
    
    def get_loss(self):
        """Loss (`criterion`) between the data and the reconstruction, a scalar torch.Tensor."""
        return self.criterion(self.data, self.reconstruction).mean()
    
    def fit(self, Niter=5000, lr=1e-3, lr_offset=1e-2,  warmup_offset=0):
        """Fit all parameters, using all events in every step.

        Staining and spectrum are fitted with Adam, the offset with RMSprop (momentum 0.4). No convergence check:
        inspect `plot_loss`. The loss of every step is appended to ``self.losshistory``.

        Parameters
        ----------
        Niter : int, default 5000
            Number of steps.
        lr : float, default 1e-3
            Adam learning rate for the staining and spectrum parameters.
        lr_offset : float, default 1e-2
            RMSprop learning rate for the offset parameter.
        warmup_offset : int, default 0
            If > 0, the offset learning rate rises linearly from ``lr_offset / warmup_offset`` to `lr_offset` over the
            first `warmup_offset` steps.
        """
        self.losshistory = []
        if self.debug:
            self.history_spectrum = []
            self.history_offset = []
        optimizer = torch.optim.Adam([self.staining_log, self.spectrum_log], lr=lr)
        optimizer_offset = torch.optim.RMSprop([self.offset_asinh], lr=lr_offset if warmup_offset==0 else 0., momentum=.4)
        
        pbar = tqdm(range(Niter)) if self.debug else range(Niter)
        for i in pbar:
            if i<warmup_offset:
                optimizer_offset.param_groups[0]["lr"] = lr_offset * (i+1) / warmup_offset
            
            optimizer.zero_grad()
            optimizer_offset.zero_grad()
            loss = self.get_loss()
            loss.backward()
            optimizer.step()
            optimizer_offset.step()
            
            self.losshistory.append(loss.item())
            
            if self.debug and i%4==0:
                self.history_spectrum.append(self.spectrum.detach().cpu().numpy())
                self.history_offset.append(self.offset.detach().cpu().numpy())
    
    def plot_loss(self):
        """Plot the loss per step (log scale)."""
        plt.plot(self.losshistory)
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Loss")
    
    def plot_parameters(self):
        """Plot the spectrum and the offset during the fit; requires ``debug=True``."""
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].plot(np.arange(len(self.history_spectrum))[:,None]*4, np.asarray(self.history_spectrum))
        ax[0].set_yscale("log")
        ax[0].set_title("Spectrum")
        ax[1].plot(np.arange(len(self.history_spectrum))[:,None]*4, np.asarray(self.history_offset))
        ax[1].set_title("Offset")
    
    def plot_NxN(self, savepath="", clipNchannel=None, channels=None):
        """Plot every pair of channels with the reconstructed line.

        Parameters
        ----------
        savepath : str, default ""
            If given, save the figure there and close it.
        clipNchannel : int, optional
            Only plot the first `clipNchannel` channels. Ignored if `channels` is given.
        channels : list of str, optional
            Channels to plot. Default: all channels (or the first `clipNchannel`).
        """
        if channels is None:
            channels = self.channels if clipNchannel is None else self.channels[:clipNchannel]
        N = len(channels)

        data = self.data.detach().cpu().numpy()
        data_rec = self.reconstruction.detach().cpu().numpy()
        cmap = get_cmap(["fit_SGD"])

        fig = plt.figure(constrained_layout=True, figsize=(6*N, 6*N))
        spec = gridspec.GridSpec(ncols=N, nrows=N, figure=fig)
        for i, ci in enumerate([self.channels.index(c) for c in channels]):
            for j, cj in enumerate([self.channels.index(c) for c in channels]):
                ax = fig.add_subplot(spec[i, j])
                ax.grid()
                x, y = data[:,ci], data[:,cj]
                ax.scatter(x, y, s=.1, color="black")
                ax.set_xlabel(f"{channels[i]}", size=22)
                ax.set_ylabel(f"{channels[j]}", size=22)

                x, y = np.sort(data_rec[:,ci]), np.sort(data_rec[:,cj])
                ax.plot(x, y, color=cmap["fit_SGD"], linewidth=4)

                if i==j:
                    ax.legend(handles=cmap_to_legendhandles(cmap), loc="upper left")

        if savepath:
            plt.savefig(savepath, dpi=200)
            plt.close()
    
    def plot_reconstruction_error(self):
        """Plot histograms of the reconstruction error (data minus reconstruction, raw units) per channel."""
        fig, ax = plt.subplots(figsize=(10,5))
        error = (self.data- self.reconstruction).detach().cpu().numpy()
        mean, std = error.mean(), error.std()
        bins = np.linspace(mean-5*std, mean+5*std, 200)
        for i in range(self.data.shape[1]):
            val = np.clip(error[:,i], a_min=bins.min(), a_max=bins.max())
            plt.hist(val, bins=bins, density=True, fill=False, histtype="step");
        plt.yscale("log")

class OutliersafeTorchSingleStainSpectrumFit(OutliersafeLinearSingleStainSpectrumFit):
    """`OutliersafeLinearSingleStainSpectrumFit` with a joint fit of all channels by gradient descent.

    After the outlier removal of `OutliersafeLinearSingleStainSpectrumFit`, fits all fluorescence channels jointly
    with `TorchSpectrumFitter` (default settings: 5000 steps, MSE loss on the raw scale) instead of one regression per
    channel. The spectrum is normalised to 1 in the main channel; the intercepts are those of every channel against
    the main channel implied by the fitted offsets. Written by `write` to the folder ``fit_torchos``.

    Parameters
    ----------
    *args
        Passed to `OutliersafeLinearSingleStainSpectrumFit` (`adata`, ...).
    device : str, default "cuda:0"
        Torch device; the CPU is used (without warning) if CUDA is not available.
    **kwargs
        Passed to `OutliersafeLinearSingleStainSpectrumFit`.

    Attributes
    ----------
    spectrum : pandas.DataFrame
        Shape (channel, 2), columns ``slope`` and ``intercept`` (no ``r``).
    tsfitter : TorchSpectrumFitter
        The fitted model.
    """
    PLOTFOLDERNAME = "plot_fit_torchos"
    WRITEFOLDERNAME = "fit_torchos"
    
    def __init__(self, *args, device="cuda:0", **kwargs):
        self.device = device if torch.cuda.is_available() else "cpu"
        super().__init__(*args, **kwargs)
    
    def fit(self):
        """Fit the spectrum with `TorchSpectrumFitter` and set ``self.spectrum`` (slope and intercept per channel)."""
        self.spectrum = pd.DataFrame(np.nan, index=self.channels, columns=["slope"])
        self.tsfitter = TorchSpectrumFitter(self.adata[:,self.channels].layers["raw"], self.channels, device=self.device)
        self.tsfitter.fit()
        spectrum = self.tsfitter.spectrum.detach().cpu().numpy()
        offset = self.tsfitter.offset.detach().cpu().numpy()
        istain = self.channels.index(self.stain)
        self.spectrum["slope"] = spectrum/spectrum[istain]
        # intercept of each channel against the stain channel (as in the other fits): offset_c - slope_c * offset_stain
        self.spectrum["intercept"] = offset - self.spectrum["slope"].to_numpy() * offset[istain]
    

class PosNegSingleStainSpectrumFit(SingleStainSpectrumFitBase):
    """Fit the spectrum of a single stain from the medians of the positive and the negative population.

    A two-component Gaussian mixture on ``arcsinh(x / asinh_cofactor)`` of the main channel splits the events into
    negative and positive (the component with the higher mean). The slope of every channel is the difference of the
    positive and negative medians (raw), divided by that difference in the main channel; the intercept is
    ``median_negative - slope * median_negative[main channel]``. The fit runs on construction; the mixture has no fixed
    random state. Written by `write` to the folder ``fit_posneg``.

    Parameters
    ----------
    adata : anndata.AnnData
        Gated single-stain events, see `SingleStainSpectrumFitBase`.
    **kwargs
        Passed to `SingleStainSpectrumFitBase` (`linearity_tolerance_factor`, `asinh_cofactor`).

    Attributes
    ----------
    spectrum : pandas.DataFrame
        Shape (channel, 4), columns ``median_positive``, ``median_negative`` (raw), ``slope`` and ``intercept``.
    gmm : sklearn.mixture.GaussianMixture
        The fitted mixture. ``self.adata.obs['is_positive']`` holds the assignment of every event.
    """
    PLOTFOLDERNAME = "plot_fit_posneg"
    WRITEFOLDERNAME = "fit_posneg"
    
    def __init__(self, adata, **kwargs):
        super().__init__(adata, **kwargs)
        
        self.fit()
    
    def fit(self):
        """Split the events into positive and negative and set ``self.spectrum``, see the class docstring."""
        self.gmm = GaussianMixture(2)
        x = np.arcsinh(self.adata[:,self.stain].layers["raw"][:,0].copy() / self.asinh_cofactor)[:,None]
        label = self.gmm.fit_predict(x)
        label = np.argsort(self.gmm.means_[:,0])[label] # correct order
        is_pos_mask = label==1
        self.adata.obs["is_positive"] = is_pos_mask

        self.spectrum = pd.DataFrame(index=self.channels)
        self.spectrum["median_positive"] = np.median(self.adata[is_pos_mask,self.channels].layers["raw"], axis=0)
        self.spectrum["median_negative"] = np.median(self.adata[~is_pos_mask,self.channels].layers["raw"], axis=0)

        diff_all = self.spectrum["median_positive"] - self.spectrum["median_negative"]
        diff_main = self.spectrum.loc[self.stain, "median_positive"] - self.spectrum.loc[self.stain, "median_negative"]
        self.spectrum["slope"] = (diff_all) / (diff_main)
        self.spectrum["intercept"] = self.spectrum["median_negative"] - self.spectrum["slope"] * self.spectrum.loc[self.stain, "median_negative"]

    def write(self, folder):
        """Write ``self.spectrum`` as ``{folder}/{WRITEFOLDERNAME}/{dye}.csv`` (folder created if needed).

        `collect_single_fits` collects these files into one spectra matrix; `collect_peakpos_fromposneg` reads the
        positive medians.
        """
        Path(os.path.dirname(os.path.join(folder, self.WRITEFOLDERNAME, ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
        self.spectrum.to_csv(os.path.join(folder, self.WRITEFOLDERNAME, f"{self.dye}.csv"))


def collect_single_fits(folder, inname, outname):
    """Collect the single-stain fits of one method, as written by the fitters' ``write``, into one spectra matrix.

    Reads every ``{folder}/{inname}/{dye}.csv`` and writes the slopes as ``{folder}/{outname}``.
    `SpilloverMeasurement.from_custom` reads the files ``spectra_linear.csv``, ``spectra_linearos.csv``,
    ``spectra_posneg.csv`` and ``spectra_torchos.csv``, so use these as `outname` for the folders ``fit_linear``,
    ``fit_linearos``, ``fit_posneg`` and ``fit_torchos``.

    Parameters
    ----------
    folder : str
        Folder of the measurement (the `collect_data_path` of the collect functions).
    inname : str
        Subfolder with the single fits, e.g. ``"fit_linearos"``.
    outname : str
        File name of the output .csv, e.g. ``"spectra_linearos.csv"``: shape (channel, dye), columns the dyes (file
        names without .csv) in sorted order.

    Examples
    --------
    >>> collect_single_fits(collect_data_path, "fit_linearos", "spectra_linearos.csv")
    """
    files = list(filter(lambda x: x.endswith(".csv"), os.listdir(os.path.join(folder, inname, ""))))
    dyes = [file[:-4] for file in files]
    
    spectra = pd.concat([pd.read_table(os.path.join(folder, inname, file), sep=",", index_col=0)["slope"] for file in files], axis=1)
    spectra.columns = dyes
    spectra = spectra[sorted(dyes)]
    
    spectra.to_csv(os.path.join(folder, outname))

def collect_peakpos_fromposneg(folder):
    """Add the positive peak position of every dye, from the `PosNegSingleStainSpectrumFit` fits, to panel_meta.csv.

    Reads ``{folder}/fit_posneg/{dye}.csv`` and writes the largest positive median over all channels (raw, rounded to
    an integer; normally the main channel) as column ``positive_peak`` into ``{folder}/panel_meta.csv`` (NaN for dyes
    without a fit). `SpilloverCollection.plot_peak_position` uses this column.

    Parameters
    ----------
    folder : str
        Folder of the measurement, with ``panel_meta.csv`` and ``fit_posneg/``.
    """
    meta = pd.read_table(os.path.join(folder, "panel_meta.csv"), sep=",", index_col=0)
    index = meta.index
    meta.index = meta["dye"]
    meta["positive_peak"] = np.nan

    files = list(filter(lambda x: x.endswith(".csv"), os.listdir(os.path.join(folder, "fit_posneg", ""))))
    dyes = [file[:-4] for file in files]

    peakpos = [pd.read_table(os.path.join(folder, "fit_posneg", file), sep=",", index_col=0)["median_positive"].max() for file in files]
    meta.loc[dyes, "positive_peak"] = np.round(peakpos, 0)

    meta.index = index
    meta.to_csv(os.path.join(folder, "panel_meta.csv"))
