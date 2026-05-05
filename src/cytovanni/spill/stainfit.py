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

from ..exceptions import SpilloverFitWarning
from ..utils import unrolled_subplots, get_cmap, cmap_to_legendhandles

def _find_max_channel(ad):
    """ Simple version of finding main channel, improve?
    """
    return ad.uns["cytoconfig"].channels_fluorescence[np.argmax(np.percentile(ad[:,ad.uns["cytoconfig"].channels_fluorescence].layers["raw"],
                                                                             95, axis=0))]

class SingleStainSpectrumFitBase():
    """ If 'channel' is in adata.uns and a valid channel, assumes regular flow.
        Otherwise assume spectral flow and fit/normalize to highest channel.
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
        """ Make sure used data is within linear range etc.
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
        """ Return fitted spectrum.
        """
        spectrum = self.spectrum[["slope"]]
        spectrum.columns = [self.dye]
        return spectrum

    def plot(self, folder="", s=.1):
        """ Plot fit across all channels.
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
    """ Like AutoSpill, fits a robust linear model to the data.
    """
    PLOTFOLDERNAME = "plot_fit_linear"
    WRITEFOLDERNAME = "fit_linear"
    
    def __init__(self, adata, robust=True, intercept=True, fit=True, **kwargs):
        super().__init__(adata, **kwargs)
        self.robust = robust
        self.intercept = intercept
        
        if fit: self.fit()
    
    def fit_single_channel(self, channel, stain=None):
        """ Fit slope for single channel.
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
        """ Fit full spectrum
        """
        self.spectrum = pd.DataFrame(np.nan, index=self.channels, columns=["intercept", "slope", "r"])
        for channel in self.channels:
            self.spectrum.loc[channel] = self.fit_single_channel(channel)

    def write(self, folder):
        """ Write fit into folder.
        """
        Path(os.path.dirname(os.path.join(folder, self.WRITEFOLDERNAME, ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
        self.spectrum.to_csv(os.path.join(folder, self.WRITEFOLDERNAME, f"{self.dye}.csv"))
    
    
    def fit_NxN(self):
        """ Fit all channels against each other, may take a bit for spectral data.
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
        """ Plot fit across all channel combinations in NxN raster.
        """
        if channels is None:
            channels = self.channels if clipNchannel is None else self.channels[:clipNchannel]
        N = len(channels)

        data = self.adata[:,self.channels].layers["raw"]
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

                slope = self.NxN_slope.loc[self.channels[j], self.channels[i]]
                intercept = self.NxN_intercept.loc[self.channels[j], self.channels[i]]
                xord = np.sort(x)
                ax.plot(xord, xord * slope + intercept, color=cmap["fit"], linewidth=4, alpha=.8)

                slope = self.NxN_slope_T.loc[self.channels[j], self.channels[i]]
                intercept = self.NxN_intercept_T.loc[self.channels[j], self.channels[i]]
                xord = np.sort(x)
                ax.plot(xord, xord * slope + intercept, color=cmap["fit_T"], linewidth=4, alpha=.8)

                slope = self.NxN_slope_imp.loc[self.channels[j], self.channels[i]]
                intercept = self.NxN_intercept_imp.loc[self.channels[j], self.channels[i]]
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
    """ Fit LinearSingleStainSpectrumFit, remove outliers based on residuals, fit again.
    """
    PLOTFOLDERNAME = "plot_fit_linearos"
    WRITEFOLDERNAME = "fit_linearos"
    WARN_OUTLIER_SHARE = .04
    
    def __init__(self, adata, neighbors_factor=100, **kwargs):
        super().__init__(adata, fit=False, **kwargs)
        self.neighbors_factor = neighbors_factor
        self.filter_outlier()
        self.fit()
    
    def filter_outlier(self):
        """ Filter outliers based on sklearn LocalOutlierFactor on z-normed data, as well as only on the main channel.
        """
        x = self.adata[:,[self.stain]].layers["raw"]
        y = self.adata[:,self.channels].layers["raw"]
        y = y[:,np.arange(y.shape[1])!=self.channels.index(self.stain)]
        y_clipped = np.clip(y, a_min=np.percentile(y, 5, axis=0, keepdims=True), a_max=np.percentile(y, 95, axis=0, keepdims=True))
        mean, std = y_clipped.mean(0, keepdims=True), y_clipped.std(0, keepdims=True)
        y = (y - mean) / std
        
        lof = LocalOutlierFactor(n_neighbors=self.adata.shape[0]//self.neighbors_factor)
        mask = lof.fit_predict(y)==1
        lof = LocalOutlierFactor(n_neighbors=self.adata.shape[0]//self.neighbors_factor)
        mask = mask & (lof.fit_predict(x)==1)
        if mask.mean()<1-self.WARN_OUTLIER_SHARE:
            warnings.warn(f"Detected outlier fraction is {1-mask.mean():.3f} for {self.dye}!", SpilloverFitWarning)
        self.adata = self.adata[mask].copy()
    
    def filter_outlier_residuals(self):
        """ Filter outliers based on residuals and sklearn LocalOutlierFactor.
            Bad idea, artificially nice if it only keeps the outliers on the fitted line!
        """
        x = self.adata[:,[self.stain]].layers["raw"]
        y = self.adata[:,self.channels].layers["raw"]
        residuals = y - (x * self.spectrum["slope"].to_numpy()[None] + self.spectrum["intercept"].to_numpy()[None])
        residuals = residuals[:,np.arange(residuals.shape[1])!=self.channels.index(self.stain)]
        residuals = (residuals - residuals.mean(0, keepdims=True)) / residuals.std(0, keepdims=True)
        
        lof = LocalOutlierFactor(n_neighbors=self.adata.shape[0]//self.neighbors_factor)
        mask = lof.fit_predict(residuals)==1
        if mask.mean()<1-self.WARN_OUTLIER_SHARE:
            warnings.warn(f"Detected outlier fraction is {1-mask.mean():.3f} for {self.dye}!", SpilloverFitWarning)
        self.adata = self.adata[mask].copy()



class TorchSpectrumFitter(Module):
    """ Fit single stain spectrum using gradient descent.
        Assumes every event can be characterized by a single number that represents the staining intensity.
        Spectrum is assumed to be strictly positive, the staining intensity as well.
        
        Spectrum initialisation does not affect the result. (?)
        Staining intensity needs to be encoded as log to train properly.
        
        Better encoding for staining! Currently can't really switch the sign!
        Better offset fitting, currently not quite stable!
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
        """ Fit simple line and use parameters as initialization.
        """
        slope, intercept = np.polyfit(self.data_input[:,self.data_input.mean(0).argmax()], self.data_input, 1)
        
        self.spectrum_log = Parameter(torch.tensor(np.log(np.clip(slope, 1e-14, None)), dtype=torch.float32, device=self.device), requires_grad=True)
        self.offset_asinh = Parameter(torch.tensor(np.arcsinh(intercept), dtype=torch.float32, device=self.device, requires_grad=True))
    
    @property
    def spectrum(self):
        """ Spectrum, exp of parameter for range (0,inf).
        """
        return self.spectrum_log.exp() #(self.spectrum_log-self.spectrum_log.max()).exp()
    
    @property
    def offset(self):
        """ Offset value, simple sinh of parameter for range (inf,inf).
        """
        return (self.offset_asinh).sinh()
    
    @property
    def staining(self):
        """ Staining value, simple exp of parameter for range (0,inf).
        """
        return self.staining_log.sinh()
    
    @property
    def reconstruction(self, subset=slice(None)):
        """ Staining value times spectrum, plus some offset.
        """
        return self.staining[:, None] * self.spectrum[None, :] + self.offset[None, :]
    
    def get_loss(self):
        return self.criterion(self.data, self.reconstruction).mean()
    
    def fit(self, Niter=5000, lr=1e-3, lr_offset=1e-2,  warmup_offset=0):
        """ Fit all parameters of model.
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
        """ Plot loss.
        """
        plt.plot(self.losshistory)
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Loss")
    
    def plot_parameters(self):
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ax[0].plot(np.arange(len(self.history_spectrum))[:,None]*4, np.asarray(self.history_spectrum))
        ax[0].set_yscale("log")
        ax[0].set_title("Spectrum")
        ax[1].plot(np.arange(len(self.history_spectrum))[:,None]*4, np.asarray(self.history_offset))
        ax[1].set_title("Offset")
    
    def plot_NxN(self, savepath="", clipNchannel=None, channels=None):
        """ Plot fit across all channel combinations in NxN raster.
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
        fig, ax = plt.subplots(figsize=(10,5))
        error = (self.data- self.reconstruction).detach().cpu().numpy()
        mean, std = error.mean(), error.std()
        bins = np.linspace(mean-5*std, mean+5*std, 200)
        for i in range(self.data.shape[1]):
            val = np.clip(error[:,i], a_min=bins.min(), a_max=bins.max())
            plt.hist(val, bins=bins, density=True, fill=False, histtype="step");
        plt.yscale("log")

class OutliersafeTorchSingleStainSpectrumFit(OutliersafeLinearSingleStainSpectrumFit):
    """ OutliersafeLinearSingleStainSpectrumFit but simultaneous line fit using gradient descent.
        Add warning if running on CPU?
    """
    PLOTFOLDERNAME = "plot_fit_torchos"
    WRITEFOLDERNAME = "fit_torchos"
    
    def __init__(self, *args, device="cuda:0", **kwargs):
        self.device = device if torch.cuda.is_available() else "cpu"
        super().__init__(*args, **kwargs)
    
    def fit(self):
        self.spectrum = pd.DataFrame(np.nan, index=self.channels, columns=["slope"])
        self.tsfitter = TorchSpectrumFitter(self.adata[:,self.channels].layers["raw"], self.channels, device=self.device)
        self.tsfitter.fit()
        spectrum = self.tsfitter.spectrum.detach().cpu().numpy()
        self.spectrum["slope"] = spectrum/spectrum[self.channels.index(self.stain)]
    

class PosNegSingleStainSpectrumFit(SingleStainSpectrumFitBase):
    """ Fit positive/negative with GMM, use medians to get spectra.
    """
    PLOTFOLDERNAME = "plot_fit_posneg"
    WRITEFOLDERNAME = "fit_posneg"
    
    def __init__(self, adata, **kwargs):
        super().__init__(adata, **kwargs)
        
        self.fit()
    
    def fit(self):
        """ Simple slope fit from medians of positive/negative.
        """
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
        """ Write fit into folder.
        """
        Path(os.path.dirname(os.path.join(folder, self.WRITEFOLDERNAME, ""))).mkdir(parents=True, exist_ok=True) # ensure path exists
        self.spectrum.to_csv(os.path.join(folder, self.WRITEFOLDERNAME, f"{self.dye}.csv"))


def collect_single_fits(folder, inname, outname):
    """ Collect single fits as saved by .write of SpectrumFit into one spectra matrix.
    """
    files = list(filter(lambda x: x.endswith(".csv"), os.listdir(os.path.join(folder, inname, ""))))
    dyes = [file[:-4] for file in files]
    
    spectra = pd.concat([pd.read_table(os.path.join(folder, inname, file), sep=",", index_col=0)["slope"] for file in files], axis=1)
    spectra.columns = dyes
    spectra = spectra[sorted(dyes)]
    
    spectra.to_csv(os.path.join(folder, outname))

def collect_peakpos_fromposneg(folder):
    """ Collects position of positive peak from posneg fit, add to panel_meta.csv
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
