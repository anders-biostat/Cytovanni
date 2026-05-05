import numpy as np
from datetime import datetime
import anndata
import warnings

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import colorsys


class textcolor:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    BLACK = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def toBOLD(cls, text):
        return cls.BOLD + text + cls.END
    
    @classmethod
    def toCOLOR(cls, text, color):
        return getattr(cls, color) + text + cls.END

def get_scanpy_palette_20():
    """ Default 20 color palette of Scanpy.
        Copy code to not add dependency just for this.
    """
    from matplotlib import cm, colors
    
    vega_10 = list(map(colors.to_hex, cm.tab10.colors))
    vega_10_scanpy = vega_10.copy()
    vega_10_scanpy[2] = "#279e68"  # green
    vega_10_scanpy[4] = "#aa40fc"  # purple
    vega_10_scanpy[8] = "#b5bd61"  # kakhi

    # default matplotlib 2.0 palette
    # see 'category20' on https://github.com/vega/vega/wiki/Scales#scale-range-literals
    vega_20 = list(map(colors.to_hex, cm.tab20.colors))

    # reorderd, some removed, some added
    vega_20_scanpy = [
        # dark without grey:
        *vega_20[0:14:2],
        *vega_20[16::2],
        # light without grey:
        *vega_20[1:15:2],
        *vega_20[17::2],
        # manual additions:
        "#ad494a",
        "#8c6d31",
    ]
    vega_20_scanpy[2] = vega_10_scanpy[2]
    vega_20_scanpy[4] = vega_10_scanpy[4]
    vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey
    # TODO: also replace pale colors if necessary
    
    return vega_20_scanpy

palette_20 = get_scanpy_palette_20()

# zeileis_28 from scanpy
palette_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    # these last ones were added:
    "#7f7f7f",
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]

def get_hex_distinct(N):
    """ Return N distinct colors.
    """
    if N<=20:
        return palette_20[:N]
    elif N<=28:
        return palette_28[:N]
    else:
        np.random.seed(0)
        hues = np.linspace(0, 1, N+1)[np.random.permutation(N)]
        cols = np.asarray([colorsys.hsv_to_rgb(h,np.random.uniform(0.5,1),1) for h in hues])
        hexs = [matplotlib.colors.to_hex(c) for c in cols]
        return hexs

def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' This converts a given wavelength of light to an 
        approximate RGB color value. The wavelength must be given
        in nanometers in the range from 380 nm through 750 nm
        (789 THz through 400 THz).
        Based on code by Dan Bruton
        http://www.physics.sfasu.edu/astro/color/spectra.html
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))

def wavelength_to_hex(wavelength):
    """ wavelength_to_rgb and convert to hex
    """
    return matplotlib.colors.to_hex(np.asarray(wavelength_to_rgb(wavelength))/255)


def get_cmap(labels, ishex=True):
    """ Get color dict for all unique labels.
    """
    uns = np.unique(labels)
    return {k:v if ishex else matplotlib.colors.to_rgb(v) for k, v in zip(uns, get_hex_distinct(len(uns)))}

MARKERS = ['o', 's', 'v', '^', '<', '>', 'p', 'P', 'X', 'D']

def get_cmap_and_markers(keys):
    cmap = get_cmap(keys)
    marker_cycle = {k: MARKERS[i % len(MARKERS)] for i, k in enumerate(cmap)}
    return cmap, marker_cycle

def cmap_to_legendhandles(cmap, markersize=20, include_label=True, outline=False):
    """ Colormap frpm get_cmap to handles for legend.
    """
    return [Line2D([0], [0], label=k if include_label else '', marker='.', markersize=markersize, markeredgecolor="black" if outline else v, markerfacecolor=v, linestyle='')
               for k, v in cmap.items()]

def cmap_to_legendhandles_wmarker(cmap, markermap, markersize=20, include_label=True, outline=False):
    return [Line2D([0], [0], label=k if include_label else '', marker=markermap[k], markersize=markersize,
                   markeredgecolor="black" if outline else v,
                   markerfacecolor=v, linestyle='')
            for k, v in cmap.items()]

def labelscolors_to_legendhandles(labels, colors, markersize=20):
    """ Label names and associated colors to handles for legend, nicer than default if point size very small etc.
    """
    return [Line2D([0], [0], label=k, marker='.', markersize=markersize, markeredgecolor=v, markerfacecolor=v, linestyle='')
               for k, v in zip(labels, colors)]

def unrolled_subplots(N, Ncol=4, elsize=(6,4)):
    """ Generate unrolled plt.subplots with N plots
    """
    Ncol = min(N, Ncol)
    Nrow = int(np.ceil(N/Ncol))
    Nax = Nrow*Ncol
    figsize = np.asarray(elsize)*np.asarray([Ncol, Nrow])
    fig, ax = plt.subplots(Nrow, Ncol, figsize=figsize)
    if Nrow==1 and Ncol==1:
        ax_unr = [ax]
    elif Nrow==1 or Ncol==1:
        ax_unr = ax
    else:
        ax_unr = []
        for i in range(Nax):
            ax_ = ax[i//Ncol, i%Ncol]
            if i<N: ax_unr.append(ax_)
            else:   ax_.axis("off")
    
    return fig, ax_unr

def printwtime(text):
    """ Print text with current time.
    """
    print(datetime.now().strftime("%H:%M:%S"),"-",text)

def silent_log(x):
    """ Numpy logarithm, but without any errors if nan etc.
    """
    with np.errstate(invalid='ignore', divide = 'ignore'):
        return np.log(x)

def label_axis_ArcSinh(ax, cofactor, onx=True, minpower=2, mark_cofactor=False):
    """ Takes axis on an ArcSinh scale, and relabels it such that positive and negative powers of ten are labeled.
        Labels all powers of 10 above 10^1 and below -10^1 as major ticks, excluding powers below minpower.
        
        Fix zero formatting!
        
        :param ax: axis. The axis on which to work.
        
        :param cofactor: float. The cofactor of the ArcSinh transformation.
        
        :param onx: True. Whether to work on x axis, or y axis.
        
        :param minpower: float. The max power of 10 to show.
        
        :param mark_cofactor: bool. If true, also mark the used cofactor on the axis.
    """
    axlim = ax.get_xlim() if onx else ax.get_ylim()
    axlim_real = np.sinh(axlim)*cofactor
    available_powers = []
    if axlim_real[0]<0:
        available_powers.extend(list(-np.arange(np.floor(np.log10(max(-axlim_real[1],10.))), np.ceil(np.log10(-axlim_real[0]))+1)[::-1]))
    if axlim_real[0]<=0 and axlim_real[1]>=0:
        available_powers.append(0)
    if axlim_real[1]>0:
        available_powers.extend(list(np.arange(np.floor(np.log10(max(axlim_real[0], 10.))), np.ceil(np.log10(axlim_real[1]))+1)))
    else:
        available_powers_neg = []
    available_powers = np.asarray(available_powers)
    available_powers = available_powers[(np.abs(available_powers)>=minpower)|(available_powers==0)]
    
    def get_minor(a, b):
        if (abs(a-b)==1) and min(a,b)<0:
            return -np.log10(10**min(abs(a),abs(b)) * np.arange(2,10))
        if (abs(a-b)==1) and min(a,b)>0:
            return np.log10(10**min(a,b) * np.arange(2,10))
        else:
            return min(a,b) + np.arange(min(abs(a),abs(b)), max(abs(a),abs(b))+1)[1:-1]
        
    minor_powers = np.hstack([get_minor(available_powers[i], available_powers[i+1]) for i in range(len(available_powers)-1)])

    def power_to_pos(p):
        if p>0: return 10**p
        elif p==0: return 0
        else: return -10**(-p)
    available_powers_pos = np.arcsinh(np.asarray([power_to_pos(p) for p in available_powers]) / cofactor)
    minor_powers_pos = np.arcsinh(np.asarray([power_to_pos(p) for p in minor_powers]) / cofactor)

    def format_power(p):
        if p==0: return r"$0$"
        elif p>0: return fr"$10^{p:.0f}$"
        else: return fr"$-10^{-p:.0f}$"
    labels=[format_power(p) for p in available_powers]
    
    if onx:
        ax.set_xticks(available_powers_pos, labels=labels) #, va='bottom')
        ax.set_xticks(minor_powers_pos, minor=True)
        ax.set_xlim(axlim)
        if mark_cofactor:
            ax.axvline( np.arcsinh(1), 0, .05, linewidth=1, color="black")
            ax.axvline(np.arcsinh(-1), 0, .05, linewidth=1, color="black")
    else:
        ax.set_yticks(available_powers_pos, labels=labels)
        ax.set_yticks(minor_powers_pos, minor=True)
        ax.set_ylim(axlim)
        if mark_cofactor:
            ax.axhline( np.arcsinh(1), 0, .05, linewidth=1, color="black")
            ax.axhline(np.arcsinh(-1), 0, .05, linewidth=1, color="black")

def anndata_concat(adatas, make_unique=False, join="inner", merge=None, permute=False):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        adata = anndata.concat(adatas, join=join, merge=merge)
    if make_unique:
        adata.obs_names_make_unique()
    if permute:
        adata = adata[adata.obs.index[np.random.permutation(adata.shape[0])]].copy()
    return adata

def anndata_subsample(adata, N=20000, copy=True):
    """ Subsample adata.
    """
    if adata.shape[0]<=N:
        return adata
    sind = adata.obs.index[np.random.permutation(adata.shape[0])[:int(N)]]
    ad = adata[sind].copy()
    return ad

def df_subsample(df, N=20000, copy=True):
    """ Subsample pd DataFrame.
    """
    if df.shape[0]<=N:
        return df
    sind = df.index[np.random.permutation(df.shape[0])[:int(N)]]
    dfs = df.loc[sind].copy()
    return dfs

from collections import Counter

def subsample_label_indices(labels, max_multiple=None):
    """ Label-aware subsampling, fixes the maximum ratio between the rarest and the most common label
    """
    labels = np.array(labels)
    label_counts = Counter(labels)
    rarest_count = min(label_counts.values())
    
    subsample_indices = []
    for label in label_counts:
        label_indices = np.where(labels == label)[0]
        if label_counts[label] > rarest_count:
            if max_multiple is None:
                chosen_indices = np.random.choice(label_indices, rarest_count, replace=False)
            else:
                if label_counts[label] > max_multiple*rarest_count:
                    chosen_indices = np.random.choice(label_indices, int(max_multiple*rarest_count), replace=False)
                else:
                    chosen_indices = label_indices
        else:
            chosen_indices = label_indices
        subsample_indices.extend(chosen_indices)
    
    return np.random.permutation(np.array(subsample_indices))

def check_pulse_width_independence(adata, channel, layer="raw", plot=False):
    """ Check whether the pulse width of 'channel' is measured independently.
        Assume it is calculated if the correlation is above .999
    """
    A = adata[:,f"{channel}-A"].layers[layer][:,0].copy()
    H = adata[:,f"{channel}-H"].layers[layer][:,0].copy()
    W = adata[:,f"{channel}-W"].layers[layer][:,0].copy()
    
    W_calc = A / H
    corr = np.corrcoef(W, W_calc)[0,1]

    if plot:
        plt.scatter(W_calc, W, s=.01)
        plt.xlabel(f"{channel}-A / {channel}-H", size=20)
        plt.ylabel(f"{channel}-W", size=20)
        plt.title(f"Correlation: {corr:.4f}")
        plt.show()

    return corr<.999

