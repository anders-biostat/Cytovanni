"""Small helpers: colours and legends, axis labelling for arcsinh scales, subsampling, AnnData concatenation."""
import numpy as np
from datetime import datetime
import anndata
import warnings

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import colorsys


class textcolor:
    """ANSI escape codes for coloured and bold terminal output (used in the ``__repr__`` of several classes).

    The class attributes (``PURPLE``, ``CYAN``, ``DARKCYAN``, ``BLUE``, ``GREEN``, ``YELLOW``, ``RED``, ``WHITE``,
    ``BLACK``, ``BOLD``, ``UNDERLINE``) are the codes; ``END`` resets the formatting.
    """
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
        """Return `text` (str) wrapped in the codes for bold text."""
        return cls.BOLD + text + cls.END
    
    @classmethod
    def toCOLOR(cls, text, color):
        """Return `text` (str) wrapped in the codes for `color`, the name of a class attribute such as ``"RED"``."""
        return getattr(cls, color) + text + cls.END

def get_scanpy_palette_20():
    """Return scanpy's default 20-colour categorical palette.

    A copy of scanpy's code (matplotlib's tab20, reordered, with a few colours replaced), so that scanpy is not
    needed as a dependency.

    Returns
    -------
    list of str
        20 hex colours.
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

# Categorical palettes: scanpy's default 20 colours, and scanpy's zeileis_28 extended by 4 colours.
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
    """Return `N` distinct colours.

    Up to 20 colours come from `palette_20`, up to 28 from `palette_28`; for more, hues are spread evenly around the
    colour wheel in a random order with random saturation (seeded, so the result is reproducible).

    Parameters
    ----------
    N : int
        Number of colours.

    Returns
    -------
    list of str
        `N` hex colours.

    Notes
    -----
    For ``N > 28`` this calls ``np.random.seed(0)``, which resets numpy's global random state.
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
    '''Convert a wavelength of visible light to an approximate RGB colour.

    Based on code by Dan Bruton (http://www.physics.sfasu.edu/astro/color/spectra.html). Used e.g. to colour lasers
    and detector channels by their wavelength.

    Parameters
    ----------
    wavelength : float
        Wavelength in nm. Visible range 380-750 nm; outside of it the result is black.
    gamma : float, default 0.8
        Gamma correction applied to every component.

    Returns
    -------
    tuple of int
        (R, G, B), each in 0-255.
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
    """Convert a wavelength of visible light to an approximate colour as a hex string.

    Parameters
    ----------
    wavelength : float
        Wavelength in nm, see `wavelength_to_rgb`.

    Returns
    -------
    str
        Hex colour, e.g. ``'#a2ff00'`` for 550 nm.
    """
    return matplotlib.colors.to_hex(np.asarray(wavelength_to_rgb(wavelength))/255)


def get_cmap(labels, ishex=True):
    """Assign a distinct colour to every unique label.

    Parameters
    ----------
    labels : array-like
        Labels, may contain repetitions.
    ishex : bool, default True
        Return colours as hex strings; if False, as RGB tuples of floats in [0, 1].

    Returns
    -------
    dict
        Label to colour, for the sorted unique labels (colours from `get_hex_distinct`).
    """
    uns = np.unique(labels)
    return {k:v if ishex else matplotlib.colors.to_rgb(v) for k, v in zip(uns, get_hex_distinct(len(uns)))}

# Marker symbols cycled through by get_cmap_and_markers.
MARKERS = ['o', 's', 'v', '^', '<', '>', 'p', 'P', 'X', 'D']

def get_cmap_and_markers(keys):
    """Assign a distinct colour and a marker symbol to every unique key.

    Parameters
    ----------
    keys : array-like
        Labels, may contain repetitions.

    Returns
    -------
    cmap : dict
        Label to hex colour, as `get_cmap`.
    markers : dict
        Label to matplotlib marker symbol, cycling through 10 symbols in the order of `cmap`.

    See Also
    --------
    cmap_to_legendhandles_wmarker : legend handles for the result.
    """
    cmap = get_cmap(keys)
    marker_cycle = {k: MARKERS[i % len(MARKERS)] for i, k in enumerate(cmap)}
    return cmap, marker_cycle

def cmap_to_legendhandles(cmap, markersize=20, include_label=True, outline=False):
    """Create legend handles (dots) for a colour dict such as the output of `get_cmap`.

    Parameters
    ----------
    cmap : dict
        Label to colour.
    markersize : float, default 20
        Size of the dots in the legend.
    include_label : bool, default True
        Label the handles with the keys of `cmap`; if False, they are unlabelled.
    outline : bool, default False
        Draw a black edge around each dot.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Handles for ``ax.legend(handles=...)``, in the order of `cmap`.
    """
    return [Line2D([0], [0], label=k if include_label else '', marker='.', markersize=markersize, markeredgecolor="black" if outline else v, markerfacecolor=v, linestyle='')
               for k, v in cmap.items()]

def cmap_to_legendhandles_wmarker(cmap, markermap, markersize=20, include_label=True, outline=False):
    """Create legend handles with a colour and a marker symbol per label, e.g. from `get_cmap_and_markers`.

    Parameters
    ----------
    cmap : dict
        Label to colour.
    markermap : dict
        Label to matplotlib marker symbol; must contain every key of `cmap`.
    markersize : float, default 20
        Size of the markers in the legend.
    include_label : bool, default True
        Label the handles with the keys of `cmap`; if False, they are unlabelled.
    outline : bool, default False
        Draw a black edge around each marker.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Handles for ``ax.legend(handles=...)``, in the order of `cmap`.
    """
    return [Line2D([0], [0], label=k if include_label else '', marker=markermap[k], markersize=markersize,
                   markeredgecolor="black" if outline else v,
                   markerfacecolor=v, linestyle='')
            for k, v in cmap.items()]

def labelscolors_to_legendhandles(labels, colors, markersize=20):
    """Create legend handles (dots) from parallel lists of labels and colours.

    Useful when the plotted points are too small to be visible in the default legend.

    Parameters
    ----------
    labels : list of str
        Legend labels.
    colors : list
        Matplotlib colour of every label.
    markersize : float, default 20
        Size of the dots in the legend.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Handles for ``ax.legend(handles=...)``.
    """
    return [Line2D([0], [0], label=k, marker='.', markersize=markersize, markeredgecolor=v, markerfacecolor=v, linestyle='')
               for k, v in zip(labels, colors)]

def unrolled_subplots(N, Ncol=4, elsize=(6,4)):
    """Create a figure with `N` subplots on a grid and return the axes as a flat sequence.

    The axes are filled row by row; grid cells beyond `N` are switched off.

    Parameters
    ----------
    N : int
        Number of subplots.
    Ncol : int, default 4
        Maximum number of columns (fewer if `N` is smaller).
    elsize : tuple of float, default (6, 4)
        Size (width, height) of every subplot in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list or numpy.ndarray of matplotlib.axes.Axes
        `N` axes in row-major order (a 1-D array if the grid has a single row or column, else a list).
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
    """Print `text` (str) prefixed with the current time (HH:MM:SS)."""
    print(datetime.now().strftime("%H:%M:%S"),"-",text)

def silent_log(x):
    """Natural logarithm (``numpy.log``) without warnings for zero, negative or NaN input.

    Parameters
    ----------
    x : array-like or float

    Returns
    -------
    numpy.ndarray or float
        ``log(x)``: -inf for 0, NaN for negative values.
    """
    with np.errstate(invalid='ignore', divide = 'ignore'):
        return np.log(x)

def label_axis_ArcSinh(ax, cofactor, onx=True, minpower=2, mark_cofactor=False):
    """Label an axis that shows ``arcsinh(x / cofactor)`` data with powers of ten of the linear values.

    Places major ticks at 0 and at +-10^p for every integer power p within the current axis limits with
    ``|p| >= minpower``, labelled ``$10^p$`` / ``$-10^p$`` / ``$0$``, and minor ticks in between (at 2..9 x 10^p
    between neighbouring powers, at the left-out powers between 0 and +-10^minpower). Call it after plotting, since
    it uses the current axis limits (which it keeps).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to relabel; the data on it must be ``arcsinh(x / cofactor)``.
    cofactor : float
        Cofactor of the arcsinh transform of the plotted data, in the units of the linear data.
    onx : bool, default True
        Relabel the x axis; if False, the y axis.
    minpower : int, default 2
        Smallest power of ten that is labelled: powers with ``|p| < minpower`` are left out (0 is always labelled).
        Small powers crowd together near 0 on the arcsinh scale; e.g. 3 for a cofactor of 1500 leaves out 10 and 100.
    mark_cofactor : bool, default False
        Also draw short marks at the edge of the axis at x = +-`cofactor`.
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
        
    # narrow ranges can leave a single major tick (e.g. only 0): then there are no minor ticks
    minor_powers = np.hstack([get_minor(available_powers[i], available_powers[i+1]) for i in range(len(available_powers)-1)]) if len(available_powers)>1 else np.zeros(0)

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
    """Concatenate AnnData objects along the events (``anndata.concat``), silencing the duplicate-name warning.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        Objects to concatenate.
    make_unique : bool, default False
        Make the event names unique (``obs_names_make_unique``). Samples read with `cytovanni.io` each have the event
        names "0", "1", ..., so set this when concatenating several samples.
    join : {'inner', 'outer'}, default 'inner'
        Keep only the channels shared by all objects ('inner') or all channels ('outer'), see ``anndata.concat``.
    merge : str, optional
        How ``.var`` (and the other variable-wise fields) are merged, see ``anndata.concat`` (e.g. ``'same'``).
        Default None: these fields are dropped. ``.uns`` is not merged either.
    permute : bool, default False
        Return the events in a random order (numpy's global random state).

    Returns
    -------
    anndata.AnnData
        The concatenated object (a copy if `permute`).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        adata = anndata.concat(adatas, join=join, merge=merge)
    if make_unique:
        adata.obs_names_make_unique()
    if permute:
        adata = adata[np.random.permutation(adata.shape[0])].copy()
    return adata

def anndata_subsample(adata, N=20000, copy=True):
    """Randomly subsample `adata` to at most `N` events (without replacement, numpy's global random state).

    Parameters
    ----------
    adata : anndata.AnnData
        Events to subsample (event names need not be unique).
    N : int, default 20000
        Maximum number of events. Objects with at most `N` events are returned complete.
    copy : bool, default True
        If True, always return a new object (also when `adata` has at most `N` events). If False, return `adata`
        itself when it is small enough, otherwise a view of the subsample.

    Returns
    -------
    anndata.AnnData
        Subsample in random order (all events, in the original order, if `adata` is small enough).
    """
    if adata.shape[0]<=N:
        return adata.copy() if copy else adata
    ad = adata[np.random.permutation(adata.shape[0])[:int(N)]]
    return ad.copy() if copy else ad

def df_subsample(df, N=20000, copy=True):
    """Randomly subsample a DataFrame to at most `N` rows (without replacement, numpy's global random state).

    Parameters
    ----------
    df : pandas.DataFrame
        Rows to subsample (the index need not be unique).
    N : int, default 20000
        Maximum number of rows. DataFrames with at most `N` rows are returned complete.
    copy : bool, default True
        If True, always return a new object (also when `df` has at most `N` rows). If False, return `df` itself when
        it is small enough; a subsample is always a copy.

    Returns
    -------
    pandas.DataFrame
        Subsample in random order (all rows, in the original order, if `df` is small enough).
    """
    if df.shape[0]<=N:
        return df.copy() if copy else df
    return df.iloc[np.random.permutation(df.shape[0])[:int(N)]].copy()

from collections import Counter

def subsample_label_indices(labels, max_multiple=None):
    """Subsample events so that no label is more than `max_multiple` times as frequent as the rarest one.

    E.g. to balance cell types or batches in a plot. Random choice with numpy's global random state.

    Parameters
    ----------
    labels : array-like
        Label of every event.
    max_multiple : float, optional
        Labels with more than ``max_multiple * n_rarest`` events are subsampled to ``int(max_multiple * n_rarest)``
        events, the others are kept complete. Default None: every label is subsampled to the count of the rarest one.

    Returns
    -------
    numpy.ndarray of int
        Positions (not index labels) of the chosen events, in random order.

    Examples
    --------
    >>> idx = subsample_label_indices(adata.obs["celltype"], max_multiple=2)
    >>> adata_balanced = adata[idx]
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
    """Check whether the width (-W) of a channel is measured independently or computed from area and height.

    Some instruments do not measure the pulse width but compute it as area / height; then the -W channel carries no
    additional information. The check correlates ``{channel}-W`` with ``{channel}-A / {channel}-H`` over all events
    and treats a correlation of at least 0.999 as computed.

    Parameters
    ----------
    adata : anndata.AnnData
        Events, with the channels ``{channel}-A``, ``{channel}-H`` and ``{channel}-W``.
    channel : str
        Channel name without the -A/-H/-W suffix, e.g. ``"FSC"``.
    layer : str, default "raw"
        Layer to use.
    plot : bool, default False
        Show a scatter plot of -W against -A / -H (``plt.show()``).

    Returns
    -------
    bool
        True if the width appears to be measured independently (correlation < 0.999), False if it appears to be
        computed from area and height.
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

