"""Cytometer configurations: channels, linear range, lasers and settings hash of an instrument.

A `CytometerConfiguration` is passed to the readers in `cytovanni.io` and stored with every sample in
``adata.uns['cytoconfig']``; the calibration, gating and spillover modules read their channel lists and limits from
there. The subclasses below are the configurations of the instruments the package was developed with; for another
instrument, create a `CytometerConfiguration` (or a subclass) with its channels.
"""
import numpy as np
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
import hashlib
import copy
from zoneinfo import ZoneInfo
from .misc import textcolor, get_cmap, cmap_to_legendhandles, wavelength_to_hex
from ..exceptions import DefaultScatterGatingException, SettingsHashException, DefaultScatterCalibrationException, CytometerConfigurationException, CytovanniWarning

def _to_builtin(value):
    """Recursively convert numpy values (as read back from .h5ad) to Python builtins, also inside containers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {_to_builtin(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_builtin(v) for v in value)
    return value

def _default_hash_settings(adata, use_voltages=True, use_gains=True, include_scatter=True, include_fluorescence=True):
    """Default settings hash of a sample: SHA-256 of the voltages and gains of its scatter and fluorescence channels.

    Used to check that a sample and its rainbow bead measurement were recorded with the same instrument settings,
    and to find the matching rainbow batch automatically. Can be replaced by any function that returns a hash, see
    `CytometerConfiguration` (`fct_hash_settings`).

    Parameters
    ----------
    adata : anndata.AnnData
        Sample with ``adata.uns['cytoconfig']`` and the channel settings in ``adata.var['pnv']`` (voltage) and
        ``adata.var['png']`` (gain).
    use_voltages : bool, default True
        Include the voltages ``pnv``.
    use_gains : bool, default True
        Include the gains ``png``.
    include_scatter : bool, default True
        Include the scatter channels of the configuration.
    include_fluorescence : bool, default True
        Include the fluorescence channels of the configuration.

    Returns
    -------
    str
        Hex digest. Only the channels present in ``adata.var`` enter it; if some configured channels are missing,
        the hash is prefixed with ``'red_'``.

    Notes
    -----
    Python's built-in ``hash()`` is not used because it is not deterministic between interpreter sessions.
    """
    channels = []
    if include_scatter:
        channels += list(adata.uns["cytoconfig"].channels_scatter)
    if include_fluorescence:
        channels += list(adata.uns["cytoconfig"].channels_fluorescence)

    channels_avail = [c for c in channels if (c in adata.var.index)]
    hashadd = "" if (len(channels) == len(channels_avail)) else "red_"
    
    hashstr = ""
    if use_voltages:
        hashstr += str(adata.var.loc[channels_avail, "pnv"].astype(float).to_list())
    if use_gains:
        hashstr += str(adata.var.loc[channels_avail, "png"].astype(float).to_list())
    
    return hashadd + hashlib.sha256(hashstr.encode('utf-8')).hexdigest()

# Default instrument time zone (CytometerConfiguration.timezone)
DEFAULT_TIMEZONE = "Europe/Berlin"

# Settings-hash functions that an exported configuration can refer to by name (see register_hash_function).
HASH_FUNCTIONS = {"_default_hash_settings": _default_hash_settings}

def register_hash_function(fct, name=None):
    """Register a custom settings-hash function by name, so that configurations using it can be exported and imported.

    Exports (`CytometerConfiguration.to_dictionary`, and with it the .h5ad and .json writers) store only the name of
    the settings-hash function. Register the function in every session before importing such a configuration.

    Parameters
    ----------
    fct : callable
        Takes a sample (anndata.AnnData) and returns a hash, see `CytometerConfiguration` (`fct_hash_settings`).
    name : str, optional
        Name under which it is stored. Default: ``fct.__name__`` (a lambda cannot be told apart this way, so give
        it a name).

    Returns
    -------
    str
        The registered name.

    Raises
    ------
    CytometerConfigurationException
        If a different function is already registered under this name.
    """
    name = fct.__name__ if name is None else str(name)
    if name in HASH_FUNCTIONS and HASH_FUNCTIONS[name] is not fct:
        raise CytometerConfigurationException(f"A different settings-hash function is already registered as '{name}'!")
    HASH_FUNCTIONS[name] = fct
    return name

def _hash_function_name(fct):
    """Registered name of `fct`, or its ``__name__`` if it is not registered (importing then raises)."""
    for name, known in HASH_FUNCTIONS.items():
        if known is fct:
            return name
    return getattr(fct, "__name__", repr(fct))

def _all_config_classes():
    """Dict of class name to class for CytometerConfiguration and all its (indirect) subclasses."""
    classes, stack = {}, [CytometerConfiguration]
    while stack:
        klass = stack.pop()
        classes[klass.__name__] = klass
        stack.extend(klass.__subclasses__())
    return classes

class CytometerConfiguration:
    """Configuration of a cytometer: channels, linear range, lasers and settings hash.

    Passed to the readers in `cytovanni.io` and stored with every sample in ``adata.uns['cytoconfig']``. All channel
    names are the names after `replace_channelname_dict` is applied. Some settings are only checked when they are
    used (`get_default_scatter_gate_channels`, `get_default_scatter_calibration`), since setting them is optional.

    Parameters
    ----------
    channels_fluorescence : list of str
        Fluorescence channels, area (-A) only; the -H and -W channels are added automatically if `try_adding_HW`.
        Reading a sample fails if one of them is missing in the file.
    channels_scatter : list of str, default ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
        Scatter channels to load. Must not overlap with the fluorescence channels.
    default_scatter_gate : list of str, default ["FSC-A", "SSC-A"]
        The two channels the scatter gate of the beads uses by default (`cytovanni.gating`).
    default_scatter_calibrate : dict of str to list of str, optional
        Default scatter calibration of the rainbow calibration: for every key, a factor is fitted on that scatter
        channel and applied to the channels in its list. All channels must be scatter channels, and the lists must
        not overlap. Default ``{"FSC-A": ["FSC-A", "FSC-H"], "SSC-A": ["SSC-A", "SSC-H"]}``.
    channels_meta : list of str, default ["Time"]
        Per-event metadata channels, stored in ``adata.obs``; reading a sample fails if one is missing.
    max_linear_range : float, default 2e5
        Raw intensity up to which the detectors are assumed to be linear; the upper limit for the rainbow
        calibration, the voltage setting, `cytovanni.gating.FluorescenceMinMaxGate` and the single-stain spillover fits.
    cytometer : str, default "default"
        Name of the instrument; stored in ``adata.uns['cytometer']`` and used in the sample uid.
    try_adding_HW : bool, default True
        Also load the -H and -W channels of every fluorescence -A channel, where the file contains them.
    laserdict : dict of str to list of str, default {}
        Optional assignment of channels to lasers, ``{laser: [channels]}``; every channel at most once. Used for
        colours in plots (see `channel_laser_key`).
    replace_channelname_dict : dict of str to str, default {}
        Optional renaming of channels directly after reading, for instruments whose channel names are not consistent
        between files: ``{name in file: name used here}``.
    fct_hash_settings : callable, default _default_hash_settings
        Takes a sample (anndata.AnnData) and returns a hash of the instrument settings that matter for the
        calibration; samples and rainbow beads are matched by it. The default uses the voltages ``pnv`` and gains
        ``png`` of all scatter and fluorescence channels, which should capture the relevant settings of most
        cytometers. Laser settings are not compared, since most cytometers don't export them to the .fcs files and
        they are usually held constant. To disable the comparison, pass a function that returns a constant, such as
        ``lambda adata: 0``. Exports store only the function's name: to import a configuration with a custom function
        again, register it first with `register_hash_function` (a lambda cannot be imported).
    timezone : str, default "Europe/Berlin"
        IANA time zone of the instrument clock. The .fcs $DATE/$BTIM are local instrument time; they are converted
        to UTC (``adata.uns['UTC']``, and from it the uid) in this time zone, independent of the machine that reads
        the file.

    Attributes
    ----------
    default_scatter_gate_channels : list of str
        `default_scatter_gate`.
    channels_fluorescence_addHW : list of str
        The -H and -W channels added for the fluorescence channels (empty if not `try_adding_HW`).
    channel_laser_key : pandas.Series
        Laser of every scatter and fluorescence channel (index: channels), ``'unknown'`` for channels not in
        `laserdict`.

    Raises
    ------
    CytometerConfigurationException
        If scatter and fluorescence channels (including the added -H/-W channels) overlap, `laserdict` lists a
        channel twice, or `timezone` is unknown.

    Notes
    -----
    Subclasses that add attributes list them in ``_export_extras`` so that they are exported too (pandas.Series are
    supported).

    Examples
    --------
    >>> cytoconfig = CytometerConfiguration(["FITC-A", "PE-A", "APC-A"], cytometer="myCanto",
    ...                                     max_linear_range=2.5e5)
    >>> adata = cytovanni.io.readfcs_sample("sample.fcs", cytoconfig)
    """
    _export_extras = ()
    
    def __init__(self,
                 channels_fluorescence,
                 channels_scatter=["FSC-A","FSC-H","FSC-W",
                                   "SSC-A","SSC-H","SSC-W"],
                 default_scatter_gate=["FSC-A", "SSC-A"],
                 default_scatter_calibrate={"FSC-A":["FSC-A", "FSC-H"],
                                            "SSC-A":["SSC-A", "SSC-H"]},
                 channels_meta=["Time"],
                 max_linear_range=2e5,
                 cytometer="default",
                 try_adding_HW=True,
                 laserdict={},
                 replace_channelname_dict={},
                 fct_hash_settings=_default_hash_settings,
                 timezone=DEFAULT_TIMEZONE,
                ):
        
        # copies, so that instances never share (default) mutable arguments
        self.channels_fluorescence = list(channels_fluorescence)
        self.channels_scatter = list(channels_scatter)
        self.channels_meta = list(channels_meta)
        self.default_scatter_gate_channels = list(default_scatter_gate)
        self.default_scatter_calibrate = copy.deepcopy(default_scatter_calibrate)
        
        self.max_linear_range = float(max_linear_range)
        self.cytometer = str(cytometer)
        self.try_adding_HW = bool(try_adding_HW)
        self.laserdict = copy.deepcopy(laserdict)
        self.replace_channelname_dict = dict(replace_channelname_dict)
        
        self.fct_hash_settings = fct_hash_settings
        self.timezone = self._check_timezone(timezone)
        
        if self.try_adding_HW:
            # find missing -H and -W fluorescence channels
            self.channels_fluorescence_addHW = []
            for channel in self.channels_fluorescence:
                if channel[-2:]=="-A":
                    channelH, channelW = channel[:-2]+"-H", channel[:-2]+"-W"
                    if channelH not in self.channels_fluorescence:
                        self.channels_fluorescence_addHW.append(channelH)
                    if channelW not in self.channels_fluorescence:
                        self.channels_fluorescence_addHW.append(channelW)
        else:
            self.channels_fluorescence_addHW = []
        
        self.check_channel_overlap()
        
        self.check_process_laserdict()
    
    
    def check_channel_overlap(self):
        """Raise CytometerConfigurationException if scatter channels are also fluorescence or added -H/-W channels."""
        overlap = set(self.channels_scatter) & set(self.channels_fluorescence)
        if len(overlap)>0:
            raise CytometerConfigurationException(f"Channels {list(overlap)} are present in both scatter and fluorescence, choose one!")
        
        overlap = set(self.channels_scatter) & set(self.channels_fluorescence_addHW)
        if len(overlap)>0:
            raise CytometerConfigurationException(f"Automatically added -H and -W channels {list(overlap)} are present in both scatter and fluorescence, either remove them from scatter or make sure automatically adding them works as expected!")
    
    def check_process_laserdict(self):
        """Check `laserdict` and build `channel_laser_key` from it.

    Raises
    ------
    CytometerConfigurationException
        If a channel appears twice in the same laser or in several lasers.
    """
        if not ( sum([len(v) for v in self.laserdict.values()]) == sum([len(set(v)) for v in self.laserdict.values()]) ):
            raise CytometerConfigurationException(f"Laser dictionary contains duplicate channel entries!")
        if not ( len(set().union(*[set(v) for v in self.laserdict.values()])) == sum([len(v) for v in self.laserdict.values()]) ):
            raise CytometerConfigurationException(f"Laser dictionary assigns some channels to multiple lasers!")
        
        index = [v_ for v in self.laserdict.values() for v_ in v]
        values = [v_ for k, v in self.laserdict.items() for v_ in [k]*len(v)]
        laserkey = pd.Series(values, index=index, dtype=str)
        self.channel_laser_key = pd.Series("unknown", index=self.channels_scatter+self.channels_fluorescence)
        overlap = self.channel_laser_key.index.intersection(laserkey.index)
        self.channel_laser_key.loc[overlap] = laserkey.loc[overlap]
        
    
    def __repr__(self):
        res = textcolor.toBOLD(f"Cytometer configuration for cytometer {self.cytometer}")
        res += textcolor.toBOLD(f"\n    Maximum linear intensity: ")+f"{self.max_linear_range:.0f}"
        res += textcolor.toBOLD(f"\n    Meta channels: ")+f"{self.channels_meta}"
        res += textcolor.toBOLD(f"\n    Scatter channels: ")+f"{self.channels_scatter}"
        res += textcolor.toBOLD(f"\n        Scatter gating defaults to: ")+f"{self.default_scatter_gate_channels}"
        res += textcolor.toBOLD(f"\n    Fluorescence channels: ")+f"{self.channels_fluorescence}"
        res += textcolor.toBOLD(f"\n    Optional fluorescence channels: ")+f"{self.channels_fluorescence_addHW}"
        return res
    
    @staticmethod
    def _check_timezone(timezone):
        """Return `timezone` as str; raise CytometerConfigurationException if it is not a known IANA time zone."""
        try:
            ZoneInfo(str(timezone))
        except Exception as e:
            raise CytometerConfigurationException(f"Unknown time zone '{timezone}', use an IANA name such as 'Europe/Berlin'!") from e
        return str(timezone)

    def to_dictionary(self):
        """Export the configuration to a dictionary (writable to .json and to ``adata.uns``).

    Returns
    -------
    dict
        All settings, plus the configuration class name (``'config_class'``), the registered name of the
        settings-hash function (``'fct_hash_settings'``, see `register_hash_function`) and the subclass attributes
        listed in ``_export_extras`` (pandas.Series as ``{'index': [...], 'values': [...]}``).

    See Also
    --------
    from_exported : the inverse.
    """
        savekeys = ["channels_fluorescence", "channels_scatter", "channels_meta",
                    "default_scatter_gate_channels", "default_scatter_calibrate",
                    "max_linear_range", "cytometer", "try_adding_HW",
                    "channels_fluorescence_addHW", "laserdict", "replace_channelname_dict", "timezone"]
        export = {}
        for key in savekeys:
            export[key] = getattr(self, key)
        export["config_class"] = type(self).__name__
        export["fct_hash_settings"] = _hash_function_name(self.fct_hash_settings)
        for key in self._export_extras:
            value = getattr(self, key)
            if isinstance(value, pd.Series):
                value = {"index": value.index.tolist(), "values": value.tolist()}
            export[key] = value
        return export
    
    def import_exported(self, exported):
        """Overwrite the settings of this instance with an exported configuration.

    Use `from_exported` to get an instance of the exported class instead. All values are converted to Python builtins
    (numpy arrays to lists, numpy scalars to Python scalars, recursively), so a configuration read back from .h5ad
    equals the original and can be written with `write_json`. Re-runs `check_process_laserdict`.

    Parameters
    ----------
    exported : dict
        Output of `to_dictionary`, possibly read back from .json or .h5ad.

    Raises
    ------
    CytometerConfigurationException
        If the exported class is not the class of this instance, the exported settings-hash function is not
        registered (see `register_hash_function`), the time zone is unknown, or the laser assignment is invalid.
    """
        exported = _to_builtin(dict(exported))
        config_class = str(exported.pop("config_class", type(self).__name__))
        if config_class != type(self).__name__:
            raise CytometerConfigurationException(f"Exported configuration is a {config_class}, cannot import it into a {type(self).__name__}! Use CytometerConfiguration.from_exported.")
        if "fct_hash_settings" in exported:
            name = str(exported.pop("fct_hash_settings"))
            if name not in HASH_FUNCTIONS:
                raise CytometerConfigurationException(f"Exported configuration uses the settings-hash function '{name}', which is not known! Register it with cytovanni.utils.register_hash_function before importing.")
            self.fct_hash_settings = HASH_FUNCTIONS[name]
        if "timezone" in exported:
            exported["timezone"] = self._check_timezone(exported["timezone"])
        makelistkeys = ["channels_fluorescence", "channels_scatter", "channels_meta",
                        "default_scatter_gate_channels", "channels_fluorescence_addHW"]
        for k, v in exported.items():
            if k in makelistkeys:
                setattr(self, k, list(v))
            elif k in self._export_extras and isinstance(v, dict) and set(v.keys())=={"index", "values"}:
                setattr(self, k, pd.Series(list(v["values"]), index=list(v["index"])))
            else:
                setattr(self, k, v)
        self.check_process_laserdict()

    @classmethod
    def from_exported(cls, exported):
        """Create a configuration from an exported dictionary (see `to_dictionary`).

    Re-creates the exported class (any subclass of CytometerConfiguration; `cls` if the export names none) and
    restores all exported attributes, including the settings-hash function by name. The subclass ``__init__`` is not
    run: the instance starts from the base defaults and gets everything else from the export.

    Parameters
    ----------
    exported : dict
        Output of `to_dictionary`, possibly read back from .json or .h5ad.

    Returns
    -------
    CytometerConfiguration
        Instance of the exported class.

    Raises
    ------
    CytometerConfigurationException
        If the exported class is unknown (import the module that defines it first), or as in `import_exported`.
    """
        classes = _all_config_classes()
        config_class = str(exported.get("config_class", cls.__name__))
        if config_class not in classes:
            raise CytometerConfigurationException(f"Unknown configuration class '{config_class}'! Import the module that defines it first.")
        klass = classes[config_class]
        cytoconfig = klass.__new__(klass)
        CytometerConfiguration.__init__(cytoconfig, []) # base defaults; everything else comes from the export
        cytoconfig.import_exported(exported)
        return cytoconfig
    
    def write_json(self, filepath):
        """Write the configuration (`to_dictionary`) to a .json file at `filepath` (str), overwriting it."""
        with open(filepath, "w") as outfile: 
            json.dump(self.to_dictionary(), outfile)
    
    @classmethod
    def load_json(cls, filepath):
        """Read a configuration written by `write_json`.

    Parameters
    ----------
    filepath : str
        Path of the .json file.

    Returns
    -------
    CytometerConfiguration
        Instance of the class stored in the file (see `from_exported`).
    """
        with open(filepath, "r") as file: 
            exported = json.load(file)
        return cls.from_exported(exported)
    
    def get_default_scatter_gate_channels(self):
        """Return `default_scatter_gate_channels` after checking them.

    Checked here instead of at construction, since setting them is optional.

    Returns
    -------
    list of str
        The two default scatter gate channels.

    Raises
    ------
    DefaultScatterGatingException
        If there are not exactly two channels, or one of them is not a scatter, fluorescence or added -H/-W channel.
    """
        if len(self.default_scatter_gate_channels)!=2:
            raise DefaultScatterGatingException(f"Default scatter gating choice {self.default_scatter_gate_channels} is invalid, need to be exactly two channels!")
        all_channels = self.channels_scatter + self.channels_fluorescence + self.channels_fluorescence_addHW
        if not np.all(np.isin(self.default_scatter_gate_channels, all_channels)):
            raise DefaultScatterGatingException(f"Default scatter gating choice {self.default_scatter_gate_channels} is not available! Choose two of possibly available channels:\n{all_channels}")
        return self.default_scatter_gate_channels
    
    def get_settings_hash(self, adata):
        """Compute the settings hash of a sample with `fct_hash_settings`.

    Parameters
    ----------
    adata : anndata.AnnData
        Sample, as needed by `fct_hash_settings`.

    Returns
    -------
    object
        The hash (a str for the default function).

    Raises
    ------
    SettingsHashException
        If `fct_hash_settings` raises any exception (chained).
    """
        try:
            return self.fct_hash_settings(adata)
        except Exception as e:
            raise SettingsHashException("Something went wrong when calculating the settings hash from adata!") from e
    
    def check_scatter_calibration(self, scatter_calibrate, message_default=False):
        """Check a scatter calibration dictionary against the scatter channels of this configuration.

    Parameters
    ----------
    scatter_calibrate : dict of str to list of str
        ``{fit channel: [channels the factor is applied to]}``, as `default_scatter_calibrate`.
    message_default : bool, default False
        Say "default scatter calibration" in the error messages.

    Raises
    ------
    DefaultScatterCalibrationException
        If a channel is not among `channels_scatter`, or two keys share a channel they are applied to.
    """

        # check that all keys are present
        missing = []
        for key in scatter_calibrate:
            if not key in self.channels_scatter:
                missing.append(key)
            for ikey in scatter_calibrate[key]:
                if not ikey in self.channels_scatter:
                    missing.append(ikey)
        if len(missing)>0:
            raise DefaultScatterCalibrationException(f"Channels {missing} are present in the {'default ' if message_default else ''}scatter calibration, but not among the loaded scatter channels {self.channels_scatter}!")

        # check that there is no overlap among the channels that should be calibrated with different factors
        overlap = []
        for i, key1 in enumerate(scatter_calibrate):
            for j, key2 in enumerate(scatter_calibrate):
                if i>j:
                    if not set(scatter_calibrate[key1]).isdisjoint(scatter_calibrate[key2]):
                        overlap.append((key1,key2))
        if len(overlap)>0:
            raise DefaultScatterCalibrationException(f"There is some overlap present within the {'default ' if message_default else ''}scatter calibration among channels that should be calibrated with different factors!")
    
    def get_default_scatter_calibration(self):
        """Return `default_scatter_calibrate` after checking it with `check_scatter_calibration`.

    Checked here instead of at construction, since setting it is optional.

    Returns
    -------
    dict of str to list of str
        The default scatter calibration (not a copy).

    Raises
    ------
    DefaultScatterCalibrationException
        See `check_scatter_calibration`.
    """
        scatter_calibrate = self.default_scatter_calibrate
        self.check_scatter_calibration(scatter_calibrate, message_default=True)
        return scatter_calibrate
    
    def plot_height_clipping(self, adata, logscale=False):
        """Scatter the area against the height of every fluorescence channel, to choose `max_linear_range`.

    Once the height saturates, the area can no longer be assumed to be linear. Preferably check this with rainbow
    beads: due to their small size, clipping sets in earlier than for cells, and for the rainbow calibration it
    matters that the linear range is set appropriately for them. Draws the raw -A against the -H channel of every
    fluorescence channel that has one in `adata` (coloured by laser) and a vertical line at `max_linear_range`, into
    a new figure (no ``plt.show()``). Warns (CytovanniWarning) if no height channel is found.

    Parameters
    ----------
    adata : anndata.AnnData
        Events with raw intensities in ``layers['raw']``, including -H channels (see `try_adding_HW`).
    logscale : bool, default False
        Use logarithmic x and y axes.
    """
        fig, ax = plt.subplots(figsize=(10,5))
        total = 0
        cmap = get_cmap(self.channel_laser_key.loc[self.channels_fluorescence])
        for c in self.channels_fluorescence:
            if c.endswith("-A") and c[:-2]+"-H" in adata.var.index:
                plt.scatter(adata[:,c].layers["raw"][:,0],
                            adata[:,c[:-2]+"-H"].layers["raw"][:,0], s=.01, color=cmap[self.channel_laser_key.loc[c]])
                total += 1
        if total==0: warnings.warn("Did not find any height channels in adata, cannot plot height clipping!", CytovanniWarning)
        plt.axvline(self.max_linear_range, color="black")
        plt.xlabel("Fluorescence Area", size=15)
        plt.ylabel("Fluorescence Height", size=15)
        plt.legend(handles=cmap_to_legendhandles(cmap))
        if logscale:
            plt.xscale("log")
            plt.yscale("log")

# ---------------------------------------------------------------------------------------------------------------------
# Configurations of the instruments the package was developed with. The channel lists and laser assignments below are
# instrument data; the class docstrings only say what differs between them.
# ---------------------------------------------------------------------------------------------------------------------

# Dye names of the A3 channels to their detector names (laser, number, wavelength), see CytometerConfiguration_A3.
A3_channelname_rpldct = {   "BB515": "B1 (515)",
                            "BB700": "B2 (710)",
                            "APC": "R1 (670)",
                            "APC-R700": "R2 (710)",
                            "APC-H7": "R3 (780)",
                            "BUV396": "UV1 (379)",
                            "BUV496": "UV2 (515)",
                            "BUV563": "UV3 (586)",
                            "BUV615": "UV4 (610)",
                            "BUV661": "UV5 (670)",
                            "BUV737": "UV6 (740)",
                            "BUV805": "UV7 (820)",
                            "BV421": "V1 (431)",
                            "BV510": "V2 (470)",
                            "BV605": "V3 (610)",
                            "BV650": "V4 (670)",
                            "BV711": "V5 (710)",
                            "BV750": "V6 (750)",
                            "BV786": "V7 (780)",
                            "BYG584": "YG1 (586)",
                            "PE-CF594": "YG2 (610)",
                            "BYG670": "YG3 (670)",
                            "BYG790": "YG4 (780)"}

class CytometerConfiguration_A3(CytometerConfiguration):
    """Configuration of a BD FACSymphony A3 with 23 fluorescence channels named by dye (``cytometer='A3'``).

    Five lasers (UV, Violet, Blue, Yellow-Green, Red), default scatter channels and linear range.

    Attributes
    ----------
    alt_name_dct : dict of str to str
        Channel name to the detector name (e.g. ``'BB515-A'`` to ``'B1 (515)-A'``), for the -A, -H and -W channels.
    """
    _export_extras = ("alt_name_dct",)
    def __init__(self):
        channels_fluorescence = ['BB515-A', 'BB700-A',
                                 'APC-A', 'APC-H7-A', 'APC-R700-A',
                                 'BUV396-A', 'BUV496-A', 'BUV563-A', 'BUV615-A', 'BUV661-A', 'BUV737-A', 'BUV805-A',
                                 'BV421-A', 'BV510-A', 'BV605-A', 'BV650-A', 'BV711-A', 'BV750-A', 'BV786-A',
                                 'BYG584-A', 'BYG670-A', 'BYG790-A', 'PE-CF594-A']
        
        laserdict = {"UV":['BUV396-A', 'BUV496-A', 'BUV563-A', 'BUV615-A', 'BUV661-A', 'BUV737-A', 'BUV805-A'],
                     "Violet":['BV421-A', 'BV510-A', 'BV605-A', 'BV650-A', 'BV711-A', 'BV750-A', 'BV786-A'],
                     "Blue":['BB515-A', 'BB700-A'] + ["FSC-A","FSC-H","FSC-W","SSC-A","SSC-H","SSC-W"],
                     "Yellow-Green":['BYG584-A', 'BYG670-A', 'BYG790-A', 'PE-CF594-A'],
                     "Red":['APC-A', 'APC-H7-A', 'APC-R700-A']}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="A3",
            laserdict=laserdict,
        )
        
        self.alt_name_dct = {k+"-A":v+"-A" for k,v in A3_channelname_rpldct.items()} | {k+"-H":v+"-H" for k,v in A3_channelname_rpldct.items()} | {k+"-W":v+"-W" for k,v in A3_channelname_rpldct.items()}

class CytometerConfiguration_A3small(CytometerConfiguration):
    """Configuration of the same BD FACSymphony A3 with 21 fluorescence channels (``cytometer='A3'``).

    As `CytometerConfiguration_A3`, without the channels BUV737-A and BV650-A, and without ``alt_name_dct``.
    """
    def __init__(self):
        channels_fluorescence = ['BB515-A', 'BB700-A',
                                 'APC-A', 'APC-H7-A', 'APC-R700-A',
                                 'BUV396-A', 'BUV496-A', 'BUV563-A', 'BUV615-A', 'BUV661-A', 'BUV805-A',
                                 'BV421-A', 'BV510-A', 'BV605-A', 'BV711-A', 'BV750-A', 'BV786-A',
                                 'BYG584-A', 'BYG670-A', 'BYG790-A', 'PE-CF594-A']
        
        laserdict = {"UV":['BUV396-A', 'BUV496-A', 'BUV563-A', 'BUV615-A', 'BUV661-A', 'BUV805-A'],
                     "Violet":['BV421-A', 'BV510-A', 'BV605-A', 'BV711-A', 'BV750-A', 'BV786-A'],
                     "Blue":['BB515-A', 'BB700-A'] + ["FSC-A","FSC-H","FSC-W","SSC-A","SSC-H","SSC-W"],
                     "Yellow-Green":['BYG584-A', 'BYG670-A', 'BYG790-A', 'PE-CF594-A'],
                     "Red":['APC-A', 'APC-H7-A', 'APC-R700-A']}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="A3",
            laserdict=laserdict,
        )

class CytometerConfiguration_S6(CytometerConfiguration):
    """Configuration of a BD FACSymphony S6 with 48 fluorescence channels (``cytometer='S6'``).

    Five lasers (UV, Violet, Blue, Yellow-Green, Red), default scatter channels and linear range.
    """
    def __init__(self):
        channels_fluorescence = ['UV379-A', 'UV446-A', 'UV515-A', 'UV540-A', 'UV585-A', 'UV610-A', 'UV660-A', 'UV695-A', 'UV736-A', 'UV809-A',
                                 'V427-A', 'V450-A', 'V470-A', 'V510-A', 'V540-A', 'V576-A', 'V595-A', 'V615-A', 'V660-A', 'V680-A', 'V710-A', 'V750-A', 'V785-A', 'V845-A',
                                 'B510-A', 'B537-A', 'B576-A', 'B602-A', 'B660-A', 'B675-A', 'B710-A', 'B750-A', 'B810-A',
                                 'YG585-A', 'YG602-A', 'YG660-A', 'YG670-A', 'YG695-A', 'YG730-A', 'YG750-A', 'YG780-A', 'YG825-A',
                                 'R660-A', 'R675-A', 'R680-A', 'R710-A', 'R730-A', 'R780-A']
        
        laserdict = {"UV":['UV379-A', 'UV446-A', 'UV515-A', 'UV540-A', 'UV585-A', 'UV610-A', 'UV660-A', 'UV695-A', 'UV736-A', 'UV809-A'],
                     "Violet":['V427-A', 'V450-A', 'V470-A', 'V510-A', 'V540-A', 'V576-A', 'V595-A', 'V615-A', 'V660-A', 'V680-A', 'V710-A', 'V750-A', 'V785-A', 'V845-A'],
                     "Blue":['B510-A', 'B537-A', 'B576-A', 'B602-A', 'B660-A', 'B675-A', 'B710-A', 'B750-A', 'B810-A'] + ["FSC-A","FSC-H","FSC-W","SSC-A","SSC-H","SSC-W"],
                     "Yellow-Green":['YG585-A', 'YG602-A', 'YG660-A', 'YG670-A', 'YG695-A', 'YG730-A', 'YG750-A', 'YG780-A', 'YG825-A'],
                     "Red":['R660-A', 'R675-A', 'R680-A', 'R710-A', 'R730-A', 'R780-A']}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="S6",
            laserdict=laserdict,
        )

class CytometerConfiguration_S8(CytometerConfiguration):
    """Configuration of a BD FACSDiscover S8 spectral sorter with imaging (``cytometer='S8'``).

    78 fluorescence channels, linear range 4e6. Scatter channels FSC, LightLoss (Imaging) and SSC (Imaging); the
    scatter gate defaults to LightLoss (Imaging)-A and SSC (Imaging)-A, the scatter calibration fits FSC-A and
    SSC (Imaging)-A. In `laserdict`, FSC and SSC (Imaging) are assigned to the Blue laser and LightLoss (Violet) to
    the Violet laser; LightLoss (Imaging) is not assigned.

    Parameters
    ----------
    include_imaging_based : bool, default False
        Also load LightLoss (Violet) and the image-derived features (size, intensity, moments, centre of mass,
        eccentricity, ... of the LightLoss, FSC and SSC images) as scatter channels.
    """
    def __init__(self, include_imaging_based=False):
        channels_fluorescence = ['B1 (500)-A', 'B2 (515)-A', 'B3 (530)-A', 'B4 (545)-A', 'B5 (575)-A', 'B6 (590)-A', 'B7 (605)-A', 'B8 (625)-A',
                                 'B9 (655)-A', 'B10 (675)-A', 'B11 (700)-A', 'B12 (725)-A', 'B13 (750)-A', 'B14 (780)-A', 'B15 (810)-A', 'B16 (845)-A', 

                                 'R1 (655)-A', 'R2 (675)-A', 'R3 (700)-A', 'R4 (725)-A', 'R5 (750)-A', 'R6 (780)-A', 'R7 (810)-A', 'R8 (845)-A',

                                 'UV1 (375)-A', 'UV2 (390)-A', 'UV3 (420)-A', 'UV4 (440)-A', 'UV5 (460)-A', 'UV6 (475)-A', 'UV7 (500)-A', 'UV8 (515)-A',
                                 'UV9 (530)-A', 'UV10 (545)-A', 'UV11 (575)-A', 'UV12 (590)-A', 'UV13 (605)-A', 'UV14 (625)-A', 'UV15 (655)-A', 'UV16 (675)-A',
                                 'UV17 (700)-A', 'UV18 (725)-A', 'UV19 (750)-A', 'UV20 (780)-A', 'UV21 (810)-A', 'UV22 (845)-A',

                                 'V1 (420)-A', 'V2 (440)-A', 'V3 (460)-A', 'V4 (475)-A', 'V5 (500)-A', 'V6 (515)-A', 'V7 (530)-A',
                                 'V8 (545)-A', 'V9 (575)-A', 'V10 (590)-A', 'V11 (605)-A', 'V12 (625)-A', 'V13 (655)-A', 'V14 (675)-A',
                                 'V15 (700)-A', 'V16 (725)-A', 'V17 (750)-A', 'V18 (780)-A', 'V19 (810)-A', 'V20 (845)-A',

                                 'YG1 (575)-A', 'YG2 (590)-A', 'YG3 (605)-A', 'YG4 (625)-A', 'YG5 (655)-A', 'YG6 (675)-A', 'YG7 (700)-A',
                                 'YG8 (725)-A', 'YG9 (750)-A', 'YG10 (780)-A', 'YG11 (810)-A', 'YG12 (845)-A']
        
        laserdict = {"UV":['UV1 (375)-A', 'UV2 (390)-A', 'UV3 (420)-A', 'UV4 (440)-A', 'UV5 (460)-A', 'UV6 (475)-A', 'UV7 (500)-A', 'UV8 (515)-A',
                           'UV9 (530)-A', 'UV10 (545)-A', 'UV11 (575)-A', 'UV12 (590)-A', 'UV13 (605)-A', 'UV14 (625)-A', 'UV15 (655)-A', 'UV16 (675)-A',
                           'UV17 (700)-A', 'UV18 (725)-A', 'UV19 (750)-A', 'UV20 (780)-A', 'UV21 (810)-A', 'UV22 (845)-A'],
                     "Violet":['V1 (420)-A', 'V2 (440)-A', 'V3 (460)-A', 'V4 (475)-A', 'V5 (500)-A', 'V6 (515)-A', 'V7 (530)-A',
                               'V8 (545)-A', 'V9 (575)-A', 'V10 (590)-A', 'V11 (605)-A', 'V12 (625)-A', 'V13 (655)-A', 'V14 (675)-A',
                               'V15 (700)-A', 'V16 (725)-A', 'V17 (750)-A', 'V18 (780)-A', 'V19 (810)-A', 'V20 (845)-A'
                              ] + ["LightLoss (Violet)-A", "LightLoss (Violet)-H", "LightLoss (Violet)-W"],
                     "Blue":['B1 (500)-A', 'B2 (515)-A', 'B3 (530)-A', 'B4 (545)-A', 'B5 (575)-A', 'B6 (590)-A', 'B7 (605)-A', 'B8 (625)-A',
                             'B9 (655)-A', 'B10 (675)-A', 'B11 (700)-A', 'B12 (725)-A', 'B13 (750)-A', 'B14 (780)-A', 'B15 (810)-A', 'B16 (845)-A'
                            ] + ["FSC-A","FSC-H","FSC-W", "SSC (Imaging)-A","SSC (Imaging)-H","SSC (Imaging)-W"],
                     "Yellow-Green":['YG1 (575)-A', 'YG2 (590)-A', 'YG3 (605)-A', 'YG4 (625)-A', 'YG5 (655)-A', 'YG6 (675)-A', 'YG7 (700)-A',
                                     'YG8 (725)-A', 'YG9 (750)-A', 'YG10 (780)-A', 'YG11 (810)-A', 'YG12 (845)-A'],
                     "Red":['R1 (655)-A', 'R2 (675)-A', 'R3 (700)-A', 'R4 (725)-A', 'R5 (750)-A', 'R6 (780)-A', 'R7 (810)-A', 'R8 (845)-A']}
        
        channels_scatter = ["FSC-A","FSC-H","FSC-W",
                            "LightLoss (Imaging)-A", "LightLoss (Imaging)-H", "LightLoss (Imaging)-W",
                            "SSC (Imaging)-A","SSC (Imaging)-H","SSC (Imaging)-W"]
        if include_imaging_based:
            channels_scatter += ['LightLoss (Violet)-A', 'LightLoss (Violet)-H', 'LightLoss (Violet)-W',
                                 'Size (LightLoss (Imaging))', 'Size (FSC)', 'Size (SSC (Imaging))',
                                 'Max Intensity (LightLoss (Imaging))', 'Max Intensity (FSC)', 'Max Intensity (SSC (Imaging))',
                                 'Long Axis Moment (LightLoss (Imaging))', 'Long Axis Moment (FSC)', 'Long Axis Moment (SSC (Imaging))',
                                 'Short Axis Moment (LightLoss (Imaging))', 'Short Axis Moment (FSC)', 'Short Axis Moment (SSC (Imaging))',
                                 'Center of Mass (Y) (LightLoss (Imaging))', 'Center of Mass (Y) (FSC)', 'Center of Mass (Y) (SSC (Imaging))',
                                 'Center of Mass (X) (LightLoss (Imaging))', 'Center of Mass (X) (FSC)', 'Center of Mass (X) (SSC (Imaging))',
                                 'Total Intensity (LightLoss (Imaging))', 'Total Intensity (FSC)', 'Total Intensity (SSC (Imaging))',
                                 'Radial Moment (SSC (Imaging))', 'Radial Moment (LightLoss (Imaging))', 'Radial Moment (FSC)',
                                 'Eccentricity (SSC (Imaging))', 'Eccentricity (LightLoss (Imaging))', 'Eccentricity (FSC)',
                                 'Diffusivity (SSC (Imaging))', 'Diffusivity (LightLoss (Imaging))', 'Diffusivity (FSC)',
                                 'Delta CoM (SSC (Imaging)/LightLoss (Imaging))', 'Delta CoM (SSC (Imaging)/FSC)', 'Delta CoM (LightLoss (Imaging)/FSC)'
                                ]
        
        default_scatter_gate = ["LightLoss (Imaging)-A", "SSC (Imaging)-A"]
        
        default_scatter_calibrate = {"FSC-A":["FSC-A", "FSC-H"],
                                     "SSC (Imaging)-A":["SSC (Imaging)-A", "SSC (Imaging)-H"]}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            channels_scatter=channels_scatter,
            default_scatter_gate=default_scatter_gate,
            default_scatter_calibrate=default_scatter_calibrate,
            max_linear_range=4e6,
            cytometer="S8",
            laserdict=laserdict
        )

def _add_AHW_replacedct(rpldict):
    """Expand ``{old: new}`` channel base names to ``{old-A: new-A, old-H: new-H, old-W: new-W}``."""
    rpldict_full = {}
    for k, v in rpldict.items():
        for add in ["-A","-H","-W"]:
            rpldict_full[k+add] = v+add
    return rpldict_full

class CytometerConfiguration_FACSAriaIII(CytometerConfiguration):
    """Configuration of a BD FACSAria III with 18 fluorescence channels (``cytometer='FACSAriaIII'``).

    Lasers UV, Violet, Blue, Green (532 nm), Red. The file channel names (e.g. ``'BL-B'``) are renamed to detector
    names (e.g. ``'B1 (525)'``) with `replace_channelname_dict`. Scatter channels are not assigned to a laser.

    Attributes
    ----------
    lasers : list of str
        Laser names.
    lasers_wavelength : pandas.Series
        Wavelength (nm) per laser.
    lasers_color : pandas.Series
        Hex colour per laser, from its wavelength (`wavelength_to_hex`), for plots.
    """
    _export_extras = ("lasers", "lasers_wavelength", "lasers_color")
    def __init__(self):
        channels_fluorescence = ["B1 (525)-A", "B2 (695)-A",
                                 "G1 (586)-A", "G2 (610)-A", "G3 (660)-A", "G4 (780)-A",
                                 "R1 (670)-A", "R2 (730)-A", "R3 (780)-A",
                                 "V1 (450)-A", "V2 (525)-A", "V3 (610)-A", "V4 (710)-A", "V5 (780)-A",
                                 "UV1 (379)-A", "UV2 (515)-A", "UV3 (710)-A", "UV4 (820)-A"]
        
        laserdict = {"UV":["UV1 (379)-A", "UV2 (515)-A", "UV3 (710)-A", "UV4 (820)-A"],
                     "Violet":["V1 (450)-A", "V2 (525)-A", "V3 (610)-A", "V4 (710)-A", "V5 (780)-A"],
                     "Blue":["B1 (525)-A", "B2 (695)-A"],
                     "Green":["G1 (586)-A", "G2 (610)-A", "G3 (660)-A", "G4 (780)-A"],
                     "Red":["R1 (670)-A", "R2 (730)-A", "R3 (780)-A"]}
        
        replace_channelname_dict = {"BL-B":"B1 (525)", "BL-A":"B2 (695)",
                                    "GL-D":"G1 (586)", "GL-C":"G2 (610)", "GL-B":"G3 (660)", "GL-A":"G4 (780)",
                                    "RL-C":"R1 (670)", "RL-B":"R2 (730)", "RL-A":"R3 (780)",
                                    "VL-E":"V1 (450)", "VL-D":"V2 (525)", "VL-C":"V3 (610)", "VL-B":"V4 (710)", "VL-A":"V5 (780)",
                                    "UV-D":"UV1 (379)", "UV-C":"UV2 (515)", "UV-B":"UV3 (710)", "UV-A":"UV4 (820)"}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="FACSAriaIII",
            laserdict=laserdict,
            replace_channelname_dict=_add_AHW_replacedct(replace_channelname_dict),
        )
        
        # For some later plotting
        self.lasers = ["UV", "Violet", "Blue", "Green", "Red"]
        self.lasers_wavelength = pd.Series([355, 405, 488, 532, 640], index=self.lasers)
        self.lasers_color = pd.Series([wavelength_to_hex(wl) for wl in self.lasers_wavelength], index=self.lasers)

class CytometerConfiguration_FACSAriaIII_v2(CytometerConfiguration):
    """Configuration of a second BD FACSAria III with 18 fluorescence channels (``cytometer='FACSAriaIII_v2'``).

    As `CytometerConfiguration_FACSAriaIII`, but with a Yellow-Green (561 nm) instead of a Green laser (channels
    YG1-YG4 instead of G1-G4).

    Attributes
    ----------
    lasers, lasers_wavelength, lasers_color
        As in `CytometerConfiguration_FACSAriaIII`.
    """
    _export_extras = ("lasers", "lasers_wavelength", "lasers_color")
    def __init__(self):
        channels_fluorescence = ["B1 (525)-A", "B2 (695)-A",
                                 "YG1 (586)-A", "YG2 (610)-A", "YG3 (660)-A", "YG4 (780)-A",
                                 "R1 (670)-A", "R2 (730)-A", "R3 (780)-A",
                                 "V1 (450)-A", "V2 (525)-A", "V3 (610)-A", "V4 (710)-A", "V5 (780)-A",
                                 "UV1 (379)-A", "UV2 (515)-A", "UV3 (710)-A", "UV4 (820)-A"]
        
        laserdict = {"UV":["UV1 (379)-A", "UV2 (515)-A", "UV3 (710)-A", "UV4 (820)-A"],
                     "Violet":["V1 (450)-A", "V2 (525)-A", "V3 (610)-A", "V4 (710)-A", "V5 (780)-A"],
                     "Blue":["B1 (525)-A", "B2 (695)-A"],
                     "Yellow-Green":["YG1 (586)-A", "YG2 (610)-A", "YG3 (660)-A", "YG4 (780)-A"],
                     "Red":["R1 (670)-A", "R2 (730)-A", "R3 (780)-A"]}
        
        replace_channelname_dict = {"BL-B":"B1 (525)", "BL-A":"B2 (695)",
                                    "YG-D":"YG1 (586)", "YG-C":"YG2 (610)", "YG-B":"YG3 (660)", "YG-A":"YG4 (780)",
                                    "RL-C":"R1 (670)", "RL-B":"R2 (730)", "RL-A":"R3 (780)",
                                    "VL-E":"V1 (450)", "VL-D":"V2 (525)", "VL-C":"V3 (610)", "VL-B":"V4 (710)", "VL-A":"V5 (780)",
                                    "UV-D":"UV1 (379)", "UV-C":"UV2 (515)", "UV-B":"UV3 (710)", "UV-A":"UV4 (820)"}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="FACSAriaIII_v2",
            laserdict=laserdict,
            replace_channelname_dict=_add_AHW_replacedct(replace_channelname_dict),
        )
        
        # For some later plotting
        self.lasers = ["UV", "Violet", "Blue", "Yellow-Green", "Red"]
        self.lasers_wavelength = pd.Series([355, 405, 488, 561, 640], index=self.lasers)
        self.lasers_color = pd.Series([wavelength_to_hex(wl) for wl in self.lasers_wavelength], index=self.lasers)

class CytometerConfiguration_FACSAriaFusion(CytometerConfiguration):
    """Configuration of a BD FACSAria Fusion with 16 fluorescence channels (``cytometer='FACSAriaFusion'``).

    Lasers Violet, Blue, Yellow-Green, Red (633 nm), no UV. File channel names are renamed to detector names with
    `replace_channelname_dict`. Scatter channels are not assigned to a laser.

    Attributes
    ----------
    lasers, lasers_wavelength, lasers_color
        As in `CytometerConfiguration_FACSAriaIII`.
    """
    _export_extras = ("lasers", "lasers_wavelength", "lasers_color")
    def __init__(self):
        channels_fluorescence = ["B1 (530)-A", "B2 (695)-A",
                                 "YG1 (586)-A", "YG2 (610)-A", "YG3 (660)-A", "YG4 (710)-A", "YG5 (780)-A",
                                 "R1 (670)-A", "R2 (730)-A", "R3 (780)-A",
                                 "V1 (450)-A", "V2 (525)-A", "V3 (610)-A", "V4 (660)-A", "V5 (710)-A", "V6 (780)-A"]
        
        laserdict = {"Violet":["V1 (450)-A", "V2 (525)-A", "V3 (610)-A", "V4 (660)-A", "V5 (710)-A", "V6 (780)-A"],
                     "Blue":["B1 (530)-A", "B2 (695)-A"],
                     "Yellow-Green":["YG1 (586)-A", "YG2 (610)-A", "YG3 (660)-A", "YG4 (710)-A", "YG5 (780)-A"],
                     "Red":["R1 (670)-A", "R2 (730)-A", "R3 (780)-A"]}
        
        replace_channelname_dict = {"BL-B":"B1 (530)", "BL-A":"B2 (695)",
                                    "YG-E":"YG1 (586)", "YG-D":"YG2 (610)", "YG-C":"YG3 (660)", "YG-B":"YG4 (710)", "YG-A":"YG5 (780)",
                                    "RL-C":"R1 (670)", "RL-B":"R2 (730)", "RL-A":"R3 (780)",
                                    "VL-F":"V1 (450)", "VL-E":"V2 (525)", "VL-D":"V3 (610)", "VL-C":"V4 (660)", "VG-C":"V4 (660)", "VL-B":"V5 (710)", "VL-A":"V6 (780)"}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="FACSAriaFusion",
            laserdict=laserdict,
            replace_channelname_dict=_add_AHW_replacedct(replace_channelname_dict),
        )
        
        # For some later plotting
        self.lasers = ["Violet", "Blue", "Yellow-Green", "Red"]
        self.lasers_wavelength = pd.Series([405, 488, 561, 633], index=self.lasers)
        self.lasers_color = pd.Series([wavelength_to_hex(wl) for wl in self.lasers_wavelength], index=self.lasers)

class CytometerConfiguration_FACSAriaFusion_v2(CytometerConfiguration):
    """Configuration of a second BD FACSAria Fusion with 18 fluorescence channels (``cytometer='FACSAriaFusion_v2'``).

    Lasers UV, Violet, Blue, Yellow-Green, Red (640 nm). File channel names are renamed to detector names with
    `replace_channelname_dict`. Scatter channels are not assigned to a laser.

    Attributes
    ----------
    lasers, lasers_wavelength, lasers_color
        As in `CytometerConfiguration_FACSAriaIII`.
    """
    _export_extras = ("lasers", "lasers_wavelength", "lasers_color")
    def __init__(self):
        channels_fluorescence = ["B1 (530)-A", "B2 (695)-A",
                                 "YG1 (586)-A", "YG2 (610)-A", "YG3 (660)-A", "YG4 (780)-A",
                                 "R1 (670)-A", "R2 (730)-A", "R3 (780)-A",
                                 "V1 (450)-A", "V2 (660)-A", "V3 (710)-A", "V4 (780)-A",
                                 "UV1 (379)-A", "UV2 (515)-A", "UV3 (560)-A", "UV4 (740)-A", "UV5 (820)-A"]
        
        laserdict = {"UV":["UV1 (379)-A", "UV2 (515)-A", "UV3 (560)-A", "UV4 (740)-A", "UV5 (820)-A"],
                     "Violet":["V1 (450)-A", "V2 (660)-A", "V3 (710)-A", "V4 (780)-A"],
                     "Blue":["B1 (530)-A", "B2 (695)-A"],
                     "Yellow-Green":["YG1 (586)-A", "YG2 (610)-A", "YG3 (660)-A", "YG4 (780)-A"],
                     "Red":["R1 (670)-A", "R2 (730)-A", "R3 (780)-A"]}
        
        replace_channelname_dict = {"BL-B":"B1 (530)", "BL-A":"B2 (695)",
                                    "YG-D":"YG1 (586)", "YG-C":"YG2 (610)", "YG-B":"YG3 (660)", "YG-A":"YG4 (780)",
                                    "RL-C":"R1 (670)", "RL-B":"R2 (730)", "RL-A":"R3 (780)",
                                    "VL-D":"V1 (450)", "VL-C":"V2 (660)", "VL-B":"V3 (710)", "VL-A":"V4 (780)",
                                    "UV-E":"UV1 (379)", "UV-D":"UV2 (515)", "UV-C":"UV3 (560)", "UV-B":"UV4 (740)", "UV-A":"UV5 (820)"}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            cytometer="FACSAriaFusion_v2",
            laserdict=laserdict,
            replace_channelname_dict=_add_AHW_replacedct(replace_channelname_dict),
        )
        
        # For some later plotting
        self.lasers = ["UV", "Violet", "Blue", "Yellow-Green", "Red"]
        self.lasers_wavelength = pd.Series([355, 405, 488, 561, 640], index=self.lasers)
        self.lasers_color = pd.Series([wavelength_to_hex(wl) for wl in self.lasers_wavelength], index=self.lasers)

class CytometerConfiguration_DxFLEX1L(CytometerConfiguration):
    """Configuration of a Beckman Coulter DxFLEX with one laser and 5 fluorescence channels FL1-FL5.

    ``cytometer='DxFLEX1L'``. Scatter channels FSC and SSC -A and -H only (no -W), linear range 2e6.
    """
    def __init__(self):
        channels_scatter = ["FSC-A","FSC-H", "SSC-A","SSC-H"]
        
        channels_fluorescence = ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 'FL5-A']
        
        laserdict = {"Green":['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A', 'FL5-A']}
        
        super().__init__(
            channels_scatter=channels_scatter,
            channels_fluorescence=channels_fluorescence,
            cytometer="DxFLEX1L",
            laserdict=laserdict,
            max_linear_range=2e6,
        )

class CytometerConfiguration_Aurora(CytometerConfiguration):
    """Configuration of a 5-laser Cytek Aurora with 64 fluorescence channels (``cytometer='Aurora'``).

    Scatter channels FSC, SSC and SSC-B (side scatter of the blue laser), all three used in the default scatter
    calibration. Linear range 4e6 (22 bit detectors), taken from the Thermo Fisher article "Spectral flow cytometry:
    expanding the research spectrum" (assets.thermofisher.com, TFS-Assets/BID/Reference-Materials).
    """
    def __init__(self):
        channels_fluorescence = ['B1-A', 'B2-A', 'B3-A', 'B4-A', 'B5-A', 'B6-A', 'B7-A', 'B8-A', 'B9-A', 'B10-A', 'B11-A', 'B12-A', 'B13-A', 'B14-A',
                                 'YG1-A', 'YG2-A', 'YG3-A', 'YG4-A', 'YG5-A', 'YG6-A', 'YG7-A', 'YG8-A', 'YG9-A', 'YG10-A',
                                 'R1-A', 'R2-A', 'R3-A', 'R4-A', 'R5-A', 'R6-A', 'R7-A', 'R8-A',
                                 'V1-A', 'V2-A', 'V3-A', 'V4-A', 'V5-A', 'V6-A', 'V7-A', 'V8-A', 'V9-A', 'V10-A', 'V11-A', 'V12-A', 'V13-A', 'V14-A', 'V15-A', 'V16-A',
                                 'UV1-A', 'UV2-A', 'UV3-A', 'UV4-A', 'UV5-A', 'UV6-A', 'UV7-A', 'UV8-A', 'UV9-A', 'UV10-A', 'UV11-A', 'UV12-A', 'UV13-A', 'UV14-A', 'UV15-A', 'UV16-A']
        
        laserdict = {"UV":['UV1-A', 'UV2-A', 'UV3-A', 'UV4-A', 'UV5-A', 'UV6-A', 'UV7-A', 'UV8-A', 'UV9-A', 'UV10-A', 'UV11-A', 'UV12-A', 'UV13-A', 'UV14-A', 'UV15-A', 'UV16-A'],
                     "Violet":['V1-A', 'V2-A', 'V3-A', 'V4-A', 'V5-A', 'V6-A', 'V7-A', 'V8-A', 'V9-A', 'V10-A', 'V11-A', 'V12-A', 'V13-A', 'V14-A', 'V15-A', 'V16-A', 'FSC-A', 'FSC-H', "SSC-A", "SSC-H"],
                     "Blue":['B1-A', 'B2-A', 'B3-A', 'B4-A', 'B5-A', 'B6-A', 'B7-A', 'B8-A', 'B9-A', 'B10-A', 'B11-A', 'B12-A', 'B13-A', 'B14-A', "SSC-B-A", "SSC-B-H"],
                     "Yellow-Green":['YG1-A', 'YG2-A', 'YG3-A', 'YG4-A', 'YG5-A', 'YG6-A', 'YG7-A', 'YG8-A', 'YG9-A', 'YG10-A'],
                     "Red":['R1-A', 'R2-A', 'R3-A', 'R4-A', 'R5-A', 'R6-A', 'R7-A', 'R8-A']}
        
        super().__init__(
            channels_fluorescence=channels_fluorescence,
            channels_scatter=['FSC-A', 'FSC-H', 'FSC-W',
                              "SSC-A", "SSC-H", "SSC-W",
                              "SSC-B-A", "SSC-B-H", "SSC-B-W"],
             default_scatter_gate=["FSC-A", "SSC-A"],
             default_scatter_calibrate={"FSC-A":["FSC-A", "FSC-H"],
                                        "SSC-A":["SSC-A", "SSC-H"],
                                        "SSC-B-A":["SSC-B-A", "SSC-B-H"]},
            cytometer="Aurora",
            max_linear_range=4e6,
            laserdict=laserdict,
        )

class CytometerConfiguration_NorthernLights(CytometerConfiguration):
    """Configuration of a 3-laser (Violet, Blue, Red) Cytek Northern Lights, as used by Gosling et al. (Zenodo 21366352).

    ``cytometer='NorthernLights'``; 38 fluorescence channels, scatter channels and linear range as
    `CytometerConfiguration_Aurora`. SpectroFlo < 4 writes '$CYT=Aurora' into the .fcs files of this instrument, so
    the name in the file header is not reliable.
    """
    def __init__(self):
        channels_fluorescence = ['V1-A', 'V2-A', 'V3-A', 'V4-A', 'V5-A', 'V6-A', 'V7-A', 'V8-A', 'V9-A', 'V10-A', 'V11-A', 'V12-A', 'V13-A', 'V14-A', 'V15-A', 'V16-A',
                                 'B1-A', 'B2-A', 'B3-A', 'B4-A', 'B5-A', 'B6-A', 'B7-A', 'B8-A', 'B9-A', 'B10-A', 'B11-A', 'B12-A', 'B13-A', 'B14-A',
                                 'R1-A', 'R2-A', 'R3-A', 'R4-A', 'R5-A', 'R6-A', 'R7-A', 'R8-A']

        laserdict = {"Violet":['V1-A', 'V2-A', 'V3-A', 'V4-A', 'V5-A', 'V6-A', 'V7-A', 'V8-A', 'V9-A', 'V10-A', 'V11-A', 'V12-A', 'V13-A', 'V14-A', 'V15-A', 'V16-A', 'FSC-A', 'FSC-H', "SSC-A", "SSC-H"],
                     "Blue":['B1-A', 'B2-A', 'B3-A', 'B4-A', 'B5-A', 'B6-A', 'B7-A', 'B8-A', 'B9-A', 'B10-A', 'B11-A', 'B12-A', 'B13-A', 'B14-A', "SSC-B-A", "SSC-B-H"],
                     "Red":['R1-A', 'R2-A', 'R3-A', 'R4-A', 'R5-A', 'R6-A', 'R7-A', 'R8-A']}

        super().__init__(
            channels_fluorescence=channels_fluorescence,
            channels_scatter=['FSC-A', 'FSC-H', 'FSC-W',
                              "SSC-A", "SSC-H", "SSC-W",
                              "SSC-B-A", "SSC-B-H", "SSC-B-W"],
             default_scatter_gate=["FSC-A", "SSC-A"],
             default_scatter_calibrate={"FSC-A":["FSC-A", "FSC-H"],
                                        "SSC-A":["SSC-A", "SSC-H"],
                                        "SSC-B-A":["SSC-B-A", "SSC-B-H"]},
            cytometer="NorthernLights",
            max_linear_range=4e6,
            laserdict=laserdict,
        )
