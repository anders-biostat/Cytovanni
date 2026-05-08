import numpy as np
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt
import hashlib
from .misc import textcolor, get_cmap, cmap_to_legendhandles, wavelength_to_hex
from ..exceptions import DefaultScatterGatingException, SettingsHashException, DefaultScatterCalibrationException, CytometerConfigurationException, CytovanniWarning

def _default_hash_settings(adata, use_voltages=True, use_gains=True, include_scatter=True, include_fluorescence=True):
    """ Convert all relevant settings of the cytometer when sample adata was measured to a hash.
        Required to later automatically check that the correct rainbow batch was used,
        and to automatically infer the correct batch.
        
        Can be replaced by arbitrary custom functions that return a hash.
        
        Uses the settings for all scatter and fluorophore channels.

        To fail gracefully, only uses available channels for calculation and adds 'red_' to the hash if some are missing.
        Maybe better to explicitly set the unique scatter channels in the config to avoid doing this?!
        
        Do not use 'hash()'! Not deterministic between different kernels!
        Instead go through hashlib or equivalent.
        
        :param use_voltages: bool. Whether to use voltage setting 'pnv'.
        
        :param use_gains: bool. Whether to use gain setting 'png'.
        
        :param include_scatter: bool. Whether to include scatter channel settings.
        
        :param include_fluorescence: bool. Whether to include fluorescence channel settings.
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

class CytometerConfiguration:
    """
    The CytometerConfiguration class stores general attributes of the data,
    such as the fluorophore channels and intensity range.

    :param channels_fluorescence: list. List of all fluorescence intensity channels. Only include -A here.

    :param channels_scatter: list. List of all relevant scatter channels, 'FSC-A', 'SSC-A' etc.
    
    :param default_scatter_gate: list. Optionally set the scatter channels that the scatter calibration should default to. Dictionary of channels on which a factor should be fitted, with the channels that it should be applied to. E.g. to fit a factor for the forward scatter and apply it to 'FSC-A' and 'FSC-H' use {'FSC-A':['FSC-A','FSC-H']}.
    
    :param max_linear_range: float. Maximum intensity at which linearity can be assumed for the detectors.
    
    :param channels_meta: list. List of per event metadata, usually just 'Time'.
    
    :param try_adding_HW: bool. Whether it should try adding -H and -W channel data for channels_fluorescence.
    
    :param laserdict: dict. Optional, assign channels to lasers. Structured as {laser:[channels]}.
    
    :param replace_channelname_dict: dict. Optional, in case channel labels are not consistent on the same machine, replaces channel names with value in dict immediately after loading data.
    
    :param fct_hash_settings: function. Function that takes loaded sample in AnnData format, and calculates a hash based on the cytometer settings. Default uses the voltages 'pnv' and gains 'png' for all scatter and fluorescence channels; this should probably capture everything that is relevant for most cytometers. Does not compare laser settings since most cytometers don't export them to the .fcs files, but those are anyway usually held constant. Can be replaced by an arbitrary custom function that returns a hash. To disable this comparison, pass something that assigns everything a default valie like (lambda adata: 0).
    """
    
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
                ):
        
        self.channels_fluorescence = list(channels_fluorescence)
        self.channels_scatter = list(channels_scatter)
        self.channels_meta = list(channels_meta)
        self.default_scatter_gate_channels = list(default_scatter_gate)
        self.default_scatter_calibrate = default_scatter_calibrate
        
        self.max_linear_range = float(max_linear_range)
        self.cytometer = str(cytometer)
        self.try_adding_HW = bool(try_adding_HW)
        self.laserdict = laserdict
        self.replace_channelname_dict = replace_channelname_dict
        
        self.fct_hash_settings = fct_hash_settings
        
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
        """ Make sure scatter and fluorescence channels don't overlap.
        """
        overlap = set(self.channels_scatter) & set(self.channels_fluorescence)
        if len(overlap)>0:
            raise CytometerConfigurationException(f"Channels {list(overlap)} are present in both scatter and fluorescence, choose one!")
        
        overlap = set(self.channels_scatter) & set(self.channels_fluorescence_addHW)
        if len(overlap)>0:
            raise CytometerConfigurationException(f"Automatically added -H and -W channels {list(overlap)} are present in both scatter and fluorescence, either remove them from scatter or make sure automatically adding them works as expected!")
    
    def check_process_laserdict(self):
        """ Make sure laserdict is properly formatted etc.
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
    
    def to_dictionary(self):
        """ Export configuration to dictionary.
        """
        savekeys = ["channels_fluorescence", "channels_scatter", "channels_meta",
                    "default_scatter_gate_channels", "default_scatter_calibrate",
                    "max_linear_range", "cytometer", "try_adding_HW",
                    "channels_fluorescence_addHW", "laserdict", "replace_channelname_dict"]
        export = {}
        for key in savekeys:
            export[key] = getattr(self, key)
        return export
    
    def import_exported(self, exported):
        """ Import configuration from exported dictionary.
        """
        makelistkeys = ["channels_fluorescence", "channels_scatter", "channels_meta",
                        "default_scatter_gate_channels", "channels_fluorescence_addHW"]
        for k, v in exported.items():
            if k in makelistkeys:
                setattr(self, k, list(v))
            else:
                setattr(self, k, v)
        self.check_process_laserdict()
    
    def write_json(self, filepath):
        """ Write configuration to .json file.
        """
        with open(filepath, "w") as outfile: 
            json.dump(self.to_dictionary(), outfile)
    
    @classmethod
    def load_json(cls, filepath):
        """ Construct cytometer configuration from exported .json file.
        """
        cytoconfig = cls([])
        with open(filepath, "r") as file: 
            exported = json.load(file)
        cytoconfig.import_exported(exported)
        
        return cytoconfig
    
    def get_default_scatter_gate_channels(self):
        """ Get self.default_scatter_gate_channels, but also implements some sanity checks.
            Do this here instead of at instantiation because setting this appropriately is optional.
        """
        if len(self.default_scatter_gate_channels)!=2:
            raise DefaultScatterGatingException(f"Default scatter gating choice {self.default_scatter_gate_channels} is invalid, need to be exactly two channels!")
        all_channels = self.channels_scatter + self.channels_fluorescence + self.channels_fluorescence_addHW
        if not np.all(np.in1d(self.default_scatter_gate_channels, all_channels)):
            raise DefaultScatterGatingException(f"Default scatter gating choice {self.default_scatter_gate_channels} is not available! Choose two of possibly available channels:\n{all_channels}")
        return self.default_scatter_gate_channels
    
    def get_settings_hash(self, adata):
        """ Get hash of channel settings in adata, raise more informative exception if it fails for any reason.
        """
        try:
            return self.fct_hash_settings(adata)
        except Exception as e:
            raise SettingsHashException("Something went wrong when calculating the settings hash from adata!") from e
    
    def check_scatter_calibration(self, scatter_calibrate, message_default=False):
        """ Sanity checks for the scatter calibration.
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
        """ Get self.default_scatter_calibrate, but also implements some sanity checks.
            Do this here instead of at instantiation because setting this appropriately is optional.
        """
        scatter_calibrate = self.default_scatter_calibrate
        self.check_scatter_calibration(scatter_calibrate, message_default=True)
        return scatter_calibrate
    
    def plot_height_clipping(self, adata, logscale=False):
        """ Plot height clipping. Area of fluorescence channels against height of fluorescence channels,
            once the height channels saturate the area can no longer be assumed to be linear.
            Preferrably test this with rainbow beads; due to their small size clipping sets in earlier
            than for cells, and for rainbow calibration it is important to set the max linear range
            appropriately for them.
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
    """ Cytometer configuration for our A3.
    """
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
    """ Cytometer configuration for our A3.
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
    """ Cytometer configuration for our S6.
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
    """ Cytometer configuration for our S8.
        
        Not sure which lasers do which scatter detectors.
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
                               'V15 (700)-A', 'V16 (725)-A', 'V17 (750)-A', 'V18 (780)-A', 'V19 (810)-A', 'V20 (845)-A'],
                     "Blue":['B1 (500)-A', 'B2 (515)-A', 'B3 (530)-A', 'B4 (545)-A', 'B5 (575)-A', 'B6 (590)-A', 'B7 (605)-A', 'B8 (625)-A',
                             'B9 (655)-A', 'B10 (675)-A', 'B11 (700)-A', 'B12 (725)-A', 'B13 (750)-A', 'B14 (780)-A', 'B15 (810)-A', 'B16 (845)-A'
                            ] + ["FSC-A","FSC-H","FSC-W"],
                     "Yellow-Green":['YG1 (575)-A', 'YG2 (590)-A', 'YG3 (605)-A', 'YG4 (625)-A', 'YG5 (655)-A', 'YG6 (675)-A', 'YG7 (700)-A',
                                     'YG8 (725)-A', 'YG9 (750)-A', 'YG10 (780)-A', 'YG11 (810)-A', 'YG12 (845)-A'],
                     "Red":['R1 (655)-A', 'R2 (675)-A', 'R3 (700)-A', 'R4 (725)-A', 'R5 (750)-A', 'R6 (780)-A', 'R7 (810)-A', 'R8 (845)-A'],
                     "Imaging":["SSC (Imaging)-A","SSC (Imaging)-H","SSC (Imaging)-W"]}
        
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
    """ Takes dict and adds -A, -H, -W to all keys and values.
        To make using replace_channelname_dict simpler, instead of having to manually add the different modalities of every channel.
    """
    rpldict_full = {}
    for k, v in rpldict.items():
        for add in ["-A","-H","-W"]:
            rpldict_full[k+add] = v+add
    return rpldict_full

class CytometerConfiguration_FACSAriaIII(CytometerConfiguration):
    """ Cytometer configuration for our FACSAria III.
    """
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
    """ Cytometer configuration for our other FACSAria III.
    """
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
    """ Cytometer configuration for our FACSAria Fusion.
    """
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
    """ Cytometer configuration for our other FACSAria Fusion.
    """
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
    """ Cytometer configuration for our DxFLEX in 1 laser configuration.
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
    """ Cytometer configuration for our Cytek Aurora.

        Max linear range taken from as 4e6:
        https://assets.thermofisher.com/TFS-Assets/BID/Reference-Materials/spectral-flow-cytometry-expanding-research-spectrum-article.pdf
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


