"""Exceptions and warnings of Cytovanni.

All exceptions derive from `CytovanniException`, all warnings from `CytovanniWarning`, so that both can be caught or
filtered as a group (e.g. ``warnings.simplefilter("ignore", CytovanniWarning)``).
"""

class CytovanniException(Exception):
    """Base class of all Cytovanni exceptions."""
    pass

class ChannelMismatchException(CytovanniException):
    """Channels of the cytometer configuration are missing in the loaded sample (`cytovanni.io.readfcs_sample`)."""
    pass

class AutoBatchException(CytovanniException):
    """The rainbow batch of a sample could not be inferred automatically from its date, cytometer and settings."""
    pass

class InconsistentBatchException(CytovanniException):
    """Different cytometer settings (settings hashes) were found within the same rainbow batch."""
    pass

class DefaultScatterGatingException(CytovanniException):
    """The default scatter gate channels of the cytometer configuration are invalid or not available."""
    pass

class ScatterGatingChannelException(CytovanniException):
    """The scatter channels chosen for gating are not available, or not exactly two."""
    pass

class SettingsHashException(CytovanniException):
    """The settings hash of a sample could not be computed (raised from the error of the hash function)."""
    pass

class DefaultScatterCalibrationException(CytovanniException):
    """The (default) scatter calibration names unknown channels or calibrates a channel with several factors."""
    pass

class RepeatedBatchException(CytovanniException):
    """The same rainbow batch name was found on several cytometers."""
    pass

class OrdinalEncoderException(CytovanniException):
    """Generic ordinal encoder module exception."""
    pass

class MissingChannelException(CytovanniException):
    """Some expected channels are missing in the adata."""
    pass

class CalibrationModuleException(CytovanniException):
    """Generic calibration module exception."""
    pass

class CytometerConfigurationException(CytovanniException):
    """Cytometer configuration is inconsistent."""
    pass

class SpilloverFitException(CytovanniException):
    """Generic exception of spillover fit."""
    pass

class SpilloverException(CytovanniException):
    """Generic exception for spillover part."""
    pass

class SpilloverInversionException(CytovanniException):
    """Something went wrong inverting a spillover matrix."""
    pass

class OverlapStandardisationException(CytovanniException):
    """Generic exception for overlap standardisation part."""
    pass



class CytovanniWarning(Warning):
    """Base class of all Cytovanni warnings."""
    pass

class ChannelMismatchWarning(CytovanniWarning):
    """Channels of the cytometer configuration are missing in the loaded sample (scatter channels, if allowed)."""
    pass

class GMMAssignmentWarning(CytovanniWarning):
    """GMM label assignment may not have worked properly."""
    pass

class RainbowSpilloverWarning(CytovanniWarning):
    """An .fcs file contains a non-trivial spillover matrix (warned by `cytovanni.io.readfcs_sample` on request)."""
    pass

class AutoBatchWarning(CytovanniWarning):
    """Warning while automatically inferring rainbow batch from adata."""
    pass

class CalibrationModuleWarning(CytovanniWarning):
    """Generic calibration module warning."""
    pass

class CalibrationModuleUntrainedParameterWarning(CytovanniWarning):
    """Some parameters were not trained and are fixed to their default value."""
    pass

class OrdinalEncoderWarning(CytovanniWarning):
    """Generic ordinal encoder module warning."""
    pass

class MissingGPUWarning(CytovanniWarning):
    """Torch doesn't find any available GPUs."""
    pass

class SpilloverFitWarning(CytovanniWarning):
    """Generic warning of spillover fit."""
    pass

class SpilloverWarning(CytovanniWarning):
    """Generic warning for spillover part."""
    pass

class NegativeSpilloverWarning(CytovanniWarning):
    """Warning that some spectra entries are negative."""
    pass

class OverlapStandardisationWarning(CytovanniWarning):
    """Generic warning for overlap standardisation part."""
    pass
