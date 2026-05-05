class CytovanniException(Exception):
    """A generic Cytovanni exception"""
    pass

class ChannelMismatchException(CytovanniException):
    """The cytometer config assumes the presence of channels that are not present in the loaded sample."""
    pass

class AutoBatchException(CytovanniException):
    """Could not automatically infer rainbow batch from adata."""
    pass

class InconsistentBatchException(CytovanniException):
    """Found different cytometer settings within the same rainbow batch."""
    pass

class DefaultScatterGatingException(CytovanniException):
    """The default for the two scatter channels to gate on is not appropriately set."""
    pass

class ScatterGatingChannelException(CytovanniException):
    """The chosen scatter channels to gate on are not available."""
    pass

class SettingsHashException(CytovanniException):
    """Something went wrong when calculating the hash of the settings."""
    pass

class DefaultScatterIntegrationException(CytovanniException):
    """The default for the scatter integration is not appropriately set."""
    pass

class RepeatedBatchException(CytovanniException):
    """Found the same rainbow batch name on multiple cytometers."""
    pass

class OrdinalEncoderException(CytovanniException):
    """Generic ordinal encoder module exception."""
    pass

class MissingChannelException(CytovanniException):
    """Some expected channels are missing in the adata."""
    pass

class IntegrationModuleException(CytovanniException):
    """Generic integration module exception."""
    pass

class CytometerConfigurationException(CytovanniException):
    """Cytometer configuration is inconsistent."""
    pass

class SpilloverFitException(CytovanniException):
    """Generic exception of spillover fit."""
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

class OverlapIntegrationException(CytovanniException):
    """Generic exception for overlap integration part."""
    pass



class CytovanniWarning(Warning):
    """A generic Cytovanni warning"""
    pass

class ChannelMismatchWarning(CytovanniWarning):
    """The cytometer config assumes the presence of channels that are not present in the loaded sample."""
    pass

class GMMAssignmentWarning(CytovanniWarning):
    """GMM label assignment may not have worked properly."""
    pass

class RainbowSpilloverWarning(CytovanniWarning):
    """Rainbow bead .fcs contains a non-trivial spillover matrix."""
    pass

class AutoBatchWarning(CytovanniWarning):
    """Warning while automatically inferring rainbow batch from adata."""
    pass

class IntegrationModuleWarning(CytovanniWarning):
    """Generic integration module warning."""
    pass

class IntegrationModuleUntrainedParameterWarning(CytovanniWarning):
    """Some parameters were not trained and are fixed to their default value."""
    pass

class OrdinalEncoderWarning(CytovanniWarning):
    """Generic ordinal encoder module warning."""
    pass

class MissingGPUWarning(CytovanniWarning):
    """Torch doesn't find any available GPUs."""
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

class OverlapIntegrationWarning(CytovanniWarning):
    """Generic warning for overlap integration part."""
    pass
