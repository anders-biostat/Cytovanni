from .utils import scale_spectra_wfactors, extract_logfactor_laserfactor, extract_logfactor_laserfactor_bylaser, reorder_peaks

from .gmm import fit_rainbow_GMM, get_rainbow_GMM_gate

from .plot import plot_rainbow_channel_GMM_single, plot_GMM_llh, plot_eval_RBInt_fit

from .setvolt import get_config_from_rainbow_reference
from .setvolt import get_voltages_from_rainbow, color_voltageconfig

from .rboe import RainbowBatchOrdinalEncoder

from .rbim_scatter import RainbowScatterIntegrationModule
from .rbim_data import RainbowFluorescenceGMMIntegrationDataset
from .rbim_fluorescence import RainbowFluorescenceGMMIntegrationModule, RainbowFluorescenceGMMIntegrationTrainer
from .rbim import RainbowIntegrator

from .pmt import PMTExponentFitter
