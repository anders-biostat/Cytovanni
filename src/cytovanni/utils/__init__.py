from .misc import textcolor, palette_20, palette_28, get_cmap, get_cmap_and_markers, cmap_to_legendhandles, cmap_to_legendhandles_wmarker
from .misc import unrolled_subplots, labelscolors_to_legendhandles, label_axis_ArcSinh
from .misc import wavelength_to_rgb, wavelength_to_hex
from .misc import silent_log
from .misc import printwtime
from .misc import anndata_concat, anndata_subsample, df_subsample
from .misc import subsample_label_indices
from .misc import check_pulse_width_independence

from .rate import EventRateEstimator, estimate_event_rate

from .unmx import apply_arcsinh
from .unmx import invert_spectra, apply_unmixing_inv, apply_unmixing_lstsq
from .unmx import add_adata_unmixed
from .unmx import unmixed_cluster_means

from .config import CytometerConfiguration
from .config import CytometerConfiguration_A3, CytometerConfiguration_A3small, CytometerConfiguration_S6, CytometerConfiguration_S8
from .config import CytometerConfiguration_FACSAriaIII, CytometerConfiguration_FACSAriaIII_v2
from .config import CytometerConfiguration_FACSAriaFusion, CytometerConfiguration_FACSAriaFusion_v2
from .config import CytometerConfiguration_DxFLEX1L, CytometerConfiguration_Aurora

from .oe import CustomOrdinalEncoder

from .nxn import base_plot_NxN, base_plot_NxN_ds

from .evl import symmetric_correlation, get_positive_paired_CV, get_positive_paired_CV_alongdiag, get_positive_single_CV
from .evl import get_asinh_paired_CV, get_asinh_single_CV, get_asinh_paired_CV_alongdiag
