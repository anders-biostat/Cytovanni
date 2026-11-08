Utilities (cytovanni.utils)
===========================

.. currentmodule:: cytovanni.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_cmap
   get_cmap_and_markers
   cmap_to_legendhandles
   cmap_to_legendhandles_wmarker
   unrolled_subplots
   labelscolors_to_legendhandles
   ArcSinhScale
   ArcSinhLocator
   ArcSinhFormatter
   set_arcsinh_ticks
   wavelength_to_rgb
   wavelength_to_hex
   silent_log
   printwtime
   anndata_concat
   anndata_subsample
   df_subsample
   subsample_label_indices
   check_pulse_width_independence
   EventRateEstimator
   estimate_event_rate
   apply_arcsinh
   invert_spectra
   apply_unmixing_inv
   apply_unmixing_lstsq
   add_adata_unmixed
   unmixed_cluster_means
   CytometerConfiguration
   register_hash_function
   CytometerConfiguration_A3
   CytometerConfiguration_A3small
   CytometerConfiguration_S6
   CytometerConfiguration_S8
   CytometerConfiguration_FACSAriaIII
   CytometerConfiguration_FACSAriaIII_v2
   CytometerConfiguration_FACSAriaFusion
   CytometerConfiguration_FACSAriaFusion_v2
   CytometerConfiguration_DxFLEX1L
   CytometerConfiguration_Aurora
   CytometerConfiguration_NorthernLights
   CustomOrdinalEncoder
   base_plot_NxN
   base_plot_NxN_ds
   symmetric_correlation
   get_positive_paired_CV
   get_positive_paired_CV_alongdiag
   get_positive_single_CV
   get_asinh_paired_CV
   get_asinh_single_CV
   get_asinh_paired_CV_alongdiag

Batch mixing (cytovanni.utils.alignment)
----------------------------------------

.. automodule:: cytovanni.utils.alignment
   :no-members:

.. currentmodule:: cytovanni.utils.alignment

.. autosummary::
   :toctree: generated
   :nosignatures:

   knn_same_batch_counts
   knn_same_batch_counts_grouped
   alignment_score_from_counts
   alignment_design
   alignment_scores
   pair_alignment_scores
   weighted_celltype_score
   stable_seed
   resolve_device

Batch evaluation (cytovanni.utils.batcheval)
--------------------------------------------

.. automodule:: cytovanni.utils.batcheval
   :no-members:

.. currentmodule:: cytovanni.utils.batcheval

.. autosummary::
   :toctree: generated
   :nosignatures:

   celltype_counts
   celltype_marker_summary
   wide_by_marker
   mfi_cv_table
   width_cv_table
   proportion_cv_table
   replicate_correlation_table
   resample_pair_shares
   proportion_cv_counting_noise
   mfi_cv_counting_noise
   subtract_counting_noise
   gaussian_negative
