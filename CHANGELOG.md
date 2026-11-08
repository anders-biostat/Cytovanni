# Changelog

## 1.2.0 (2026-11-08)

A review release: about 130 bug fixes, a test suite, numpydoc docstrings for the whole API, new evaluation modules,
and a set of API clean-ups. There are no compatibility aliases for renamed functions or arguments: code that uses the
old names has to be updated (see "Breaking changes"). `FIXLOG.md` in the repository lists every fix with the
conditions under which it changes results.

### Breaking changes

Renamed or removed functions and arguments:

| Before (1.1.0) | Now (1.2.0) |
|---|---|
| `cytovanni.utils.label_axis_ArcSinh(ax, cofactor, onx=True, minpower=2, ...)` | `set_arcsinh_ticks(ax, cofactor, axis="x", minpower=None, ...)` (`onx=False` -> `axis="y"`; `minpower` is chosen automatically, pass `minpower=2` for the earlier labels) |
| `cytovanni.rainbow.plot_eval_RBInt_fit(rbint, ...)` | `plot_eval_calibration(calibrator, ...)` |
| `SpilloverCollection(rbint=...)`, `.from_custom(rbint, ...)`, `.from_AutoSpill(rbint, ...)`, attribute `.rbint` | `calibrator=...`, `.calibrator` |
| `SpectralFitTransformer.add_standardisation_parameters(rbint_key=...)` | `calibration_factor_key=...` |
| `SpilloverCollection.normalize_spectra(dye_channel_key=...)` | `norm_channels=...` (Series stain -> channel) |
| `PanelModule.get_grouped_jac_single_fromad(s_dig=...)` | `decimals=...` |
| `cytovanni.ref.factor.fit_factor_Sinkhorn_LBFGS` | removed (diverged); use `fit_factor_Sinkhorn` |
| `base_plot_NxN*(dask=..., label_arcsinh=...)` | arguments removed (TypeError) |

Changed return values and defaults:
- `CutoffUnmxGate.apply` and `UnmxMinGate.apply` return a numpy bool array (were pandas Series), like the other gates.
  `CutoffUnmxGate.apply` takes `addkey`. `PolygonGate` stores `vertices` as a float array.
- `ScalingTransformer.get_standardisationparams_fromad` returns 5 values, like the `SpectralFitTransformer` version:
  `(None, stainscale, spectra, unmx_int_matrix, None)` (was 3).
- `unrolled_subplots` always returns a 1-D numpy array of the N axes (was a list for 2-D grids).
- `OverlapFitModel.export` / `.save` and the transformers' `export` default to `include_history=False`. The loss per
  step is always saved; the per-step loss matrices and the parameter history only with `include_history=True`.
- `PanelModule.get_unmixing_resistance` and `get_unmixing_resistance_fromad` default to `single_step=False,
  gradstep_maxnorm=False`.
- The panel diagnostics on AnnData (`get_unmixing_resistance_fromad`, `get_grouped_jac_fromad`,
  `get_grouped_jac_single_fromad`, `get_clustermeans_withz`) apply the fitted channel and stain factors stored in the
  AnnData (`apply_factors=True`; `False` gives the earlier behaviour).
- `ConnectionSampler(max_anchor_dist=k)` trains batches up to k connections from an anchor (was k + 1); 1 now means
  only connections with anchor samples. `SampleOverlapDataset` has a `max_anchor_dist` argument whose default (2)
  keeps its earlier behaviour.
- `LineGate`: the kept side no longer depends on the order of the two points; vertical lines raise.
- `DummyTransformer.from_data` requires one stain per channel.
- `anndata_subsample` / `df_subsample` return a copy also when the input has at most N events (`copy=True`).
- Voltage setting: the `target` and `voltage_target` columns are nullable integers (`Int64`).

Saved files:
- `OverlapFitModel.from_exported` / `from_saved` require the `DDIST_separate` key. Models saved with 1.1.0 must be
  re-saved with it added (`exported["DDIST_separate"] = False`, unless the model was trained with `True`).
- Cytometer configurations are exported with their class, hash function, time zone and instrument-specific
  attributes, and are read back as the same subclass. A configuration with a custom hash function can only be read
  after registering that function (`register_hash_function`); lambdas cannot be exported. Configuration objects
  pickled with 1.1.0 lack the new `timezone` attribute.

Dependencies:
- `pandas<3` (pandas 3 is not tested yet); `joblib` and `tzdata` are declared.

### Changes that alter numerical results

Default calls:
- Scatter gating of bead samples (`gate_scatter_bead_sample`, the `collect_bead_controls_*` functions, rainbow bead
  processing) uses a binned kernel density estimate and no random subsampling: gates are deterministic and move
  slightly.
- `fit_factor_Sinkhorn(aggregate=True)` (the default) divides the summed loss by the number of batches (was the number
  of pairwise comparisons), which changes the effective step size and the fitted factors.
- Event weights in the Sinkhorn / Hausdorff losses (`OverlapFitModel` with `RareMarkerWeights` or a custom marker
  weight) are normalised per side; before, weights summing to N inflated the loss about N-fold. Models fitted with
  event weights change.
- FCS start times are read in the instrument's time zone (`CytometerConfiguration.timezone`, default
  "Europe/Berlin") instead of the reading machine's. `uns["UTC"]` and `uns["uid"]` change for files read on machines
  not set to Berlin time.
- `get_positive_single_CV` uses the sample variance (ddof=1), like the other CVs; single-sample proportion and width
  CVs grow by about sqrt(n / (n - 1)).
- The rainbow scatter calibration is solved in closed form (differences below 1e-4 on the log scale).

Only in specific cases:
- `DDIST_mode="MMD"` with `DDIST_separate=True` now uses a per-marker loss and bandwidth (the loss was silently
  joint); MMD bandwidths use only the loss stains.
- `make_calibrated_fcs` calibrates channels whose $PnN names the configuration renames (FACSAria), and no longer
  changes the wrong column when a calibrated channel is missing from the sample.
- Single-stain control files: channel names with regex characters are found; a channel name that ends another
  (V540 / UV540) no longer matches the wrong file; custom `filename_regex` is used.
- `fit_factors_Sinkhorn` aligns marker columns by name (was by position).
- Extended transformers keep disabled stain / channel scaling disabled for the new batches.
- `RareMarkerWeights` keeps fractional weights; nested batch imputation keeps known labels.
- `fit_rainbow_GMM` retries draw different initialisations; empty components trigger a retry.
- PCA spectra: zero-variance and null components are dropped; NaN no longer spreads through unused components.
- `get_unmixing_resistance` adds (was: set) the single step; events with a zero Jacobian give 0 instead of NaN.
- `get_spectra_fromembedding(include_unused=True)` returns the unmasked spectra.
- `EventRateEstimator` handles quantised event times; `check_pulse_width_independence` handles heights of 0.
- `get_asinh_*_CV_varcorr` keep events with error 0 and give 0 (with a warning) for a negative corrected variance;
  positive CVs ignore values <= 0 instead of returning NaN.
- `ConnectionSampler`: batches without a path to an anchor are no longer trained as anchors.
- `fit_factors_Sinkhorn(dummy=True)` returns factors of 1 without fitting.
- `fix_sample_bug` removes `$SPILLOVER` and keeps the $PnS labels in the written file.

Many further fixes turn crashes into working calls or clear errors and do not change results of calls that worked.

### New

- `cytovanni.utils.alignment`: exact k-nearest-neighbour batch mixing (scikit-learn, or torch on CPU / GPU) and the
  Seurat alignment score with balanced, seeded subsampling per sample and cell type.
- `cytovanni.utils.batcheval`: evaluation with replicate pairs: per-cell-type summaries, MFI / population width /
  proportion CV tables, counting-noise corrections, a Gaussian-negative control (intensities in [-10^3, 10^3]
  replaced by noise), replicate correlations.
- `"arcsinh"` matplotlib axis scale for data in its own units (`ax.set_xscale("arcsinh", cofactor=...)`) and
  `set_arcsinh_ticks` for already transformed data.
- Rainbow fluorescence fit: optional penalty on the broadening of the per-measurement peak covariances
  (`fit_fluorescence(cov_penalty=...)`, default 0; about .3 for QC beads that fluoresce weakly in some channels).
- `random_state` arguments for the voltage-setting functions and `PosNegSingleStainSpectrumFit`.
- `CytometerConfiguration.timezone`, `CytometerConfiguration.from_exported`, `register_hash_function`; configuration
  for the Northern Lights.
- `OverlapFitModel.from_exported`, `incorporate_stain_scaling`, `add_ad_standardised(name_by=...)`,
  `fit_factor_Sinkhorn_grouped`, `fit_rainbow_GMM(order_channels=...)`,
  `PanelModule.get_grouped_jac_fromad(include_unused=...)`, `sum()` of `LossContainer`s.
- The rainbow fluorescence `fit` uses the `lr` of every call (a later call with `warmup=0` kept the old one).
- `get_hex_distinct` no longer reseeds numpy's global random state.

### Documentation, tests, container

- numpydoc docstrings for the public API; API reference with conventions (data scales, cofactors, batch keys).
- Test suite (`pytest`): unit tests on synthetic data, snapshot tests, optional end-to-end tests on example data
  (`data` marker) and CUDA variants (`gpu` marker); run in CI on Python 3.10-3.12.
- The container image includes the test dependencies; `run_test_container.sh` runs the test suite in it.
