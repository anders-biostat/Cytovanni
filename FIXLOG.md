# Cytovanni fix log

Fixes since 227b63b ("refactor, doc", the state before the 2026 review), with what each one can change in existing
analyses.

Reference points:
- **"publish"**: the published repository state 227b63b.
- **"repro copy"**: the divergent development copy at ~/Cytovanni, which `cytovanni_reproducibility` imports.

A fix can change results relative to one of these and not the other; both are stated.

## Summary

| ID | Title | Changes results? | Affected when |
|---|---|---|---|
| R1-D1 | AutoSpill tessellation gating: binned KDE, deterministic | yes vs publish; no vs repro copy | every scatter bead gate (`gate_scatter_bead_sample`, single-stain collection, rainbow gating) |
| R1-D2 | Sinkhorn aggregate loss divided by Nbatch | yes vs publish; LBFGS variant also vs repro copy | `fit_factor_Sinkhorn(aggregate=True)` (default) and `fit_factor_Sinkhorn_LBFGS(aggregate=True)` |
| R1-gate-layer | `FluorescenceMinMaxGate` passes `layer` on | only with non-default `layer` | `FluorescenceMinMaxGate(..., layer!="raw")` |
| R1-other | new options (`order_channels`, gating kwargs, NorthernLights config, `from_exported`, `name_by`, Sinkhorn grouped) | no (additive, defaults unchanged) | only when the new options are used |
| R2-A1 | `make_calibrated_fcs` did not calibrate renamed channels (FACSAria) | yes (written .fcs) | configs whose $PnN names are renamed (FACSAria) |
| R2-A2 | calibration of channels missing from the sample | yes (wrong column was changed) | calibrated channel (e.g. FSC-H) absent from the sample |
| R2-A7 | scatter calibration fit in closed form | tiny (optimiser tolerance, <1e-4 log); NaN fits now exact | every `RainbowCalibrator` fit (re-fit, not loading) |
| R2-C1 | `scale_spectra_wfactors` NameError on ambiguous normalisation | no (crash -> works) | conventional spectra without exactly one entry 1.0 |
| R2-C4 | inconsistent peak counts per bead type were not raised | no (later crash -> clear error) | inconsistent `GMM_means` shapes within a rainbow type |
| R2-C8 | `PMTExponentFitter`: one-voltage channels, `plot_exponents`, `get_theoretical_shift` | no (crash -> works) | channels with a single voltage; these two methods |
| R2-C16a | `np.in1d`/2-D `np.cross` deprecations | no | numpy 2 |
| R2-D1 | PCA path of `extract_logfactor_laserfactor` | yes, for `fixed_components=False` only | `fixed_components=False` |
| R2-D3 | `custom-uniform` GMM precision init per peak | possibly (local optimum) | `fit_rainbow_GMM(init_params="custom-uniform")` |
| R2-E7 | `fit_rainbow_GMM` retry: KeyError, identical retries | yes, only where a retry was triggered with an int seed | uneven peak shares + `Nretry>0` |
| R2-E13 | peak share init sized by labels present; empty sample NaN | only if a sample or its brightest peak had no beads | rainbow fluorescence fit init |
| R2-E14 | `Sampler.sample(split=True)` for non-contiguous groups | no for current callers (contiguous) | non-contiguous index groups |
| R2-K6 | `RainbowScatterCalibrationModule.from_data(scatter_calibration=...)` NameError | no (crash -> works) | explicit `scatter_calibration` |
| R2-A3 | `ScalingTransformer.add_standardisation_parameters` stain factor all NaN | no (stored factor becomes usable) | `ScalingTransformer` only |
| R2-A4 | nested batch imputation overwrote known labels | yes, multi-row `transform_df` with mixed known/unknown rows | `MultiOrdinalEncoder.transform_df` with nested keys |
| R2-A6 | `RareMarkerWeights` truncated fractional weights | yes, non-integer weights only | `RareMarkerWeights` groups with weights like 1.5 |
| R2-B1 | `PanelModule` components allocated (channel, channel, comp) | no (crash -> works) | spectral panels (fewer stains than channels) |
| R2-B2 | spectra inverted transposed in `get_standardisationparams_fromad` | no for square panels (identical); spectral crashed before | spectral panels |
| R2-C2 | `ScalingTransformer` warning NameError | no (crash -> warning) | `scale_stains=False` |
| R2-C9 | default `DDIST_mode="sinnkhorn"` typo | no (crash -> works) | direct `OverlapFitModel(...)` without `DDIST_mode` |
| R2-C10 | `fit_factors_Sinkhorn(dummy=True)` ran the fit | yes, `dummy=True` callers get 1.0 | `dummy=True` |
| R2-C12 | MMD with `DDIST_separate`; flag not exported | yes, MMD + separate only; **saved models need the new key** | `DDIST_mode="MMD", DDIST_separate=True`; loading any saved model |
| R2-C13 | batches without path to an anchor trained as anchors | yes, disconnected designs only | `ConnectionSampler` with anchors and disconnected batches |
| R2-D2 | `get_unmixing_resistance(single_step=True)` set instead of added the step | yes, for non-zero embeddings | default `single_step=True` |
| R2-D6 | `plot_eval_NxN(key_color=...)` colours | plot only | `key_color` given |
| R2-E1 | `requires_grad_` overrides (undefined names, return None) | no | `.requires_grad_()` on these modules |
| R2-E8 | `ConnectionSampler` without connections: obscure error | no (clear error) | no cross-batch connections |
| R2-E11 | `fit_factor_PercentileMatching` only took ndarrays | no (crash -> works) | Series/list input |
| R2-E17 | `ScalingTransformer` anchor rejected lists | no (crash -> works) | `anchor_bidx={"batch":[0]}` |
| R2-K1 | `raise NotImplemented` (ref/factor.py) | no (TypeError -> NotImplementedError) | unknown `init` |
| R2-K2 | `incorporate_stain_scaling` lost the class / `cls_smp`; silent clipping | no (warning added) | subclasses, custom smp, `max_abslogstain` |
| R2-K4 | `fit_factor_Sinkhorn_grouped` without shared samples | no (clear error, warning for unconstrained batches) | no sample in two batches |
| R2-C15b | dependencies: `pandas<3` pinned, `joblib` and `tzdata` declared | no | installation |
| R2-C3 | `SpilloverMeasurement.from_custom(abort_inconsistent=False)` TypeError | no (crash -> warning) | inconsistent `set_hash` + `abort_inconsistent=False` |
| R2-C5 | single-stain file regex not escaped / not anchored | yes, where channel names contain regex characters: those stains were "missing", now found | channel names with `( ) . + ...` (FACSAria "B1 (525)", S8); `collect_bead_controls_regularflow`, `make_AutoSpill_fcs_config` |
| R2-C6 | `warn_missing` and `filename_regex` ignored | yes, custom `filename_regex` callers (the regex is now used) | `collect_bead_controls_spectralflow(filename_regex=...)`; `collect_bead_controls_regularflow(warn_missing=False)` (warnings only) |
| R2-C7 | AutoSpill padded channels all got the same file name | possibly (AutoSpill input) | `make_AutoSpill_fcs_config(pad_missing=True)` with >1 missing channel |
| R2-C11 | `OutliersafeTorchSingleStainSpectrumFit.plot` KeyError; `debug=True` NameError | no for slopes; `spectrum` (and written csv) gains an `intercept` column | torch single-stain fit |
| R2-C14 | PCA component count not capped by channel count | no (crash -> works) | `SingleSpectrumPCAFitter` with N > n_channels < n_spectra |
| R2-C15a | str written into float `set_hash` column | no (pandas 3: crash -> works; pandas 2: FutureWarning gone) | `collect_bead_controls_*` |
| R2-E6 | `LocalOutlierFactor(n_neighbors=N//100)` 0..4 for small files | yes, <500 events per fit (100..499: n_neighbors 1-4 -> 5; <100: crash -> works) | `OutliersafeLinear/TorchSingleStainSpectrumFit` on small files |
| R2-E10 | `_find_single_stain` on a folder without .fcs files | no (crash -> "not found") | empty single-stain folder |
| R2-E20 | `plot_test_spectra_reconstruction`: log base, axis, row order | plot + returned errors, if rows not in panel order | `PanelConfiguration.plot_test_spectra_reconstruction` |
| R2-F4 | internal column `channnel_name` -> `channel_name` | no (not in the returned frame) | none |
| R2-A5 | geomloss event weights normalised per side, one-sided weights no longer dropped | yes, models fitted with event weights (vs publish and repro copy) | `OverlapFitModel(markerweight=RareMarkerWeights or custom)` with `DDIST_mode` "sinkhorn" (default) / "hausdorff"; direct `GeomSampleLoss(X, Y, wx, wy)` calls with weights not summing to 1 or one side None |
| R2-D5 | `CustomMultivariateNormal.rsample` ignored `sample_shape` | no (wrong shape -> right shape) | `rsample(shape)` / `sample(shape)` with non-empty shape |
| R2-E2 | `apply_unmixing_inv` error message NameError | no (error message only) | non-DataFrame inverted spectra |
| R2-E3 | `label_axis_ArcSinh` crash on narrow ranges | no (crash -> works) | axis ranges that contain only one major tick (e.g. a narrow range around 0) |
| R2-E4 | `anndata_subsample`/`df_subsample` ignored `copy` | no values; small inputs now return a copy by default | input with <= N events and default `copy=True` (previously the input object itself) |
| R2-E5 | mutable default arguments shared between configs | only if a config's dicts were mutated after construction | `CytometerConfiguration` default `laserdict`, `replace_channelname_dict`, `default_scatter_calibrate`; `readfcs_sample`/`writefcs`/`readfcs_rainbow_sample` dict/list defaults (not mutated, cleanup) |
| R2-E9 | `read_folder_timestamps` on a folder without .fcs files | no (KeyError -> empty frame) | empty folder |
| R2-E12 | `pdsave_collect` swallowed every exception | no (silent wrong object -> error) | corrupted pdsave tuples in saved models |
| R2-E15 | `base_plot_NxN*`: dead `dask` branch, transposed `colors_df`, unused `label_arcsinh` | plot colours only, with `colors_df` in `base_plot_NxN` | `base_plot_NxN(colors_df=...)` (now `colors_df.loc[xkey, ykey]` as documented); `dask=`/`label_arcsinh=` arguments removed (TypeError if passed) |
| R2-E16 | `get_asinh_single_CV_varcorr` NaN for negative corrected variance | yes: NaN -> 0 (with warning) | error variance > observed variance |
| R2-E18 | `fix_sample_bug(delete_spill=True)` kept `$SPILLOVER` | yes (written .fcs no longer carries $SPILLOVER) | source files with `$SPILLOVER` (instead of `$SPILL`) |
| R2-E19 | unused deprecated `torch.cuda.amp` import | no | import-time FutureWarning only |
| R2-E21 | `apply_df` missing on LineGate, CutoffUnmxGate, UnmxMinGate, CombinedGate | no (AttributeError -> works) | `make_calibrated_fcs(gates=[...])` with these gates |
| R2-K1 (geomloss) | `raise NotImplemented(...)` -> `NotImplementedError` | no (error type only) | `GeomSampleLoss(mode)` with an unknown mode |
| R2-K3 | `gate_scatter_bead_sample(**gating_kwargs)` keyword clash | no for working calls (TypeError -> override works) | `gating_kwargs` naming a default the function sets (`final_density_cutoff`, `bound_density_mad_factor`, ...) |
| R2-K5 | `add_adata_unmixed` ignored `layer` | only for explicit `layer` != "calibrated" (now honoured); default unchanged ("calibrated") | calls passing `layer=` |
| R2-G1 | `GeomSampleLoss("hausdorff")` AttributeError at construction | no (crash -> works) | `DDIST_mode="hausdorff"` |
| R2-K7 | invalid escape `\D` in utils/rate.py docstring | no | SyntaxWarning at import |
| R2-F1 | FCS start time read in the instrument time zone (`CytometerConfiguration.timezone`) | yes, only where the reading machine is not on Europe/Berlin time (UTC, uid shift by the offset); none for machines in Germany | `readfcs_sample` / `readfcs_rainbow_sample` / `read_fcs_timestamp` / `read_folder_timestamps` on a machine/container whose local time is not Europe/Berlin (incl. containers without zoneinfo, TZ falls back to UTC) |
| R2-F2 | config export keeps class, hash function, timezone, subclass extras | no for existing files; files written now re-create the subclass | `to_dictionary`/`write_json`/`write_*_h5ad` + import; configs with a custom `fct_hash_settings` now raise on import unless registered |
| R2-D4 | batched eccentricity used the padded height for the bottom-edge test | yes: eccentricity/deform angle of cells cut off at their own bottom edge -> NaN (as the single version) | `add_S8_singlet_maskfeatures` / `calculate_batched_eccentricity_angle` with masks of different heights (arges image containers) |
| R2-F3 | `add_S8_singlet_maskfeatures` kept, duck-typed `imgcont` documented | no (docstring) | arges/lucy: see todo/ARGES_LUCY.md |
| R2b-G2-01 | `get_rainbow_peak_parameters` / `plot_rainbow_with_PDF` crashed after a GPU fit | no (crash -> works) | `RainbowCalibrator` fitted with `gpu=True` (the default), not reloaded |
| R2b-G2-02 | config read from .h5ad kept ndarrays, `write_json` failed | no (crash -> works) | `write_json` / json export of a config read back from .h5ad |
| R2b-G2-05 | `extend_fit` silently calibrated orphan new rainbow types | no for valid extensions; orphan cases now raise instead of arbitrary factors | `extend_fit` with a new rainbow type measured only in new batches without a known type |
| R2b-G2-08 | `LineGate` kept side depended on the p1/p2 order | yes, gates with p1.x > p2.x | `LineGate` with p1[0] > p2[0] (vertical lines now raise) |
| R2b-G2-09 | `fit_rainbow_GMM` ignored empty GMM components | rarely: samples with an empty component get retries/a warning | a GMM component without beads |
| R2b-G2-10 | dataset with `warn_missing=False` lacked `df_channel_train` | no (latent crash -> works) | `RainbowFluorescenceGMMCalibrationDataset(warn_missing=False)` passed to the trainer |
| R2b-G2-13 | `np.log(0)` RuntimeWarning in every bead gate | no (warning noise only) | every scatter bead gate (binned KDE) |
| R2b-G2-15 | spectrum PCA fitted a zero-variance last component | yes, only stains with <= N spectra | `add_PCA_fit` with n_spectra <= N (default N=5) for some dye |
| R2b-G2-17 | `readfcs_sample`: missing meta channel gave a bare KeyError | no (error message only) | .fcs without a `channels_meta` channel (e.g. `Time`) |
| R2b-G2-03 | `SampleOverlapDataset(keys_sample_covariate=...)` KeyError | no (crash -> works) | covariate keys that are not also batch/id/uid keys |
| R2b-G2-04 | nested imputation with an unknown major batch silently took the first reference | no for valid inputs; such rows now raise instead of getting an arbitrary label | `transform_df` with an unknown minor batch whose major batch has no known minor batch, major key not in `keys_batch` |
| R2b-G2-06 | `ConnectionSampler.max_anchor_dist` off by one; `SampleOverlapDataset` gets `max_anchor_dist` (default 2) | no via `SampleOverlapDataset` (default 2 = old behaviour, verified identical); yes for direct `ConnectionSampler(max_anchor_dist=k)` callers (now trains k, before k+1) | direct `ConnectionSampler` use with anchors; new warning for reachable batches beyond the limit |
| R2b-G2-07 | MMD bandwidth estimated on all stains, loss uses only the loss stains | yes: bandwidth (and so the fit) for `ScalingTransformer` with a stain subset + MMD | `DDIST_mode="MMD"`, `ScalingTransformer` with `scale_stains` not all True (joint and `DDIST_separate`) |
| R2b-G2-11 | `fit_factor_Sinkhorn_LBFGS` removed (diverged) | no (removed; not exported, no callers found) | code importing `cytovanni.ref.factor.fit_factor_Sinkhorn_LBFGS` (ImportError) |
| R2b-G2-12 | empty batch in the Sinkhorn factor fits: IndexError | no (error message only) | `fit_factor_Sinkhorn`, `fit_factor_Sinkhorn_grouped`, `fit_factors_Sinkhorn` with an empty batch |
| R2b-G2-14 | zero/NaN event weight sums gave a silent NaN loss | no for valid weights (identical values); invalid ones raise | `GeomSampleLoss`/`MMDLoss` with weights whose sum is <= 0 or not finite |
| R2b-G2-16 | `get_df_sample_connections` without anchors: AttributeError | no (crash -> works, distances NaN) | `ConnectionSampler(is_anchor=None).get_df_sample_connections()` |
| R2b-G2-18 | `SpectralFitTransformer` + `FixedPanelModule`: standardisation export crashed | no (crash -> works) | `SpectralFitTransformer` with a `FixedPanelModule` |
| R2b-G2-19 | `SampleOverlapDataset(key_batch_sample=None)`: KeyError, then no connections | no (crash -> works) | `key_batch_sample=None` |
| R2b-G2-20 | unmixing error and R NaN for events with a zero Jacobian | yes: NaN -> 0 (error), NaN -> 0 or inf (R), only for those events/stains | `get_unmixing_resistance(_fromad)` on events whose Jacobian row is 0 (e.g. all-zero intensities) |
| R2b-G2-21 | MMD with automatic bandwidth on identical points: NaN | no for ordinary inputs (identical); NaN -> 0 | `MMDLoss()` (no fixed bandwidth) / `estimate_bandwidth` on all-identical points |
| R2b-G2-22 | `get_spectra_fromembedding(include_unused=True)` returned masked spectra | yes, `include_unused=True` callers only | `PanelModule.get_spectra_fromembedding(include_unused=True)` |
| R2b-G2-23 | `apply_component_mask` failed for (components, k>1) | no (crash -> works) | `apply_component_mask` with a (components, k) input, k > 1 |
| R2c-H1 | positive CVs: a single value <= 0 made the CV NaN | yes: NaN -> CV over the positive values | `get_positive_paired_CV(_alongdiag)`, `get_positive_single_CV` with a value <= 0 (e.g. a share of 0) |
| R2c-H2 | `get_positive_single_CV` used ddof=0 | yes, slightly larger: log-variance x n/(n-1) | every `get_positive_single_CV` call (small n most) |
| R2c-H3 | CV functions: NaN without warning; paired varcorr clamped silently | no (values identical; warnings added) | NaN results (< 2 usable values, along-diagonal variance < 0); `get_asinh_paired_CV_varcorr` clamp |
| R2d-D1 | `fit_factors_Sinkhorn` matched markers by column position | yes, frames with differently ordered columns (now aligned); different marker sets now raise | `fit_factors_Sinkhorn` with frames whose columns are not in the same order |
| R2d-D2 | `extend` made disabled stain/channel scaling trainable for new batches | yes, extended transformers with `scale_stains`/`scale_channels` not all True (new batches) | `SpectralFitTransformer`/`ScalingTransformer.extend` |
| R2d-D3 | `dataset_test` channel order not checked | no for valid inputs; mismatched test datasets now raise | `OverlapFitModel(dataset_test=...)`, `get_DDIST_from_dataset(other)` |
| R2d-D4 | PCA kept null components (normalised spectra); NaN spread through unused components | yes: NaN embeddings/spectra -> finite; fewer components where the rank is lower | `add_PCA_fit` on spectra of rank < N (few channels); reconstructions with a non-finite unused embedding |
| R2d-D5 | variance-corrected CVs dropped events with error 0 | yes, inputs with zero errors (NaN -> CV) | `get_asinh_*_CV_varcorr` with errors exactly 0 |
| R2d-D6 | PMT fit with fixed laser exponent needed 3 points | no for >= 3 voltages (same least-squares optimum); 2 voltages: crash -> works | `PMTExponentFitter(fit_laser_exponent=False)` |
| R2d-D7 | `UnmxErrorExclusionMultiplier` crashed in float64 | no (crash -> works; float32 identical) | `.double()` models with a non-empty exclusion multiplier |
| R2d-D8 | `get_spectra_embedding` crashed for an all-fixed panel | no (crash -> empty embedding) | PCA panels without any components |
| R3b-B1 | `fix_sample_bug` dropped the $PnS labels | only the written .fcs ($PnS kept now) | repaired files read via `readfcs_dataframe` or other tools ($PnN reads unchanged) |
| R3b-B2 | `read_folder_timestamps` sorted by the $DATE string without UTC | row order of the table only | folders where some file lacks $DATE/$BTIM |
| R3b-B3 | `CovarianceTrilExtendableParameter` ignored `device` | no (crash -> works) | `extend()` on the GPU (class unused) |
| R3b-B4 | `plot_eval_NxN(datashader=True)` showed nothing without savepath | no (plot only) | inline datashader plots |
| R3b-B5 | `plot_GMM_llh` drew the cutoffs on the current axes | no (plot only) | `plot_GMM_llh(ax=...)` with another current axes |
| R3b-B6 | voltage setting: no linear peak gave target 0 silently | no (target 0/inf error -> NaN + warning) | `get_config_from_rainbow_reference` with a saturated channel |
| R3b-B7 | `DummyTransformer` not usable with `OverlapFitModel` | no (crash -> identity works) | `DummyTransformer` |
| R3b-B8 | `SpilloverCollection` skipped a stain under another dye name silently | no (warning added; collection unchanged) | stains recorded under several dye names |
| R3b-B9 | `LinearSingleStainSpectrumFit.plot_NxN` plotted the wrong channels for a subset | no (plot only) | `plot_NxN(channels=...)` not a leading subset |
| R3b-B10 | `anndata_concat(permute)`, `anndata_subsample`, `df_subsample` selected by name | no for unique names (same draws); crash/extra rows -> works | non-unique event names / index |
| R3b-B11 | S8 scatter channels' laser assignment | no (plot colours, `channel_laser_key`) | `CytometerConfiguration_S8` |
| R3b-U2 | `get_unmixing_resistance(_fromad)` defaults = the author's settings | yes, calls relying on the defaults | `single_step`/`gradstep_maxnorm` not passed |
| R3b-U3 | panel `*_fromad` diagnostics ignored the fitted channel/stain factors | yes, diagnostics on AnnData with stored standardisation parameters | `get_unmixing_resistance_fromad`, `get_grouped_jac(_single)_fromad`, `get_clustermeans_withz` |
| R4-B1 | `CovarianceTrilExtendableParameter` with one axis: RuntimeError -> documented ValueError | no | shape with fewer than 2 axes |
| R4-B2 | `RBF_batched(bandwidth=float)` crashed | no (crash -> works) | float bandwidth (not used by the package) |
| R4-B3 | single-stain file pattern matched a channel that ends another (V540 -> UV540 file) | yes, which control file is used | `collect_bead_controls_regularflow`, `make_AutoSpill_fcs_config`; configs with such channel pairs (S6, Aurora) |
| R4-B4 | `HistogramTesselator` crashed with Nx != Ny | no (crash -> works; Nx == Ny unchanged) | non-square grids |
| R4-B5 | `SpilloverFitException`, `MissingGPUWarning` defined twice | no | none |
| R4-B6 | `EventRateEstimator` inf/biased on quantised event times | yes, rates of recordings whose median event spacing is < 20 time steps | `estimate_event_rate` (arges) |
| R4-B7 | `check_pulse_width_independence` NaN with a height of 0 | yes (NaN -> correct answer) | events with -H <= 0 |
| R4-B8 | `anndata_concat(permute=True)` showed the duplicate-name warning | no (warning only) | `permute=True` |
| R4-B9 | `get_voltages_from_rainbow` crashed on a targetconfig read from .csv | no (crash -> works) | targetconfig round-tripped through .csv |
| R4-B10 | `ExtendableParameter.extend` with mean norm: new entries started at -mean(old raw values) | starting point of new batches only; fitted values after `extend` + fit up to convergence | `RainbowCalibrator.extend_fit` (hd7-01), `ScalingTransformer.extend`, `fit_factor_*` extension |
| R4-P4d | rainbow fit: optional penalty on the broadening of the per-measurement peak covariances | no by default (`cov_penalty=0`); yes where used | `fit_fluorescence(cov_penalty>0)`: reproducibility clbr2pks fits (.3, broadening only) |

Entry format: see the review notes (todo/00_CONVENTIONS.md "Fix log" in the review project).

---

## R1 (merged into main 2026-11-06, commits c405da8..810f7f1)

### R1-D1 AutoSpill tessellation gating: binned KDE, deterministic
- **Commit:** fcb4482 (main), 2026-11-06. **Approval:** R1_approval.md D1.
- **Where:** `gating/autospill.py`, `AutoSpillTesselationGating`.
- **What changed:**
  - The exact `gaussian_kde` is replaced by `_BinnedKDE` (512 grid, FFT convolution).
  - `point_to_ind` uses a linear inverse instead of `searchsorted`.
  - The dummy points for the two-maxima case are spread out.
  - `MIN_REL_MAXIMUM=1e-6` ignores flat-region maxima.
  - `subsample_kde_estimates=None`: no random subsampling, so the gate is deterministic.
- **Affected when:** every scatter gate of bead samples (`gate_scatter_bead_sample`, the `collect_bead_controls_*`
  functions, and the rainbow bead processing).
- **Effect on results:**
  - **Vs publish:** gate boundaries change slightly. In the R1 tests the density error was ~3e-4 of the maximum and
    the gates matched the exact KDE. Results are now deterministic, where publish subsampled 20000 points at random.
  - **Vs repro copy:** none; the code is identical to ~/Cytovanni.
- **Affected analyses:** bead gating in `processing/{refmohammed,refjohanna,instrument_rainbow,gosling}`. They already
  ran with this code (repro copy), so no action is needed.
- **Action:** none for reproducibility. Analyses run with the publish version get slightly different bead gates if
  re-run.

### R1-D2 Sinkhorn aggregate loss divided by Nbatch
- **Commit:** 810f7f1 (main), 2026-11-06. **Approval:** R1_approval.md D2.
- **Where:** `ref/factor.py`, `fit_factor_Sinkhorn` and `fit_factor_Sinkhorn_LBFGS` (aggregate branch).
- **What changed:** with `aggregate=True` the summed loss over Nbatch terms was divided by the number of pairwise
  comparisons. It is now divided by Nbatch. With SGD and a fixed lr this scales the effective step size.
- **Affected when:** `aggregate=True`, which is the default.
- **Effect on results:**
  - **Vs publish:** the fitted factors change (a different optimisation trajectory and convergence within Niter).
  - **Vs repro copy:** `fit_factor_Sinkhorn` is unchanged (the repro copy already divides by Nbatch).
    `fit_factor_Sinkhorn_LBFGS` changes; LBFGS is largely scale-invariant, so the effect is expected to be small.
- **Affected analyses:** `fit_factor_Sinkhorn` is used in `processing/refjohanna/lib/refjoh.py:137`,
  `figures/plots_cytonorm/lib/mohref.py:114` and `jobs/workers/S2_save_formats/...`; these are unchanged vs the repro
  copy. No use of `fit_factor_Sinkhorn_LBFGS` was found. The LBFGS variant was later removed (R2b-G2-11).
- **Action:** none for reproducibility.

### R1-gate-layer `FluorescenceMinMaxGate` passes its `layer` argument on
- **Commit:** 00a2feb (main), 2026-11-06.
- **What changed:** the `layer` argument was ignored, so the gate always used "raw".
- **Affected when:** only with a non-default `layer`. **Effect on results:** with a non-default layer the gate now uses
  that layer. **Action:** none for default use.

### R1-other additive changes (no result change with default arguments)
- c405da8 `OverlapFitModel.from_exported`, `incorporate_stain_scaling` (plus helpers)
- e82b69d `add_ad_standardised(name_by=...)`
- 79cbe4d AutoSpill plots
- e161b4c `main_tile` / `merge_tiles_min_saddle` / `gating_kwargs`
- 9587d64 NorthernLights config
- a113432 `fit_rainbow_GMM(order_channels=None)` and `uns["GMM_order_ok"]`
- d229f71 tqdm `verbose`
- 86649ee `fit_factor_Sinkhorn_grouped`

---

## R2 (bug fixes, branch review/R2-bugs)

Commits are referenced by their message tag (`git log --grep "R2-<group>"`), since hashes change until the branch is
merged.

### R2-A1 `make_calibrated_fcs` did not calibrate renamed channels
- **Commit:** R2-rainbow, 2026-11-07. **Approval:** R2_approval.md A1.
- **Where:** `rainbow/rbim.py`, `RainbowCalibrator.make_calibrated_fcs`.
- **What was wrong:** the frame read with flowkit kept the raw $PnN names (`BL-B-A`), the calibration factors use the
  canonical cytoconfig names (`B1 (525)-A`). The overlap was empty, nothing was multiplied, and the file was still
  marked `__cytovanni_rainbow_calibrated=True`. `gates` were also applied to the raw names.
- **Now:** columns are renamed with `_replace_channel_name` for calibration and gating and written back with the
  original $PnN names; an empty overlap raises `MissingChannelException`.
- **Affected when:** configs that rename channels (FACSAria; any config with a $PnN -> name mapping). Configs without
  renaming were calibrated correctly.
- **Effect on results:** calibrated .fcs files written for FACSAria were in fact uncalibrated (and marked as
  calibrated, so `force=False` skips them on a re-run). Now they are calibrated.
- **Affected analyses:** none found in cytovanni_reproducibility (`make_calibrated_fcs` is not called). Any calibrated
  FACSAria .fcs written elsewhere carries the old behaviour.
- **Action:** re-write calibrated FACSAria .fcs files from the original files (not from the falsely marked ones).
- **Test:** jobs/r2_verify/v_all.py A1.

### R2-A2 calibration of channels missing from the sample
- **Commit:** R2-rainbow. **Approval:** A2.
- **Where:** `rainbow/rbim.py`, `add_calibrated`, `add_calibrated_rainbow`, `add_calibration_factor`.
- **What was wrong:** `get_indexer` returns -1 for a calibrated channel that is not in the sample, so the factor of
  that channel was applied to the *last* column of the layer (`add_calibration_factor` wrote a new row).
- **Now:** factors and shifts are restricted to the sample's channels, with a `CalibrationModuleWarning` naming the
  dropped ones.
- **Affected when:** a channel of the calibrator (typically a scatter channel such as FSC-H) is absent from the sample.
- **Effect on results:** the last column of the calibrated layer was scaled by a wrong factor; now it is untouched
  by the missing channel's factor.
- **Affected analyses:** none found (reproducibility does not call these functions directly).
- **Action:** re-calibrate samples that lack a calibrated channel.
- **Test:** v_all.py A2.

### R2-A7 scatter calibration fit in closed form
- **Commit:** R2-rainbow. **Approval:** A7, user decision 2026-11-07 "Closed-form least squares".
- **Where:** `rainbow/rbim_scatter.py`, `RainbowScatterCalibrationModule.fit` (and `get_loss`, new `get_prediction`).
- **What was wrong:** SGD with a fixed lr on a loss weighted by sqrt(N events) diverged to NaN above ~360k events per
  bead sample, and the NaN parameters were frozen with `is_trained=True`. With ~1000 batches 3000 iterations left
  ~1.7% error.
- **Now:** the same weighted least squares problem is solved exactly (per channel, with the frozen entries and the
  mean norm of `ExtendableParameter`); among equivalent solutions the one nearest the current values is taken, which
  is the limit of the old gradient descent. `fit()` no longer takes `Niter`, `lr`, `showprogress`, `minloss`;
  `losshistory` holds the loss before and after. Non-finite data or results raise.
- **Affected when:** every fit of a `RainbowCalibrator` (`RainbowCalibrator(adatas)`, `extend`). Loading a saved
  calibrator (`from_saved`) does not re-fit and is unaffected.
- **Effect on results:** converged fits change by the old optimiser tolerance (verified: max |param diff| <1e-4 in
  log vs 20000 SGD steps, f_rainbow.py); fits that were NaN or unconverged (very large bead files, ~1000 batches)
  become exact. Scatter calibration factors are the only output.
- **Affected analyses:** `processing/instrument_rainbow/lib/pipeline.py:282,324` and `processing/refmohammed/lib/refmoh.py:79`
  fit `RainbowIntegrator` (the repro copy's name) and would change at the tolerance level on a re-fit; saved
  `RBInt.pt` files keep the old factors.
- **Action:** none (differences below 1e-4 in log). Check any saved calibrator for NaN scatter factors.
- **Test:** v_all.py A7; jobs/r2_verify/f_rainbow.py (SGD equivalence, extend, 1000 x 500k, NaN raise).

### R2-C1 `scale_spectra_wfactors` NameError
- **Commit:** R2-rainbow. **Approval:** C1. **Where:** `rainbow/utils.py`.
- **What was wrong:** `SpilloverWarning` was not imported; conventional spectra without exactly one entry 1.0 raised
  NameError instead of warning and falling back to the spectral convention.
- **Affected when:** conventional spectra (`spectral=False`) with zero or several entries equal to 1.0.
- **Effect on results:** none (crash -> works as documented). **Affected analyses:** `scale_spectra_wfactors` is used in
  `processing/batchint_data/lib/{validation,sources,layers}.py`; they ran, so they never hit this path.
- **Action:** nothing. **Test:** v_all.py C1.

### R2-C4 inconsistent rainbow peak counts not raised
- **Commit:** R2-rainbow. **Approval:** C4. **Where:** `rainbow/rbim_data.py`, `check_type_Npeak_consistency`.
- **What was wrong:** the exception was built but not raised, and the comparison with `prior` was inverted.
- **Affected when:** samples of one rainbow type with different numbers of GMM peaks, or a peak count that differs
  from the trained calibrator on `extend`.
- **Effect on results:** none (obscure shape error later -> clear `CalibrationModuleException`). **Action:** nothing.
- **Test:** v_all.py C4.

### R2-C8 `PMTExponentFitter`
- **Commit:** R2-rainbow. **Approval:** C8. **Where:** `rainbow/pmt.py`.
- **What was wrong:** `fit_all` crashed for a channel with one voltage (2 instead of 3 parameters returned);
  `plot_exponents` used a missing column; `get_theoretical_shift` used a 2-parameter formula on 3 parameters.
- **Now:** one-voltage channels get NaN parameters; `plot_exponents` uses `exponent_PMT`; `get_theoretical_shift`
  returns the fitted model's prediction (intercept included). With `fit_laser_exponent=False` the stored
  `exponent_LAS` is now exactly 1 (it was the unused start value 1 before as well) and the plots use 1.
- **Effect on results:** none for channels that fitted before. **Affected analyses:**
  `figures/plots_calib/lib/calib.py:1238` (`fit_laser_exponent=True`, multi-voltage channels): unchanged.
- **Action:** nothing. **Test:** v_all.py C8; f_rainbow.py.

### R2-C16a numpy 2 deprecations (rainbow, gating, config)
- **Commit:** R2-rainbow. **Approval:** C16. **Where:** `np.in1d` -> `np.isin` in gating/bead.py, rainbow/setvolt.py,
  rainbow/rbim.py, rainbow/rbim_scatter.py (2), utils/config.py; `gating/line.py` `_distance_from_line` uses an explicit
  2-D cross product. **Effect on results:** none. **Action:** nothing.

### R2-D1 PCA path of `extract_logfactor_laserfactor`
- **Commit:** R2-rainbow. **Approval:** D1, user decision 2026-11-07 "Consistent w/ default".
- **Where:** `rainbow/utils.py`, `extract_logfactor_laserfactor(fixed_components=False)` (also via `_bylaser`).
- **What was wrong:** residuals omitted `pca.mean_` (so they contained the full offset), the common factor was centred,
  and the PCA sign was arbitrary.
- **Now:** `logfactors = common * component + residuals` with the component rescaled to mean 1 (fixes the sign) and the
  common factor = channel mean of the PCA reconstruction, offset included. A constant component gives exactly the
  `fixed_components=True` result. Raises if the component has no common direction.
- **Affected when:** `fixed_components=False` only.
- **Effect on results:** common factor (now offset-included, consistently signed) and residuals change for this option.
- **Affected analyses:** none: reproducibility only uses the default `fixed_components=True`
  (`processing/instrument_rainbow/lib/{comparison,exports}.py`, `figures/plots_calib`, `figures/plots_gosling`).
- **Action:** nothing. **Test:** v_all.py D1; f_rainbow.py.

### R2-D3 `custom-uniform` GMM precision init
- **Commit:** R2-rainbow. **Approval:** D3. **Where:** `rainbow/gmm.py`, `fit_rainbow_GMM`.
- **What was wrong:** the mean over channels of event *argsort indices* (not ranks) was used to split events into
  peaks, so each initial precision came from a mix of all peaks.
- **Affected when:** `init_params="custom-uniform"` only. **Effect on results:** EM usually converges to the same
  solution; a different local optimum is possible. **Affected analyses:** none found (reproducibility uses the
  default "kmeans"). **Action:** nothing. **Test:** v_all.py D3.

### R2-E7 `fit_rainbow_GMM` retry path
- **Commit:** R2-rainbow. **Approval:** E7. **Where:** `rainbow/gmm.py`.
- **What was wrong:** the retry warning read `adata.uns['name_uid']` (KeyError without it), and a retry with an int
  `random_state` repeated the identical fit.
- **Now:** `uns.get('name_uid')`; an int seed is incremented per retry (None / RandomState instances unchanged).
- **Affected when:** peak shares fail the `warningthreshold` check, `Nretry>0`, and an int `random_state`.
- **Effect on results:** for such samples the retries now try different initialisations, so the final labels can
  differ (before, the last identical fit was kept with a warning).
- **Affected analyses:** `processing/refmohammed/lib/refmoh.py:64,66` (`random_state=seed`, int): changes only for
  samples that showed the "trying again" warning. `figures/plots_rainbow/01_gating_gmm.ipynb` calls
  `fit_rainbow_GMM(adata, random_state=0)` (int seed; added 2026-11-07, R2c): affected only if a retry was triggered
  there. `processing/instrument_rainbow/lib/pipeline.py:53` passes a `RandomState` instance: unchanged.
- **Action:** check the refmohammed logs for "GMM rainbow bead class assignment may not have worked"; re-run those
  samples if present. **Test:** v_all.py E7.

### R2-E13 rainbow peak share init
- **Commit:** R2-rainbow. **Approval:** E13. **Where:** `rainbow/rbim_data.py`, `_intype_GMM_crosstab_shares`.
- **What was wrong:** the share table was sized by the largest sample and label present, so a missing brightest peak
  or trailing empty sample gave a too small table, and an empty sample gave a NaN row.
- **Now:** sized (samples, peaks of the type); empty samples get uniform shares.
- **Affected when:** a sample of a type without usable beads, or no bead in its highest peak.
- **Effect on results:** initial class weights of the fluorescence model in those cases (previously shape errors or
  NaN). **Action:** nothing. **Test:** v_all.py E13.

### R2-E14 `Sampler.sample(split=True)`
- **Commit:** R2-rainbow. **Approval:** E14. **Where:** `rainbow/rbim_data.py`, `Sampler.sample`.
- **What was wrong:** group membership was looked up by index value, correct only if the index groups were the
  contiguous ranges 0..N.
- **Now:** split by the known per-group counts; empty groups are left out (as before).
- **Effect on results:** none for the current callers (ref/factor.py, ref/data.py pass contiguous groups).
  **Action:** nothing. **Test:** v_all.py E14; f_rainbow.py.

### R2-K6 `scatter_calibration` NameError
- **Commit:** R2-rainbow. **Approval:** K6. **Where:** `rainbow/rbim_scatter.py`, `from_data`.
- **What was wrong:** an explicit `scatter_calibration` referenced the undefined `scatter_calibrate`, and the result
  of `check_scatter_calibration` (None) was assigned to `scatter_calibration`. The check is now only called.
- **Effect on results:** none (crash -> works). **Action:** nothing. **Test:** v_all.py K6.

### R2 ref group (OverlapFitModel, transformers, encoders, factor fits)
All bugs of this group are also present in the repro copy, so "vs publish" and "vs repro copy" are the same.
Commit: R2-ref (`git log --grep R2-ref`), 2026-11-07. Approvals: R2_approval.md, same IDs.

### R2-A3 `ScalingTransformer` stain factor all NaN
- **Where:** `ref/model_transformer.py`, `ScalingTransformer.add_standardisation_parameters`.
- **What was wrong:** a stain-indexed Series was assigned to `adata.var` (channel index), giving all NaN;
  `varm["sfm_stain_factor"]`, which the docstring promises, was never written. Padding of `sfm_spectra` /
  `sfm_standardise_unmx` used row-by-row `.loc` (pandas FutureWarning), now `reindex` (both transformers).
- **Now:** writes `varm["sfm_stain_factor"]` as `SpectralFitTransformer` does; no `var["sfm_stain_factor"]` any more.
- **Affected when:** `ScalingTransformer` only. **Effect on results:** the stored factor becomes usable; unmixing
  matrices unchanged. **Affected analyses:** none found (reproducibility reads `varm["sfm_stain_factor"]` written by
  `SpectralFitTransformer`, which was correct). **Action:** nothing. **Test:** v_all.py A3; f_ref.py.

### R2-A4 nested batch imputation overwrote known labels
- **Where:** `ref/moe.py`, `MultiOrdinalEncoder.transform_df`.
- **What was wrong:** if any row needed imputation of a nested (minor) batch, *all* rows got the nearest-order label,
  including rows with a known label.
- **Affected when:** multi-row `transform_df` calls whose frame mixes known and unknown minor batches. Single-row calls
  (`transform_ad`, `transform_dict`) and training datasets (all known) are unaffected.
- **Effect on results:** batch indices of known rows in such calls (they now keep their own label).
- **Affected analyses:** `figures/plots_batchintHD7/lib/parameters.py:43` (`moe.transform_df(meta_B)`); `meta_B` is
  filtered to known `batch_ref`/`batch` labels first, so it is unaffected unless another nested key has unknown values.
- **Action:** none expected; re-run that figure only if its nested keys contain unknown labels. **Test:** v_all.py A4; f_ref.py.

### R2-A6 `RareMarkerWeights` truncated fractional weights
- **Where:** `ref/model_weight.py`, `RareMarkerWeights.__init__` (`group_weight` dtype long -> float32); also E1.
- **Affected when:** group weights that are not integers (1.5 became 1, i.e. no upweighting).
- **Effect on results:** event weights and therefore the fit change for such configurations.
- **Affected analyses:** none found (`RareMarkerWeights` not used in reproducibility). **Action:** re-fit models that
  used fractional rare-marker weights. **Test:** v_all.py A6; f_ref.py.

### R2-B1 `PanelModule` component array shape
- **Where:** `ref/panel.py`, `PanelModule.from_panelconfig`. **What was wrong:** allocated (channel, channel, component)
  instead of (stain, channel, component). **Affected when:** spectral panels (fewer stains than channels); square panels
  are identical. **Effect on results:** none (crash -> works). **Action:** nothing. **Test:** v_all.py B1.

### R2-B2 spectra inverted transposed
- **Where:** `ref/model_transformer.py`, `get_standardisationparams_fromad` of `SpectralFitTransformer` and `ScalingTransformer`.
- **What was wrong:** `invert_spectra(spectra).T` on (channel, stain) spectra; `invert_spectra` expects (stain, channel).
- **Now:** `invert_spectra(spectra.T)`. For square panels the result is mathematically identical (inv(S^T)^T = inv(S)).
- **Effect on results:** none for square panels (all reproducibility panels); spectral panels crashed before. Verified
  that the spectral unmixing matrix now reproduces the abundances exactly (f_ref.py). **Action:** nothing. **Test:** v_all.py B2.

### R2-C2 `ScalingTransformer` warning
- **Where:** `ref/model_transformer.py`. **What was wrong:** `warnings` not imported, so `scale_stains=False` crashed.
  **Now:** imported, warning category `OverlapStandardisationWarning`. **Effect on results:** none. **Test:** v_all.py C2.

### R2-C9 default `DDIST_mode` typo
- **Where:** `ref/model.py`, `OverlapFitModel.__init__`: `"sinnkhorn"` -> `"sinkhorn"` (as in `from_data`). Direct
  construction without `DDIST_mode` crashed. **Effect on results:** none. **Test:** v_all.py C9; f_ref.py.

### R2-C10 `fit_factors_Sinkhorn(dummy=True)`
- **Where:** `ref/factor.py`. **What was wrong:** the frame of ones was built and discarded, so the full fit ran.
- **Effect on results:** callers with `dummy=True` now get the documented 1.0 instead of fitted factors.
- **Affected analyses:** none found (no `dummy=True` in reproducibility). **Action:** nothing. **Test:** v_all.py C10; f_ref.py.

### R2-C12 MMD with `DDIST_separate`; flag not exported
- **Where:** `ref/model.py`, `OverlapFitModel` (`__init__`, new `setup_ddist`, `estimate_MMD_bandwidth`,
  `set_MMD_bandwidth_from_dataset`, `export`, `from_exported`).
- **What was wrong:** `set_MMD_bandwidth_from_dataset` replaced the loss by a plain joint `MMDLoss`, dropping the
  per-marker wrapper; the batched per-marker helper does not work with `MMDLoss`; `export` did not store
  `DDIST_separate`, so a reloaded model continued training with the joint loss.
- **Now:** one `setup_ddist` builds the loss in both places; MMD uses a per-marker loop; with `DDIST_separate` the MMD
  bandwidth is estimated per marker (median over markers and samples), matching the 1-D loss; `DDIST_separate` is
  exported and restored.
- **Affected when:** `DDIST_mode="MMD"` with `DDIST_separate=True` (results change: the loss was silently joint), and
  continued training of reloaded models that used `DDIST_separate=True` with any mode.
- **Breaking change (no compatibility layer, user rule):** `from_exported`/`from_saved` require the `DDIST_separate` key.
  Models saved before this fix must be re-saved with the key added (`exported["DDIST_separate"] = False` unless the
  model was trained with True). Reproducibility loads saved models at `processing/batchint_data/lib/sources.py:284`,
  `figures/plots_batchintHD7/lib/parameters.py:34`, `figures/plots_dye/lib/dye.py:843` (see USER_REMINDERS).
- Decision (user 2026-11-07): keep it strict, no fallback for the missing key.
- **Action:** re-save existing saved OverlapFitModels (R7). **Test:** v_all.py C12; f_ref.py.

### R2-C13 batches without path to an anchor
- **Where:** `ref/data.py`, `graph_shortest_path` and `ConnectionSampler`.
- **What was wrong:** unreachable batches got distance 0 (the zero initialisation), so they were treated as anchors.
- **Now:** distances start at inf; unreachable non-anchor batches get no connections, a warning lists them, and
  `find_nontraining_batches` reports them.
- **Affected when:** anchor designs whose batch graph is disconnected from every anchor.
- **Effect on results:** those batches are no longer trained as (fixed-reference-like) anchors; their parameters stay
  untrained. Connected designs are unchanged (verified: chain distances 0/1/2).
- **Affected analyses:** none found directly (reproducibility loads trained models); any training on such a design
  changes. **Action:** check warnings when re-training. **Test:** v_all.py C13; f_ref.py.

### R2-D2 `get_unmixing_resistance(single_step=True)`
- **Where:** `ref/panel.py`, `PanelModule.get_unmixing_resistance` (default `single_step=True`).
- **What was wrong:** the perturbed embedding component was *set* to ±step instead of shifted by ±step (the
  multi-step sibling adds).
- **Effect on results:** unmixing resistance values for non-zero embeddings (zero embeddings unchanged).
- **Affected analyses:** none found (not used in reproducibility; internal caller `get_unmixing_resistance_fromad`).
- **Action:** re-compute stored unmixing resistances if any. **Test:** v_all.py D2.

### R2-D6 `plot_eval_NxN(key_color=...)`
- **Where:** `ref/plot.py`. **What was wrong:** one colour per unique label was passed where one per dataset was
  expected. **Effect:** plot colours only. **Affected analyses:** none found. **Test:** v_all.py D6.

### R2-E1 `requires_grad_` overrides
- **Where:** `ref/panel.py` (PanelModule, PanelEmbeddingModule, FixedPanelModule), `ref/model_weight.py`
  (RareMarkerWeights), `ref/model_transformer_smp.py` (UnmxErrorExclusionMultiplier), `torch/parameter.py`
  (ExtendableParameter).
- **What was wrong:** `FixedPanelModule` used the undefined `pem`, `UnmxErrorExclusionMultiplier` the missing
  `group_value`, dummy instances (no groups) failed, and all returned None instead of `self` (torch convention).
- **Effect on results:** none. **Test:** v_all.py E1; f_ref.py.

### R2-E8 / R2-E11 / R2-E17 / R2-K1 / R2-K4 (robustness, no result change)
- **E8** `ref/data.py` `ConnectionSampler`: no cross-batch connections now raises a clear `ValueError` (was `max()` of
  an empty array).
- **E11** `ref/factor.py` `fit_factor_PercentileMatching`: accepts any array-like (`np.asarray(v).ravel()`).
- **E17** `ref/model_transformer.py` `ScalingTransformer.anchor_batches`: accepts an int or a one-element list (bare
  `except` removed); several anchors raise `ValueError` as before. Docstring corrected (only one batch can be anchored).
- **K1** `ref/factor.py`: `raise NotImplemented(...)` -> `NotImplementedError` (4 places).
- **K4** `ref/factor.py` `fit_factor_Sinkhorn_grouped`: no sample shared between batches raises `ValueError` (was
  ZeroDivisionError); batches that share no sample are named in a warning (their factor is unconstrained).
- **Test:** v_all.py E8 E11 E17 K1 K4; f_ref.py.

### R2-K2 `incorporate_stain_scaling`
- **Where:** `ref/model.py`, `OverlapFitModel.incorporate_stain_scaling`; `ref/model_transformer.py`, `ScaleFactor.add_to_fixed`.
- **What was wrong:** the copy was always an `OverlapFitModel` (subclasses lost), a custom `cls_smp` could not be
  passed, and the combined scale was clipped by `max_abslogstain` without notice.
- **Now:** `type(self).from_exported(..., cls_smp=cls_smp)`; `ScaleFactor.add_to_fixed` warns
  (`OverlapStandardisationWarning`) when the result exceeds `max_abs_value`. Clipping itself is unchanged.
- **Effect on results:** none. **Affected analyses:** none found. **Test:** v_all.py K2; f_ref.py.

### R2-C15b dependencies
- **Where:** `pyproject.toml`. `pandas<3` pinned until the test suite (R4) runs on pandas 3 (copy-on-write may turn
  chained assignments into silent no-ops; user decision 2026-11-07 "Fix + pin <3 for now"); `joblib` declared (used by
  `utils/nxn.py`, so far only installed through scikit-learn). No change to the installed environment.

### R2 spill group (single-stain collection, stain fits, AutoSpill, panel PCA)
Applied by a worker agent, reviewed and re-verified by the lead (v_all.py, f_spill.py, f_spill_pd.py: log
r2vspill2_61325, all NOT REPRODUCED / ALL PASS; C15 also under pandas 3.0.6). The repro copy has the same bugs except
where noted. Reproducibility does not call these collection/fit functions directly, except E6 (refjohanna).
Small extensions beyond the approved text: C5 also anchors the default regex of `collect_bead_controls_spectralflow`;
E20 allows a subset of the panel's dyes (`allow_missing=True`).

### R2-C3 `from_custom(abort_inconsistent=False)` raised TypeError
- **Commit:** R2-spill. **Approval:** C3. **Where:** `spill/collection.py`, `SpilloverMeasurement.from_custom` (:107).
- **What was wrong:** `warnings.warn(..., SpilloverException)` passes an Exception class as warning category, so the
  "warn instead of abort" path raised TypeError. Now `SpilloverWarning`.
- **Affected when:** single stains with different `set_hash` values and `abort_inconsistent=False`.
- **Effect on results:** none (crash -> warning). **Affected analyses:** `from_custom` is called with the default
  `abort_inconsistent=True` in `figures/plots_overview/lib/overview.py:215`, `figures/plots_dye/lib/dye.py:67`,
  `figures/plots_rainbow/00_prepare_A3.ipynb`; unaffected. Same bug in the repro copy.
- **Action:** nothing. **Test:** v_all.py C3; f_spill.py "C3 warning class".

### R2-C5 single-stain file regex built from unescaped channel names
- **Commit:** R2-spill. **Approval:** C5. **Where:** `spill/utils.py`, `collect_bead_controls_regularflow` (:69);
  `spill/autospill.py`, `make_AutoSpill_fcs_config` (:34); default of `collect_bead_controls_spectralflow` (:107).
- **What was wrong:** the pattern `Compensation Controls_(.*){channel} Stained Control.fcs` used the channel name as
  regex: parentheses (all FACSAria configs, "B1 (525)"; S8) never matched, `.` matched any character, and the pattern
  was not anchored. Now `re.escape(channel)` and `Stained Control\.fcs$`; the spectral default is
  `Compensation Controls_(.*)\.fcs$`.
- **Affected when:** channel names containing regex metacharacters (FACSAria, S8). For channels without them, and for
  normal file names, nothing changes (only names with text after `.fcs` or a non-dot before `fcs` no longer match).
- **Effect on results:** changes outputs where such channels exist: before, their single stains were reported missing
  (warning), skipped by `collect_bead_controls_regularflow` (no data file / `panel_meta.csv` row without file), and
  padded with a dummy stain by `make_AutoSpill_fcs_config` (or "no single stains" if every channel had a parenthesis).
  Now they are found and collected.
- **Affected analyses:** none found in reproducibility. Collections or AutoSpill configs made earlier for FACSAria/S8
  with these functions lack those stains. Same bug in the repro copy.
- **Action:** re-collect FACSAria/S8 single-stain folders made with these functions, if any. **Test:** v_all.py C5
  (adapted: tests the package-built regex via `make_AutoSpill_fcs_config`); f_spill.py "C5/C7"; f_spill_pd.py.

### R2-C6 `warn_missing` and `filename_regex` ignored
- **Commit:** R2-spill. **Approval:** C6. **Where:** `spill/utils.py`, `collect_bead_controls_regularflow` (:74,
  `warn_missing=True` hard-coded), `collect_bead_controls_spectralflow` (:121, argument overwritten by the default).
- **What was wrong:** both arguments had no effect. Now passed on / used.
- **Affected when:** `collect_bead_controls_regularflow(warn_missing=False)` (warnings only);
  `collect_bead_controls_spectralflow(filename_regex=<non-default>)`.
- **Effect on results:** custom-regex callers now collect the files their regex matches (before: the default regex,
  often 0 rows). `warn_missing`: warnings only.
- **Affected analyses:** none found. Same bug in the repro copy.
- **Action:** nothing, unless a spectral collection was made with a custom regex. **Test:** v_all.py C6; f_spill_pd.py.

### R2-C7 AutoSpill padded channels got the same file name
- **Commit:** R2-spill. **Approval:** C7. **Where:** `spill/autospill.py`, `make_AutoSpill_fcs_config` (:46-49).
- **What was wrong:** the counter `i` was never incremented, so every padded (missing) channel got
  `<pad_file>_0.fcs`. Now `_0`, `_1`, ...
- **Affected when:** `pad_missing=True` (default) and more than one channel without a single stain.
- **Effect on results:** the AutoSpill input table now has distinct file names for the padded channels; how AutoSpill
  treated the duplicates before was not tested (R integration not exercised), so AutoSpill spectra could change.
- **Affected analyses:** none found. Same bug in the repro copy.
- **Action:** re-run AutoSpill runs that had more than one padded channel, if any. **Test:** v_all.py C7; f_spill.py.

### R2-C11 torch single-stain fit: plot KeyError, debug NameError
- **Commit:** R2-spill. **Approval:** C11. **Where:** `spill/stainfit.py`,
  `OutliersafeTorchSingleStainSpectrumFit.fit` (:436-441), `TorchSpectrumFitter.fit` (:349).
- **What was wrong:** `fit` stored only `slope`, so the inherited `plot` raised `KeyError 'intercept'`; `tqdm` was not
  imported, so `TorchSpectrumFitter(debug=True).fit` raised NameError.
- **Now:** `fit` also stores `intercept`, the intercept of each channel against the stain channel as in the linear and
  pos/neg fits: `offset_c - slope_c * offset_stain` (from `tsfitter.offset`; 0 for the stain channel); `tqdm` imported.
- **Affected when:** `OutliersafeTorchSingleStainSpectrumFit`; `TorchSpectrumFitter(debug=True)`.
- **Effect on results:** slopes unchanged; `spectrum` and the csv written by `write` gain an `intercept` column
  (`collect_single_fits` reads only `slope`). Synthetic check: torch intercept [0, 441.3, -110.0] vs linear
  [0, 440.3, -110.1] (true [0, 440, -110]).
- **Affected analyses:** none found. Same bug in the repro copy.
- **Action:** nothing. **Test:** v_all.py C11; f_spill.py "C11 ...".

### R2-C14 PCA component count not capped by the channel count
- **Commit:** R2-spill. **Approval:** C14. **Where:** `spill/pca.py`, `SingleSpectrumPCAFitter.__init__` (:24).
- **What was wrong:** `N = min(N, n_spectra)`; sklearn's PCA needs `N <= min(n_spectra, n_channels)`, so panels with
  fewer channels than N and more spectra than channels crashed. Now `min(N, n_spectra, n_channels)`.
- **Affected when:** fewer channels than `N` (default 5) and more spectra than channels.
- **Effect on results:** none (crash -> works; N is unchanged wherever it ran before).
- **Affected analyses:** used via `SpilloverCollection` / `PanelConfiguration.collect_PCA_data`
  (`figures/plots_dye/lib/dye.py:707-708`, `figures/plots_dye/04_panel_choice.ipynb`); these ran, so unchanged. Same
  bug in the repro copy.
- **Action:** nothing. **Test:** v_all.py C14 (check fixed to format a non-exception result); f_spill.py.

### R2-C15a str written into a float `set_hash` column
- **Commit:** R2-spill. **Approval:** C15 (code part; the `pandas<3` pin is R2-C15b). **Where:** `spill/utils.py`,
  `collect_bead_controls_regularflow` (:82/:94), `collect_bead_controls_spectralflow` (:133/:143).
- **What was wrong:** `set_hash` was created as a float NaN column and then given str hashes: a FutureWarning under
  pandas 2, `TypeError` under pandas 3. Now created as an object column (NaN where no file), so the NaN filter and the
  csv output are unchanged.
- **Affected when:** every call of these two functions.
- **Effect on results:** none (`panel_meta.csv` identical; pandas 3: crash -> works).
- **Affected analyses:** none found. Same bug in the repro copy.
- **Action:** nothing. **Test:** v_all.py C15; f_spill_pd.py (pandas 2.3.3 and 3.0.6).

### R2-E6 `LocalOutlierFactor` neighbours 0..4 for small files
- **Commit:** R2-spill. **Approval:** E6. **Where:** `spill/stainfit.py`,
  `OutliersafeLinearSingleStainSpectrumFit.filter_outlier` (:237/:239), `.filter_outlier_residuals` (:255).
- **What was wrong:** `n_neighbors = N_events // neighbors_factor` (default 100) is 0 below 100 events (sklearn error)
  and 1-4 below 500 (a very noisy outlier score). Now `max(N_events // neighbors_factor, 5)`.
- **Affected when:** fits on fewer than 5*`neighbors_factor` (500) events; also `OutliersafeTorchSingleStainSpectrumFit`
  (inherits the filter).
- **Effect on results:** 100-499 events: different outlier mask, so the fitted slopes/intercepts can change; <100 events:
  crash -> works. >= 500 events: unchanged.
- **Affected analyses:** `processing/refjohanna/lib/refjoh.py:65` (bead single stains after scatter gate) and `:77`
  (LIVE/DEAD Blue cells with UV5-A > 1e4): change only if one of these inputs has fewer than 500 events; event counts
  not checked here. Same bug in the repro copy.
- **Action:** check the event counts of the refjohanna single-stain fits; re-run that step only if one is < 500.
  **Test:** v_all.py E6; f_spill.py "E6 small files".

### R2-E10 `_find_single_stain` on a folder without .fcs files
- **Commit:** R2-spill. **Approval:** E10. **Where:** `spill/utils.py`, `_find_single_stain` (:22).
- **What was wrong:** `max()` of an empty list raised ValueError. Now the stain is reported as not found (warning,
  `("", "")`), so `make_AutoSpill_fcs_config` raises its clear "Could not find any single stains" error.
- **Affected when:** single-stain folder without .fcs files.
- **Effect on results:** none (obscure crash -> warning / clear error). **Affected analyses:** none found. Same bug in
  the repro copy.
- **Action:** nothing. **Test:** v_all.py E10; f_spill_pd.py.

### R2-E20 `plot_test_spectra_reconstruction`: log base, axis, row order
- **Commit:** R2-spill. **Approval:** E20. **Where:** `spill/panel.py`,
  `PanelConfiguration.plot_test_spectra_reconstruction` (:218-246).
- **What was wrong:** the gray 100%/10% guide lines used `np.log` as exponent of `np.logspace` (base 10), and the
  right panel used the left panel's limits; the reconstructions (indexed by `self.stains`) were compared with `spectra`
  by position, so rows in a different order were mismatched (and a subset of the panel's dyes raised ValueError).
- **Now:** `np.log10`, each panel its own limits; `spectra` is restricted to `self.channels` and the reconstructions are
  aligned to its rows/columns (`allow_missing=True`, so a subset of dyes works).
- **Affected when:** every call (guide lines); rows of `spectra` not in panel order (errors).
- **Effect on results:** plot only, plus the returned error Series when rows were not in panel order.
- **Affected analyses:** none found. vs repro copy: it has no guide lines at all, but the same row-order bug.
- **Action:** nothing. **Test:** v_all.py E20; f_spill.py "E20 aligned rows".

### R2-F4 internal column `channnel_name` renamed
- **Commit:** R2-spill. **Approval:** F4 (user: rename, internal only). **Where:** `spill/autospill.py`,
  `make_AutoSpill_fcs_config` (:33-38).
- **What was wrong:** misspelt internal column. Now `channel_name`; it is not part of the returned frame.
- **Effect on results:** none. **Affected analyses:** none. **Action:** nothing. **Test:** v_all.py F4.

### R2 core group (torch, utils, io, gating)
- Commit: R2-core (review/R2-bugs), 2026-11-07. Checks: jobs/r2_verify/v_all.py for the IDs below (all NOT REPRODUCED except F3, CONFIRMED by design), f_core.py (all PASS).
- Dependency: `tzdata` declared in pyproject (F1 uses `zoneinfo`; containers without /usr/share/zoneinfo need the Python package; it was already installed via pandas).
- Repro copy: same code as publish for every item, except K5 (hard-coded `layer="integrated"`).

### R2-A5 geomloss event weights normalised per side
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item A5.
- Where: torch.geomloss.GeomSampleLoss.forward (torch/geomloss.py:41-47 before the fix).
- What was wrong: the weights were passed to geomloss as given. With `reach=1` the loss is unbalanced, so total mass
  matters; `RareMarkerWeights` sums to N (correct for MMD), and the same data gave 70.9 instead of 0.13. If only one side
  had weights, both were silently dropped.
- Now: each weight vector is divided by its sum (also batched, `(B, N)` weights along the last axis, as used by
  `DDIST_separate`); a missing side gets uniform 1/N; both None is the unweighted loss as before.
- Affected when: `OverlapFitModel` with a `markerweight` that returns weights (`RareMarkerWeights` or custom), and
  `DDIST_mode` "sinkhorn" (default) or "hausdorff"; direct weighted `GeomSampleLoss` calls with weights not summing to 1
  or with one side None. Not affected: `DummyWeights` (default, returns None), MMD, and the `ref/factor.py` Sinkhorn fits
  (they never pass weights).
- Effect on results: changes outputs for those fits: the DDIST loss values and therefore fitted transformer parameters
  (relative weight of the sinkhorn term changes by ~N-fold; check: unweighted 0.1306, sum-N weights now 0.1306 instead of
  70.94). Same vs publish and vs repro copy (identical code there).
- Affected analyses (corrected 2026-11-07, R2c; the first search covered only cytovanni_reproducibility, which loads the
  HD7 models but does not fit them): the HD7 Identity models of the paper, fitted in ~/hd7_myeloma/cytovanni/pipeline:
  - hd7-03-spectra-integration.ipynb, Identity refined spectral fit: `RareMarkerWeights` [("PE (BD)", 5e3, 5),
    ("BV750", 5e3, 5), ("BV605", 9e3, 5)], sinkhorn (with `UnmxErrorExclusionMultiplier`).
  - hd7-05-additional-scale.ipynb, Identity post fit: CD34 (5e3, x5), CD56 (9e3, x5), `DDIST_separate=True` (batched
    (B, N) weight path).
  These are the Identity Cytovanni layers in batchint_data (normal, normal_post, onlyref_post, ...). TCell fits use no
  weights (not affected). Loading the saved models is unaffected; a refit with the fixed package will not reproduce them.
  No other `markerweight` / `GeomSampleLoss` use in cytovanni_reproducibility processing/ or figures/.
- Action: decide refit vs keep the saved models and document in Methods (user, 2026-11-07: "settle #1 at the end").
- Test: jobs/r2_verify/v_all.py A5; jobs/r2_verify/f_core.py "A5".

### R2-D5 `CustomMultivariateNormal.rsample` ignored `sample_shape`
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item D5.
- Where: torch.dist.CustomMultivariateNormal.rsample (torch/dist.py:48).
- What was wrong: the noise had shape `loc.shape`, so `rsample((5,))` returned one sample.
- Now: noise of shape `self._extended_shape(sample_shape)`, as in torch's MultivariateNormal.
- Affected when: `rsample`/`sample` with a non-empty shape on this distribution (also through `ASinhMultivariateNormal`
  with precomputed precision logdet). No package code samples from it.
- Effect on results: none for existing outputs (wrong shape -> right shape).
- Affected analyses: none found.
- Action: nothing.
- Test: jobs/r2_verify/v_all.py D5; f_core.py "D5" (shape and covariance).

### R2-E2 `apply_unmixing_inv` error message NameError
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E2.
- Where: utils.unmx.apply_unmixing_inv (utils/unmx.py:50).
- What was wrong: the type check's message referred to the undefined `spectra`, so a NameError replaced the ValueError.
- Affected when: inverted spectra passed as non-DataFrame.
- Effect on results: none (error message only).
- Affected analyses: none (refjoh.py:122 and sarcoma unmx_error.py:14 pass DataFrames).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E2.

### R2-E3 `label_axis_ArcSinh` crash on narrow ranges
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E3.
- Where: utils.misc.label_axis_ArcSinh (utils/misc.py:269).
- What was wrong: with a single major tick (e.g. only 0) `np.hstack([])` raised.
- Now: no minor ticks in that case. arges/lucy: see todo/ARGES_LUCY.md (signature unchanged).
- Affected when: axis ranges narrow enough to contain only one power of ten.
- Effect on results: none (crash -> works); plots that worked are unchanged.
- Affected analyses: figures/plots_liechti/lib/plots.py:150, figures/plots_batchintHD7/lib/problems.py:235-248,342 use it
  (they worked, so unchanged).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E3; f_core.py "E3".

### R2-E4 `anndata_subsample`/`df_subsample` ignored `copy`
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E4.
- Where: utils.misc.anndata_subsample, df_subsample (utils/misc.py:309-325).
- What was wrong: inputs with <= N rows were returned as the input object even with `copy=True`.
- Now: `copy=True` always returns a new object; `copy=False` returns the input when small, or (AnnData) a view.
- Affected when: inputs with <= N events whose result was later modified in place (the modification reached the input).
- Effect on results: none for values; only aliasing.
- Affected analyses: none found (not called in cytovanni_reproducibility).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E4; f_core.py "E4".

### R2-E5 mutable default arguments shared between instances
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E5.
- Where: utils.config.CytometerConfiguration.__init__ (utils/config.py:76-87); io.utils.readfcs_sample (io/utils.py:147),
  io.utils.writefcs (io/utils.py:297), io.rainbow.readfcs_rainbow_sample.
- What was wrong: the default `laserdict`, `replace_channelname_dict` and `default_scatter_calibrate` dicts were stored
  directly, so all instances using the default (and the caller's dict) shared one object.
- Now: the config stores copies (deep copy for the dicts of lists); the io functions use `None` defaults.
- Affected when: code that mutated one config's dicts (the edit appeared in every other config using the default).
- Effect on results: none unless such mutation happened.
- Affected analyses: none found.
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E5; f_core.py "E5".

### R2-E9 `read_folder_timestamps` on an empty folder
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E9.
- Where: io.utils.read_folder_timestamps (io/utils.py:280).
- What was wrong: without .fcs files the frame had no "UTC" column -> KeyError.
- Now: the frame is created with its columns; an empty folder gives an empty frame.
- Affected when: folder without .fcs files.
- Effect on results: none (crash -> empty result).
- Affected analyses: none found (not called in cytovanni_reproducibility).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E9; f_core.py "E9/E12".

### R2-E12 `pdsave_collect` swallowed every exception
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E12.
- Where: torch.pdsave.pdsave_collect (torch/pdsave.py:39).
- What was wrong: any error while rebuilding a pandas object was caught and the raw tuple returned silently.
- Now: only tuples whose first element is a `"_pdsave_expanded_*"` string are treated as expansions (all other values,
  including ordinary tuples, pass through as before); errors while rebuilding them propagate.
- Affected when: corrupted/inconsistent pdsave tuples in exported models.
- Effect on results: none for valid data (silent wrong object -> error).
- Affected analyses: none found.
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E12 (check adapted: the error now propagates); f_core.py "E9/E12".

### R2-E15 `base_plot_NxN*`: dask branch, `colors_df` transposed, `label_arcsinh`
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E15.
- Where: utils.nxn.base_plot_NxN_ds (utils/nxn.py:165, 227), utils.nxn.base_plot_NxN (utils/nxn.py:39, 69).
- What was wrong: `dask` was forced to False (dead branch, and an undeclared dependency); `base_plot_NxN` indexed
  `colors_df.loc[ykey, xkey]` while the docstring and the `_ds` version use `[xkey, ykey]`; `label_arcsinh` did nothing.
- Now: `dask` argument and branch removed (joblib only, declared by K8); `colors_df.loc[xkey, ykey]` in both;
  `label_arcsinh` argument removed. Breaking: passing `dask=` or `label_arcsinh=` raises TypeError.
- Affected when: `base_plot_NxN(colors_df=...)` with an asymmetric `colors_df` (colours of the upper triangle panels).
- Effect on results: plot colours only.
- Affected analyses: none found (no `base_plot_NxN*`, `colors_df`, `dask` use in cytovanni_reproducibility; ref/plot.py
  calls without these arguments).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E15.

### R2-E16 `get_asinh_single_CV_varcorr` NaN for negative corrected variance
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E16.
- Where: utils.evl.get_asinh_single_CV_varcorr (utils/evl.py:270-300).
- What was wrong: if the mean error variance exceeded the observed variance, `sqrt` of a negative number gave NaN silently.
- Now: clamped to 0 with a CytovanniWarning (CV 0), as `get_asinh_paired_CV_varcorr` already clamps (there without a warning).
- Affected when: error variance > observed variance.
- Effect on results: those CVs change from NaN to 0.
- Affected analyses: figures/plots_batchintHD7/lib/cv.py:36 defines an alias, but it is not called anywhere (none).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E16; f_core.py "E16".

### R2-E18 `fix_sample_bug(delete_spill=True)` kept `$SPILLOVER`
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E18.
- Where: io.utils.fix_sample_bug (io/utils.py:322).
- What was wrong: only flowkit's `spill` key was removed; `$SPILLOVER` (key `spillover`) survived.
- Now: both are removed.
- Affected when: source files that carry `$SPILLOVER`.
- Effect on results: the rewritten .fcs no longer contains `$SPILLOVER` (software reading it no longer compensates).
- Affected analyses: none found (not called in cytovanni_reproducibility). Files written earlier with `delete_spill=True`
  may still carry `$SPILLOVER`.
- Action: re-write such files if they are compensated downstream.
- Test: jobs/r2_verify/v_all.py E18.

### R2-E19 unused deprecated `torch.cuda.amp` import
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E19.
- Where: torch/utils.py:2 (plus the commented-out decorators and their docstring note).
- Effect on results: none.
- Affected analyses: none. Action: nothing. Test: jobs/r2_verify/v_all.py E19.

### R2-E21 `apply_df` for LineGate, CutoffUnmxGate, UnmxMinGate, CombinedGate
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item E21.
- Where: gating/line.py, gating/unmx.py, gating/utils.py.
- What was wrong: no `apply_df`, so `make_calibrated_fcs(gates=[...])` with these gates raised AttributeError.
- Now: `apply_df(df)` like the existing PolygonGate/MinMaxGate versions (boolean array; df columns are what the gate
  gates on). LineGate: columns x, y (same scaling as `apply`; distance code shared via `get_distance_points`);
  CombinedGate: all gates' `apply_df`; CutoffUnmxGate/UnmxMinGate: df holds the unmixed data (what `ad.obsm[key]` holds).
  Note: the raw frame in `make_calibrated_fcs` has no unmixed columns, so the two unmx gates raise KeyError there.
- Affected when: these gates passed to `apply_df` callers.
- Effect on results: none (crash -> works).
- Affected analyses: none found.
- Action: nothing.
- Test: jobs/r2_verify/v_all.py E21; f_core.py "E21" (apply_df == apply).

### R2-K1 (geomloss part) `raise NotImplemented` -> `NotImplementedError`
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item K1.
- Where: torch.geomloss.GeomSampleLoss.__init__ (torch/geomloss.py:34). (ref/factor.py part: lead.)
- What was wrong: `NotImplemented` is not callable -> TypeError instead of the intended error.
- Effect on results: none (error type only). Affected analyses: none. Action: nothing.
- Test: jobs/r2_verify/v_all.py K1; f_core.py "A5" (unknown mode).

### R2-K3 `gate_scatter_bead_sample(**gating_kwargs)` keyword clash
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item K3.
- Where: gating.bead.gate_scatter_bead_sample (gating/bead.py:37-49).
- What was wrong: kwargs naming a setting the function passes itself (e.g. `final_density_cutoff`) raised
  "got multiple values for keyword argument".
- Now: the defaults (per robust/very_robust branch, plus verbose/name_x/name_y/trim_beadmode) are put in a dict and
  `gating_kwargs` override them.
- Affected when: such kwargs. Calls that worked before get identical arguments.
- Effect on results: none for working calls.
- Affected analyses: refmoh.py:63, refjoh.py:64, instrument_rainbow pipeline.py:32, gosling.py:350 (`main_tile`,
  `merge_tiles_min_saddle`: no clash) -> unchanged.
- Action: nothing.
- Test: jobs/r2_verify/v_all.py K3; f_core.py "K3".

### R2-K5 `add_adata_unmixed` ignored `layer`
- Commit: R2-core (review/R2-bugs), 2026-11-07; default set in R2-K5-G1. Approval: todo/R2_approval.md item K5; default
  decided by the user 2026-11-07 ("it should keep the default behaviour it had before").
- Where: utils.unmx.add_adata_unmixed.
- What was wrong: the data was always taken from layer "calibrated" (repro copy: "integrated"), whatever `layer` said;
  the signature default `layer="raw"` was never used.
- Now: from `layer`; the default is `layer="calibrated"`, so default calls behave as before.
- Affected when: calls passing `layer` other than "calibrated" (now honoured).
- Effect on results: none for default calls; `adata.obsm[addkey]` (+ `_arcsinh`) change for explicit other layers.
  Vs the repro copy (hard-coded "integrated"): different layer name, see R7.
- Affected analyses: none found (no calls in cytovanni_reproducibility or in the package).
- Action: nothing.
- Test: jobs/r2_verify/v_all.py K5; f_core.py K5.

### R2-K7 invalid escape in utils/rate.py docstring
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item K7.
- Where: utils.rate.EventRateEstimator docstring (utils/rate.py:5) -> raw string. arges/lucy: see todo/ARGES_LUCY.md.
- Effect on results: none. Affected analyses: none. Action: nothing. Test: jobs/r2_verify/v_all.py K7.

### R2-F1 FCS start time in the instrument time zone
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item F1, group 3 decision "Instrument timezone".
- Where: io.utils.readfcs_sample (io/utils.py:229), io.utils.read_fcs_timestamp (io/utils.py:264),
  io.utils.read_folder_timestamps; utils.config.CytometerConfiguration.
- What was wrong: `dateutil.parser.parse(date+"-"+btim).timestamp()` interprets the naive FCS time in the time zone of
  the reading machine, so `uns["UTC"]` and the uid (`<cytometer>-<int(UTC)>`) depended on where/how the file was read.
- Now: new `CytometerConfiguration.timezone` (IANA name, default "Europe/Berlin", validated, exported/imported);
  `_fcs_utc` attaches it with `zoneinfo` before `.timestamp()`. `readfcs_sample` uses `cytoconfig.timezone`;
  `read_fcs_timestamp(filepath, timezone="Europe/Berlin")`; `read_folder_timestamps(folder, cytoconfig=None,
  timezone=None)` (cytoconfig.timezone, else Berlin).
- Affected when: reading on a machine/container whose local time is not Europe/Berlin. Note: the dev image has no
  /usr/share/zoneinfo; there `TZ=Europe/Berlin` silently means UTC to the C library (zoneinfo uses the `tzdata` Python
  package, installed with pandas). In jobs on this cluster TZ is unset and /etc/localtime (Europe/Berlin) applies, so
  results were Berlin time.
- Effect on results: `uns["UTC"]` and `uns["uid"]` (and everything keyed by uid: `name_uid`, rainbow calibrator batches,
  sort orders by UTC) change by the UTC offset of the old reading machine relative to Berlin; none for files read on
  Berlin time (verified: old formula under Berlin time == new value, winter and summer). Instruments not in Berlin keep
  the Berlin interpretation unless their config sets `timezone` (then their UTC/uid change, by design).
- Breaking: config objects pickled before this change have no `timezone` attribute (AttributeError in `readfcs_sample`);
  configs restored from exports (h5ad/json) get the default.
- Affected analyses: readfcs_rainbow_sample/readfcs_sample in processing/refmohammed, refjohanna, instrument_rainbow,
  gosling, figures/plots_sarcoma_thesis/00_prepare.ipynb, plots_calib, plots_rainbow/01_gating_gmm.ipynb; refmoh.py:77
  sorts by UTC. All run on this cluster (Europe/Berlin local time) -> unchanged. Gosling (NorthernLights, recorded outside
  Germany) keeps the Berlin interpretation by default, as before.
- Action: nothing for analyses read in Germany; uids/UTC from files read elsewhere (e.g. a UTC laptop/CI) differ.
- Test: jobs/r2_verify/v_all.py F1; f_core.py "F1" (TZ UTC0 / JST-9 / Berlin POSIX / node default give identical UTC;
  2025-01-01 12:00 -> 1735729200, 2025-07-01 12:00 -> 1751364000; Tokyo config -8h/-7h).

### R2-F2 config export keeps class, hash function, timezone and subclass extras
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item F2, group 3 decision "Save names, raise".
- Where: utils.config.CytometerConfiguration.to_dictionary / import_exported / load_json (utils/config.py:159-198);
  io.bead.read_bead_h5ad, io.rainbow.read_rainbow_h5ad.
- What was wrong: the export dropped `fct_hash_settings` (a custom hash came back as the default, so batch inference
  failed), the class (always re-created as the base class) and subclass attributes (`lasers`, `lasers_wavelength`,
  `lasers_color`, `alt_name_dct`); `load_json` on a subclass crashed (`cls([])`).
- Now: export adds `config_class`, `fct_hash_settings` (registered name), `timezone`, and the attributes listed in the new
  class attribute `_export_extras` (pd.Series stored as index/values lists, h5ad- and json-safe). New
  `CytometerConfiguration.from_exported(exported)` re-creates the exported subclass (base defaults + exported attributes,
  subclass `__init__` not called); `load_json` and the h5ad readers use it. New `register_hash_function(fct, name=None)`
  (exported from cytovanni.utils, registry `HASH_FUNCTIONS`); import raises CytometerConfigurationException for an
  unregistered name (always for lambdas, e.g. the documented `lambda adata: 0`), for an unknown class, and when
  `import_exported` is called on an instance of a different class.
- Affected when: writing/reading configs (`write_json`/`load_json`, `write_*_h5ad`/`read_*_h5ad`).
- Effect on results: none for existing files (no `config_class` -> base class, default hash, Berlin, as before). Files
  written from now on re-create e.g. `CytometerConfiguration_FACSAriaIII` instead of the base class. Breaking: configs with
  a custom/lambda hash function cannot be read back without registering it (previously they silently got the default).
- Affected analyses: read_rainbow_h5ad in refmoh.py:76/154-155, instrument_rainbow pipeline.py:269, comparison.py:210,
  gosling.py:406, plots_calib/calib.py:84 + 00_prepare_rainbowtest.ipynb, plots_rainbow/rainbow.py:61,
  plots_overview/overview.py:46 (existing files, default hash -> unchanged). No custom `fct_hash_settings` anywhere.
- Action: nothing.
- Test: jobs/r2_verify/v_all.py F2 (check adapted: unknown custom function raises); f_core.py "F2" x2 (11 classes x
  dict/json/h5ad round trip; registered/unregistered/lambda/old export/class mismatch/bad time zone).

### R2-D4 batched eccentricity: bottom-edge test on the padded height
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item D4, group 3 decision "Keep both, fix D4".
- Where: io.S8.calculate_batched_eccentricity_angle (io/S8.py:208-220).
- What was wrong: masks are padded to the tallest height; the bottom-edge test used the padded last row, so cells cut off
  at their own bottom edge got a value (the single version returns NaN), depending on the other masks in the batch.
- Now: tests each mask's own last row (`heights-1`). Batched == single on 300 random masks incl. 35 cut off at the bottom.
- Affected when: `add_S8_singlet_maskfeatures` (only caller) with masks of different heights, cells touching their own
  bottom edge.
- Effect on results: `cell_eccentricity` and `cell_deformangle` of those cells become NaN (were finite). Same vs publish
  and repro copy.
- Affected analyses: none found in cytovanni_reproducibility (no `add_S8_singlet_maskfeatures`/eccentricity use);
  arges-computed S8 features. arges/lucy: see todo/ARGES_LUCY.md.
- Action: re-compute S8 mask features if used.
- Test: jobs/r2_verify/v_all.py D4; f_core.py "D4".

### R2-F3 `add_S8_singlet_maskfeatures` kept, duck-typed image container documented
- Commit: R2-core (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md item F3, group 3 decision "Keep both, fix D4".
- Where: io.S8.add_S8_singlet_maskfeatures (io/S8.py:272).
- Now: docstring states that `imgcont` is any object with `load_masks_batched(index)` (arges `ImageContainer`, not part of
  this package) and lists the features written. No code change.
- Effect on results: none. arges/lucy: see todo/ARGES_LUCY.md.
- Test: jobs/r2_verify/v_all.py F3 stays CONFIRMED by design (the container is still not in the core).

### R2-G1 `GeomSampleLoss("hausdorff")` could not be constructed
- Commit: R2-K5-G1 (review/R2-bugs), 2026-11-07. Approval: todo/R2_approval.md addendum G1 (user 2026-11-07).
- Where: torch.geomloss.GeomSampleLoss.__init__.
- What was wrong: used `geomloss.kernel_samples.gaussian_kernel` without importing the submodule -> AttributeError, so
  `OverlapFitModel(DDIST_mode="hausdorff")` was unusable (same in publish and the repro copy).
- Now: imports `gaussian_kernel` from `geomloss.kernel_samples` (geomloss 0.2.x) or `geomloss._legacy.kernel_samples`
  (0.3.x, which the container installs).
- Effect on results: none (crash -> works). Affected analyses: none (hausdorff not used).
- Test: jobs/r2_verify/f_core.py G1.

## R2b (round-2 bug fixes, branch review/R2b-bugs; approvals and decisions in todo/R2b_approval.md)

### R2b group A (rainbow, gating, io, spill, config)
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Checks: jobs/r2b_verify/v2_all.py + v2_gpu.py (NOT REPRODUCED), f2_A.py ALL PASS; jobs/r2_verify/f_rainbow.py, f_spill.py ALL PASS (f_spill C14 expectation adapted to G2-15).

### R2b-G2-01 `get_rainbow_peak_parameters` / `plot_rainbow_with_PDF` crash after a GPU fit
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-01.
- Where: rainbow.rbim.RainbowCalibrator.get_rainbow_peak_parameters (rainbow/rbim.py:331), used by
  plot_rainbow_with_PDF (:468).
- What was wrong: the helper dataset was always built on the CPU, while `fit_fluorescence(gpu=True)` leaves `rfim` on
  cuda, so the parameter lookup raised a cuda/cpu RuntimeError.
- Now: the dataset is built on the device of `rfim` (`next(self.rfim.parameters()).device`); outputs stay numpy on the CPU.
- Affected when: a `RainbowCalibrator` fitted on the GPU and used in the same session (models loaded with `from_saved`
  are on the CPU and worked).
- Effect on results: none (crash -> works). GPU values equal the values after `from_saved` (f2_A.py).
- Affected analyses: cytovanni_reproducibility processing/gosling/lib/gosling.py:427 works around it (moves rfim to CPU
  before :475); comparison.py:228 and figures/plots_rainbow/00_prepare_A3.ipynb use `from_saved` models. The workaround
  can stay or be dropped. Pipeline: none found.
- Action: nothing.
- Test: jobs/r2b_verify/v2_gpu.py G2-01 (--gpu); f2_A.py "G2-01".

### R2b-G2-02 configuration read from .h5ad cannot be written to json
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-02.
- Where: utils.config.CytometerConfiguration.import_exported (utils/config.py:244-265).
- What was wrong: .h5ad returns lists inside `laserdict` / `default_scatter_calibrate` as numpy arrays; only top-level
  list keys and `_export_extras` arrays were converted, so `write_json` raised TypeError (R2-F2 was incomplete).
- Now: the exported dict is converted recursively to builtins (new `_to_builtin`: ndarray -> list, numpy scalar ->
  Python scalar, also inside dicts/lists/tuples) before the attributes are set; the separate ndarray branch for
  `_export_extras` was removed (covered).
- Affected when: a configuration that went through .h5ad (`read_rainbow_h5ad`, AnnData `uns["cytoconfig"]`) and is then
  exported to json, or compared with `==` on these attributes.
- Effect on results: none (crash -> works). The restored attributes are lists instead of arrays; the export equals the
  original's.
- Affected analyses: none found (no `write_json` / `from_exported` use in cytovanni_reproducibility or the pipeline).
- Action: nothing.
- Test: jobs/r2b_verify/v2_all.py G2-02; f2_A.py "G2-02".

### R2b-G2-05 `extend_fit`: orphan new rainbow types got arbitrary factors
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-05.
- Where: rainbow.rbim.RainbowCalibrator.extend_fit (rainbow/rbim.py:194-212); the check was only in
  rainbow.rbim_scatter.RainbowScatterCalibrationModule.extend_standalone (rainbow/rbim_scatter.py:234-248).
- What was wrong: a new rainbow type measured only in new batches, without any known type in those batches, makes the
  type and batch parameters degenerate (scatter and fluorescence). `extend_fit` fitted it anyway: in the lead check the
  scatter factor C/A was 2.40 / 3.39 (FSC/SSC) instead of 1.25, without a warning.
- Now: the check is `RainbowBatchOrdinalEncoder.check_orphan_types(df)` (rainbow/rboe.py), called by `extend_fit` before
  anything is changed (encoder, `self.adatas`) and by `extend_standalone`. It raises `CalibrationModuleException`
  naming the types.
- Affected when: `extend_fit` with a new rainbow type whose batches are all new and contain no known type.
- Effect on results: no change for valid extensions (new type in a known batch, or together with a known type). Orphan
  extensions now raise instead of producing arbitrary calibration factors for the new batches.
- Affected analyses: none. The only `extend_fit` (pipeline hd7-01, cell 20) adds `20240924_comp_*` batches with the known
  type 3285117. cytovanni_reproducibility: no `extend_fit` call.
- Action: nothing.
- Test: jobs/r2b_verify/v2_all.py G2-05; f2_A.py "G2-05".

### R2b-G2-08 `LineGate`: kept side depended on the order of p1 and p2
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-08 (user: "Below = lower y").
- Where: gating.line.LineGate.__init__ / get_distance_points (gating/line.py:27-50), plot (:79).
- What was wrong: the sign of the distance came from the direction p1 -> p2, so `below=True` kept the points below the
  line for p1.x < p2.x and above it for p1.x > p2.x. `plot` always shaded as if "below" meant lower y; for
  `below=False` it shaded down to y=0 instead of the lower axis limit.
- Now: the points are ordered by x in `__init__`, so `below=True` always keeps lower y; `p1[0] == p2[0]` (vertical line)
  raises ValueError. `plot` shades the removed side (above for `below=True`, below to the axis limit otherwise).
  Docstrings for the class, `get_distance_points` and `plot`.
- Affected when: `LineGate` with p1[0] > p2[0] (kept side flips to the documented one); vertical lines (now an error);
  plots with `below=False`.
- Effect on results: changes gate masks (`apply`, `apply_df`, `obs[addkey]`) for gates defined with p1.x > p2.x: they now
  keep the other side, as documented.
- Affected analyses: none found (no `LineGate` in the package, cytovanni_reproducibility or the pipeline).
- Action: nothing; anyone with own LineGates defined right-to-left: check the kept side.
- Test: jobs/r2b_verify/v2_all.py G2-08; f2_A.py "G2-08".

### R2b-G2-09 `fit_rainbow_GMM` retry ignored empty components
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-09.
- Where: rainbow.gmm.fit_rainbow_GMM (rainbow/gmm.py:109-112).
- What was wrong: the label shares came from `np.unique`, which omits components with no beads, so an empty component
  was not seen. Even with zero counts, the retry needed both a small and a large share; with 7 even peaks + 1 empty
  (N=8) no retry happened.
- Now: counts with `np.bincount(labels, minlength=N)`; the fit counts as failed if the shares are uneven (as before) or
  any component is empty. Failed -> retry with a new seed (up to `Nretry`), then a GMMAssignmentWarning (also with
  `Nretry=0`). Docstring of `warningthreshold` / `Nretry`.
- Affected when: a GMM fit in which a component gets no beads (sklearn rarely leaves a component exactly empty; a lost
  peak usually keeps a tiny nonzero share, which was already caught if another share is large).
- Effect on results: rare. Such samples are now refitted (different GMM labels, means, gate) or, if all retries fail,
  warned about. Samples without empty components are unchanged. Same vs repro copy (it has the `np.unique` version).
- Affected analyses: every `fit_rainbow_GMM` call (pipeline hd7-01; reproducibility gosling.py:357/371,
  instrument_rainbow/lib/pipeline.py:64/68, refmoh.py:64/66, plots_calib/lib/calib.py:1213/1215,
  figures/plots_rainbow/01_gating_gmm.ipynb). No evidence that any of them hit an empty component (not checked on the data:
  a re-run shows it as new `GMMAssignmentWarning`s).
- Action: nothing; if re-running hd7-01, look for new GMMAssignmentWarnings.
- Test: jobs/r2b_verify/v2_all.py G2-09; f2_A.py "G2-09".

### R2b-G2-10 `df_channel_train` only set with `warn_missing=True`
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-10.
- Where: rainbow.rbim_data.RainbowFluorescenceGMMCalibrationDataset.setup_data (rainbow/rbim_data.py:262-264),
  check_batch_training_missing (:172).
- What was wrong: `df_channel_train`, which `RainbowFluorescenceGMMCalibrationTrainer.fit_init` needs, was only computed
  when the warning was requested, so a dataset with `warn_missing=False` failed with AttributeError in the trainer.
- Now: `check_batch_training_missing(warn=...)` always sets `df_channel_train`; only the warning depends on `warn_missing`.
- Affected when: datasets built with `warn_missing=False` and passed to the trainer (`fit_fluorescence` always used True).
- Effect on results: none (latent crash -> works).
- Affected analyses: none. `get_rainbow_peak_parameters` (package) and repro comparison.py:221 use `warn_missing=False`
  without fitting; they now also compute the table (no output change).
- Action: nothing.
- Test: jobs/r2b_verify/v2_all.py G2-10; f2_A.py "G2-10".

### R2b-G2-13 `log(0)` RuntimeWarning in every bead gate
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-13.
- Where: gating.autospill.HistogramTesselator (gating/autospill.py:179).
- What was wrong: the binned KDE (R1-D1) is exactly 0 far from the data, so `np.log` warned "divide by zero" in every
  `gate_scatter_bead_sample` call; the -inf was clipped to `MIN_LOGDENSITY` right after.
- Now: `np.errstate(divide="ignore")` around the log; values unchanged.
- Affected when: every scatter bead gate with the binned KDE (default).
- Effect on results: none (warning noise only).
- Affected analyses: all bead gates (e.g. hd7-01) lose the RuntimeWarning; nothing else.
- Action: nothing.
- Test: jobs/r2b_verify/v2_all.py G2-13; f2_A.py "G2-13".

### R2b-G2-15 spectrum PCA: zero-variance last component
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-15.
- Where: spill.pca.SingleSpectrumPCAFitter.__init__ (spill/pca.py:24); spill.panel.PanelConfiguration.collect_PCA_data
  (spill/panel.py:101).
- What was wrong: N was capped at min(N, n_spectra, n_channels), but n mean-centred spectra span only n-1 dimensions. With
  n_spectra <= N the last component had (numerically) zero variance, and after the std norming in `collect_PCA_data` its
  embedding column was NaN or above 1e6 in magnitude.
- Now: N = min(N, n_spectra-1, n_channels) (1 spectrum: the dummy fit with N=0, as before). `collect_PCA_data` raises
  ValueError naming the components a panel requests that were not fitted (before: a bare KeyError from pandas).
- Affected when: dyes with n_spectra <= N (default N=5, `SpilloverSingleStain.add_PCA_fit`).
- Effect on results: for those dyes one PCA component fewer: `pca_embedding` / `pca_components` /
  `pca_explained_variance` lose the degenerate column. The remaining components are unchanged. A panel that used the
  removed (last) component now raises ValueError; it was degenerate. Dyes with more than N spectra are unchanged. vs repro
  copy: same (it caps at min(N, n_spectra)).
- Affected analyses: none found. figures/plots_dye/lib/dye.py:78/522/707-708 and pipeline hd7-02 (cells 64, 79, 89):
  all fluorophores in the 23 runs have more than 5 runs (lead check, VERIFY.md), so N=5 < n_spectra-1 is unchanged.
- Action: nothing.
- Test: jobs/r2b_verify/v2_all.py G2-15; f2_A.py "G2-15"; jobs/r2_verify/f_spill.py "C14" (expectation updated: 3
  spectra -> N=2).

### R2b-G2-17 `readfcs_sample`: missing meta channel gave a bare KeyError
- Commit: R2b-A (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-17.
- Where: io.utils.readfcs_sample (io/utils.py:228).
- What was wrong: a `channels_meta` channel (e.g. `Time`) missing from the file raised a pandas KeyError, while missing
  fluorescence channels give a ChannelMismatchException with the available channels.
- Now: ChannelMismatchException "Could not find meta channels [...] in file ...! Available channels are: [...]".
  Docstring lists the raised cases.
- Affected when: .fcs files lacking a configured meta channel (also via `readfcs_rainbow_sample`).
- Effect on results: none (error type/message only).
- Affected analyses: none (A3 files have Time).
- Action: nothing.
- Test: jobs/r2b_verify/v2_all.py G2-17; f2_A.py "G2-17".

### R2b group B (ref, torch)
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Checks: jobs/r2b_verify/v2_all.py (NOT REPRODUCED), f2_B.py ALL PASS; jobs/r2_verify/f_ref.py, f_core.py ALL PASS. G2-06 equivalence (new max_anchor_dist k+1 == old k) checked against jobs/r2b_verify/old_ref_data.py.

### R2b-G2-03 `SampleOverlapDataset(keys_sample_covariate=...)` KeyError
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-03.
- Where: ref.data.SampleOverlapDataset.setup (ref/data.py:251, :279).
- What was wrong: the covariate keys were read from `self.meta`, but not added to the metadata keys loaded from
  `ad.uns`, so any covariate that was not also a batch, id or uid key raised KeyError.
- Now: `keys_sample_covariate` are part of the loaded metadata keys.
- Affected when: covariate keys that are not batch/`key_id`/`key_uid` keys.
- Effect on results: none (crash -> works).
- Affected analyses: none. hd7-03 (pipeline) passes `keys_sample_covariate=["ovlp_uid"]`, which is also its `key_id`,
  so it worked before and is unchanged.
- Action: nothing.
- Test: v2_all.py G2-03; f2_B.py G2-03.

### R2b-G2-04 nested imputation with an unknown major batch
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-04.
- Where: ref.moe.MultiOrdinalEncoder.transform_df (ref/moe.py:147-152).
- What was wrong: a row with an unknown minor batch whose major batch has no known minor batch had all distances
  `inf`; `argmin` then returned 0, so the row silently got the first reference label (only the `_imputed` flag was set).
- Now: such rows raise ValueError naming the major values and keys. Rows whose major batch is known are imputed as
  before.
- Affected when: only when the major key is not itself in `keys_batch` (otherwise its own encoder already raised).
- Effect on results: none for valid inputs; such calls now raise instead of returning an arbitrary label.
- Affected analyses: none (hd7-03/05 have the major key `batch` in `keys_batch`; reproducibility
  `figures/plots_batchintHD7/lib/parameters.py:43` filters to known labels first).
- Action: nothing.
- Test: v2_all.py G2-04 (adapted: ValueError naming the value = fixed); f2_B.py G2-04.

### R2b-G2-06 `ConnectionSampler.max_anchor_dist` off by one
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-06 (user decision "Fix meaning,
  pass 2").
- Where: ref.data.ConnectionSampler.__init__ (ref/data.py:114), ref.data.SampleOverlapDataset.setup (:286, hard-coded
  `max_anchor_dist=1`).
- What was wrong: the growth loop ran `range(1, max_anchor_dist+1)`, so `max_anchor_dist=1` also trained batches at
  anchor distance 2, against the docstring ("Set to one to only sample connections with anchors"). Batches further away
  got no connections and no warning.
- Now: the loop runs `range(1, max_anchor_dist)`: batches up to `max_anchor_dist` connections from an anchor are
  trained, 1 = only connections with anchor samples; `max_anchor_dist < 1` raises. `SampleOverlapDataset` has a new
  parameter `max_anchor_dist`, default 2, which reproduces the old hard-coded behaviour. A warning lists batches that
  are connected to an anchor but beyond the limit (not trained; `find_nontraining_batches`). Docstrings describe the
  growth rule.
- Affected when: direct `ConnectionSampler(..., max_anchor_dist=k)` with anchors: now trains distance <= k instead of
  k+1. Through `SampleOverlapDataset` (default 2): unchanged.
- Effect on results: none via `SampleOverlapDataset` (verified: identical connection matrices and samples for old
  `k` vs new `k+1`, k = 1..3, on a chain and 40 random graphs, both `train_higher` values; and old hard-coded 1 vs new
  default 2 through `SampleOverlapDataset`). New warning only.
- Affected analyses: none. No direct `ConnectionSampler` use in reproducibility or the pipeline. hd7-03 (anchor batch
  18-APR-2024, Ref0 in every batch, all other batches at distance 1) and hd7-05 (no anchors) are unaffected by the
  meaning change; no new warning expected for hd7-03 (not checked on the data).
- Action: nothing.
- Test: v2_all.py G2-06; f2_B.py G2-06 (three checks, against jobs/r2b_verify/old_ref_data.py).

### R2b-G2-07 MMD bandwidth on the loss stains
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-07 (user decision "Use loss
  stains").
- Where: ref.model.OverlapFitModel.estimate_MMD_bandwidth (ref/model.py:197-213).
- What was wrong: the bandwidth was estimated on all transformed stains, while the loss (`get_DDIST_from_dataset`)
  only uses `transformer.loss_use_stain_mask`. In the lead check: 6.555 (joint) / 1.573 (per marker) instead of 2.227.
- Now: estimated on `transformed[:, loss_use_stain_mask]`, in the joint and the `DDIST_separate` branch.
- Affected when: `DDIST_mode="MMD"` with `ScalingTransformer` and `scale_stains` not all True. `SpectralFitTransformer`
  (mask all True) and sinkhorn/hausdorff are unchanged (verified: all-True mask gives the old values).
- Effect on results: changes the MMD bandwidth and therefore the fitted stain factors of such fits (newly fitted
  models; saved models keep their stored `bandwidth_MMD`).
- Affected analyses: none found. hd7-05 uses `ScalingTransformer` with sinkhorn; hd7-03 uses MMD with
  `SpectralFitTransformer` (mask all True).
- Action: nothing (re-fit only MMD + ScalingTransformer-subset models, none known).
- Test: v2_all.py G2-07; f2_B.py G2-07.

### R2b-G2-11 `fit_factor_Sinkhorn_LBFGS` removed
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-11 (user, verbatim: "Add a
  comment somewhere that this didn't work well, then remove").
- Where: ref.factor.fit_factor_Sinkhorn_LBFGS (ref/factor.py:21-114, removed).
- What was wrong: L-BFGS (lr=1, no line search) on a stochastic closure (the aggregate subsample was redrawn at every
  closure call) diverged in 2/8 seeds to absurd factors without an error, and crashed in some runs; lr=0.1 + strong
  Wolfe still crashed 1/4. A fixed subsample per step looked stable in a small test (0/6 failures).
- Now: removed. `fit_factor_Sinkhorn` (SGD) covers the use. A note in the `fit_factor_Sinkhorn` docstring that an
  L-BFGS variant was tried and diverged was added here, and removed again in the author's review of ref/factor.py
  (R3, 2026-11-07); this entry is the record. (Corrected 2026-11-07 in R4b; the entry first said the note exists.)
- Affected when: code that imports `fit_factor_Sinkhorn_LBFGS` from `cytovanni.ref.factor` (it was not exported in
  `ref/__init__`): ImportError.
- Effect on results: none (no callers).
- Affected analyses: none found (reproducibility and pipeline use `fit_factor_Sinkhorn`: refjohanna/lib/refjoh.py:137,
  plots_cytonorm/lib/mohref.py:114, plots_dye/00_prepare_scaling.ipynb). R1-D2 mentions the removed function.
- Action: nothing.
- Test: v2_all.py G2-11 (adapted: missing function = fixed); f2_B.py G2-11.

### R2b-G2-12 empty batch in the Sinkhorn factor fits
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-12.
- Where: ref.factor.fit_factor_Sinkhorn (ref/factor.py:116), fit_factor_Sinkhorn_grouped (:204), fit_factors_Sinkhorn
  (:305).
- What was wrong: an empty batch reached the sampler and gave an IndexError.
- Now: input validation (`_check_nonempty`) raises ValueError naming the empty batches (for `_grouped`: the empty
  (batch, sample) entries), after the `dummy` shortcut.
- Affected when: a batch (or batch/sample entry) with no events.
- Effect on results: none (error message only).
- Affected analyses: none.
- Action: nothing.
- Test: v2_all.py G2-12; f2_B.py G2-12.

### R2b-G2-14 non-positive or non-finite event weight sums
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-14.
- Where: torch.geomloss.GeomSampleLoss._normalised_weights (torch/geomloss.py:59-64); torch.mmd.MMDLoss.forward
  (torch/mmd.py:70-71). New helper torch.utils._check_weight_sums.
- What was wrong: weights summing to 0 (or containing NaN/inf) were normalised by their sum, giving a silent NaN loss
  (sinkhorn/hausdorff and MMD).
- Now: ValueError if any weight sum (per batch row for batched geomloss input) is not positive and finite.
- Affected when: custom event weights with such sums. `RareMarkerWeights` is clamped to >= 1, so model fits are not
  affected.
- Effect on results: none for valid weights (verified identical to the explicit formulas); invalid weights raise.
  Note: the check reads the sum on the host (one device sync per weighted loss call on GPU).
- Affected analyses: none.
- Action: nothing.
- Test: v2_all.py G2-14 (adapted: ValueError in both losses = fixed); f2_B.py G2-14.

### R2b-G2-16 `get_df_sample_connections` without anchors
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-16.
- Where: ref.data.ConnectionSampler.__init__ no-anchor branch (ref/data.py:96); get_df_sample_connections (:170).
- What was wrong: `idx_anchor_distance` was only set with anchors, so `get_df_sample_connections` raised
  AttributeError without them.
- Now: without anchors `idx_anchor_distance` is a NaN array; the DataFrame has NaN anchor distances. Docstring added.
- Affected when: `is_anchor=None` + `get_df_sample_connections`.
- Effect on results: none (crash -> works).
- Affected analyses: none (hd7-05 uses no anchors but does not call it).
- Action: nothing.
- Test: v2_all.py G2-16; f2_B.py G2-16.

### R2b-G2-18 `SpectralFitTransformer` with a `FixedPanelModule`
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-18.
- Where: ref.model_transformer.SpectralFitTransformer.get_standardisationparams_fromad (ref/model_transformer.py:561,
  :567), used by add_standardisation_parameters.
- What was wrong: the code assumed batched spectra (batch, stain, channel) and an embedding from the panel module;
  `FixedPanelModule` returns one (stain, channel) matrix and an empty 1-D embedding, so both methods raised.
- Now: branch on `pem.TRAINED_EMBEDDING`: fixed panels use the full spectra matrix and an empty embedding
  (`np.zeros(0)`, also stored as `uns["panel_embedding"]`). The trained-embedding path is unchanged. Docstrings note the
  empty embedding.
- Affected when: `SpectralFitTransformer` with `FixedPanelModule`.
- Effect on results: none (crash -> works; verified spectra, unmixing matrix and varm for 2x3 and 3x3 panels).
- Affected analyses: none (FixedPanelModule is only used inside ScalingTransformer there).
- Action: nothing.
- Test: v2_all.py G2-18; f2_B.py G2-18; f_ref.py B2+A3 (trained path, unchanged).

### R2b-G2-19 `SampleOverlapDataset(key_batch_sample=None)`
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-19.
- Where: ref.data.SampleOverlapDataset.setup (ref/data.py:251, :283-286).
- What was wrong: `None` was added to the metadata keys (KeyError None); bypassing that, the sample id was passed as
  the batch index with `block_within_batch=True`, which blocks every same-id connection ("No connections").
- Now: `None` is left out of the metadata keys; with `None`, `block_within_batch=False` and every sample is its own
  "batch" (`idx_batch = arange(Nsample)`), so all same-id connections are used and, with anchors, anchor distances
  count connections between samples. Docstring updated.
- Affected when: `key_batch_sample=None`.
- Effect on results: none (crash -> works).
- Affected analyses: none (all callers pass "batch" or "batchmix").
- Action: nothing.
- Test: v2_all.py G2-19; f2_B.py G2-19.

### R2b-G2-20 unmixing error and resistance for a zero Jacobian
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-20 (user decision "Error 0, R 0").
- Where: ref.panel.PanelModule.get_unmixing_resistance (ref/panel.py:230-233, :257); also via
  get_unmixing_resistance_fromad.
- What was wrong: with `single_step=False` the gradient direction was 0/0 for a stain whose Jacobian row is zero, so
  the error was NaN; in both modes R = arcsinh(|val/err|/20) was NaN where val == err == 0 (e.g. all-zero events).
- Now: zero Jacobian -> zero direction -> error 0. R = 0 where val == 0. Where val != 0 and the error is 0, R = inf
  (as computed before). Docstring documents the outputs and conventions.
- Affected when: events/stains with an exactly zero Jacobian row (all-zero intensities in the panel channels).
  Events with a nonzero Jacobian are bit-identical (verified against the pre-fix formula, both norms).
- Effect on results: `val_std` (error) NaN -> 0 and R NaN -> 0 for those events; a stain with zero Jacobian but
  nonzero value: error NaN -> 0, R NaN -> inf.
- Affected analyses: pipeline hd7-03 calls `get_unmixing_resistance_fromad(..., single_step=False,
  gradstep_maxnorm=False)` (plots) and `get_unmixing_resistance(..., single_step=False)` storing `val_std` in obsm.
  Only all-zero events change (NaN -> 0); with measurement noise these should be absent (not checked on the data).
- Action: nothing unless hd7-03 outputs contain NaN in R/val_std; then re-run those cells.
- Test: v2_all.py G2-20; f2_B.py G2-20 (two checks).

### R2b-G2-21 MMD automatic bandwidth floor
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-21.
- Where: torch.mmd.RBF.get_bandwidth and RBF.estimate_bandwidth (torch/mmd.py:28-41), so also
  MMDLoss.estimate_bandwidth and OverlapFitModel.estimate_MMD_bandwidth. New helper torch.mmd._bandwidth_floor.
- What was wrong: on identical points (all distances 0) the automatic bandwidth was 0 and the kernel 0/0 = NaN.
- Now: automatic and estimated bandwidths are clamped to sqrt(smallest normal number) of their dtype, at least that of
  float32 (~1e-19; the kernel multiplies with float32 multipliers, and estimated bandwidths are reused as fixed ones).
  Fixed bandwidths passed by the user are not changed.
- Affected when: `MMDLoss()` without a fixed bandwidth, or `estimate_bandwidth`, on all-identical points.
- Effect on results: none for ordinary inputs (verified identical bandwidth and loss); NaN -> 0 for identical points.
- Affected analyses: none (OverlapFitModel passes a fixed bandwidth; `figures/plots_distmet/lib/distmet.py:40` uses
  automatic bandwidth on non-degenerate data).
- Action: nothing.
- Test: v2_all.py G2-21; f2_B.py G2-21.

### R2b-G2-22 `get_spectra_fromembedding(include_unused=True)`
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-22.
- Where: ref.panel.PanelModule.get_spectra_fromembedding (ref/panel.py:367).
- What was wrong: `include_unused` was ignored; the spectra were always computed with unused components masked.
- Now: `mask_unused=not include_unused` is passed on.
- Affected when: `include_unused=True`.
- Effect on results: those callers now get the unmasked spectra (default `False` unchanged).
- Affected analyses: none (`figures/plots_dye/lib/dye.py:859` uses the default).
- Action: nothing.
- Test: v2_all.py G2-22; f2_B.py G2-22.

### R2b-G2-23 `apply_component_mask` broadcasting
- Commit: R2b-B (review/R2b-bugs), 2026-11-07. Approval: todo/R2b_approval.md item G2-23.
- Where: ref.panel.PanelModule.apply_component_mask (ref/panel.py:96).
- What was wrong: the mask was viewed to the full input shape, which fails for (components, k) with k > 1.
- Now: reshaped to `[n_components] + [1]*(ndim-1)` and broadcast.
- Affected when: inputs with more than one entry per component.
- Effect on results: none (crash -> works; (components,) and (components, 1) unchanged).
- Affected analyses: none (internal `get_L1` uses (components, 1)).
- Action: nothing.
- Test: v2_all.py G2-23; f2_B.py G2-23.

## R2c (items from the reproducibility review, branch review/R2c-evl; approvals in todo/R2c_approval.md)

### R2c-H1 positive CVs NaN for a single value <= 0
- Commit: R2c-evl (review/R2c-evl), 2026-11-07. Approval: todo/R2c_approval.md item H1 (option a).
- Where: utils.evl.get_positive_paired_CV, get_positive_paired_CV_alongdiag, get_positive_single_CV.
- What was wrong: only non-finite values were masked, before `np.log`; a 0 became -inf and made the whole CV NaN.
- Now: values <= 0 are dropped with the non-finite ones (pairs: if either value is <= 0); fewer than 2 remaining values
  give NaN with a CytovanniWarning.
- Affected when: inputs containing values <= 0 (shares of 0 for rare cell types).
- Effect on results: those CVs change from NaN to the CV over the positive values; all-positive inputs are identical.
- Affected analyses (cytovanni_reproducibility): figures/plots_liechti/lib/liechti.py:218, 298, 299, 309
  (intermediate/liechti_share_metrics.csv: 2 cell types were NaN, ICS:P784:F gd T IL-17+ and TNK:P10:F NKT
  CD4+CCR5-; no visible effect in the figures); figures/plots_batchintHD7/lib/cv.py:222, 224, 260 and
  composition.py:46, 47, 57 (share CVs; only cell types with a share of 0 in some sample);
  processing/batchint_data/lib/validation.py:556 (its own `np.isfinite(x*y).sum() >= 2` guard does not catch zeros).
- Action: re-run the share-CV tables if NaN entries mattered; otherwise nothing.
- Test: jobs/r2c_check/v_evl.py "H1".

### R2c-H2 `get_positive_single_CV` used the population variance
- Commit: R2c-evl, 2026-11-07. Approval: todo/R2c_approval.md item H2 (option a).
- Where: utils.evl.get_positive_single_CV.
- What was wrong: `x.var()` (ddof=0), while the asinh single CVs use ddof=1 (and the paired ones `np.cov`, ddof=1).
- Now: `x.var(ddof=1)`.
- Affected when: every call.
- Effect on results: the log-variance grows by n/(n-1), the CV by about sqrt(n/(n-1)) (n=10: +5%, n=50: +1%).
- Affected analyses: figures/plots_liechti/lib/liechti.py:218, 309; figures/plots_batchintHD7/lib/cv.py:260;
  composition.py:57 (reference-sample share CVs).
- Action: re-run those tables/figures for the final version (small change).
- Test: jobs/r2c_check/v_evl.py "H2".

### R2c-H3 CV functions: warnings for NaN, paired varcorr clamp; linearisation and cofactor documented
- Commit: R2c-evl, 2026-11-07. Approval: todo/R2c_approval.md items H3, H4.
- Where: all CV functions in utils/evl.py.
- What was wrong: NaN results (along-diagonal variance < 0; fewer than 2 usable values) came without a warning;
  `get_asinh_paired_CV_varcorr` clamped a negative corrected variance to 0 silently (the single variant warns, R2-E16).
- Now: every NaN result comes with a CytovanniWarning naming the reason; the paired varcorr clamp warns. Docstrings
  state the global linearisation (factor in [1, sqrt 2]) and that below the cofactor the value is SD/f by design (H4).
- Affected when: those cases only.
- Effect on results: none (values identical, verified on clean inputs for all asinh functions and the positive paired ones).
- Affected analyses: none (warnings only).
- Action: nothing.
- Test: jobs/r2c_check/v_evl.py "H3".

### R2c-H5 mean norm of `ExtendableParameter` documented
- Commit: R2c-evl, 2026-11-07. Approval: todo/R2c_approval.md item H5 (document only).
- Where: torch.parameter.ExtendableParameter docstring (`axis_norm`).
- Now: states that entries without training data enter the mean, so the absolute zero point depends on them.
- Effect on results: none (docstring). HD7 models checked: no such entries (jobs/r2c_check/h5_*.py).

## R2d (codex pass at full settings, branch review/R2d-bugs; approvals in todo/R2d_approval.md)
Findings and reproducers: todo/R2d_findings/codex.md, jobs/r2d_codex/ (review project). Lead checks: jobs/r2d_verify/.
Caller check for all items: cytovanni_reproducibility figures/ + processing/ and the HD7 pipeline notebooks.

### R2d-D1 `fit_factors_Sinkhorn` column alignment
- Commit: R2d (review/R2d-bugs), 2026-11-07. Approval: todo/R2d_approval.md item D1 (codex3-01).
- Where: ref.factor.fit_factors_Sinkhorn.
- What was wrong: marker names came from the first frame, but every frame was converted positionally (`to_numpy`), so
  frames with the same markers in another order were compared marker against wrong marker (identical data: factors
  [0.05, 19.6] instead of [1, 1]); frames with other marker names were accepted silently.
- Now: all frames must have the same, unique marker set (ValueError otherwise) and are aligned to the first frame's order.
- Affected when: frames with differently ordered columns.
- Effect on results: those fits change (to the correct factors); aligned inputs identical.
- Affected analyses: none (no `fit_factors_Sinkhorn` call in reproducibility or the HD7 pipeline).
- Action: nothing. Test: jobs/r2d_verify/v_r2d.py "D1"; repro_core.py factor_column_order.

### R2d-D2 transformer `extend` kept the stain/channel scaling masks
- Commit: R2d, 2026-11-07. Approval: D2 (codex3-02).
- Where: ref.model_transformer.ScaleFactor (`__init__`, `extend`).
- What was wrong: `scale_stains`/`scale_channels` (use_channel_mask) were applied to the trainable mask only at
  construction; `extend` added all-trainable rows, so disabled stains/channels were fitted for new batches.
- Now: `ScaleFactor` keeps the excluded feature indices and sets them untrainable in the new rows after `extend`.
- Affected when: `SpectralFitTransformer`/`ScalingTransformer` with masks not all True, extended and then fitted.
- Effect on results: the new batches' factors for excluded features stay 1 (before: fitted). Old batches unchanged.
- Affected analyses: none (no transformer `extend` call found; hd7-03 uses the masks but does not extend).
- Action: nothing. Test: v_r2d.py "D2"; repro_core.py extension_scale_masks; repro_followup.py extension_actual_fit.

### R2d-D3 `dataset_test` channel check
- Commit: R2d, 2026-11-07. Approval: D3 (codex3-03).
- Where: ref.model.OverlapFitModel `__init__`, `get_DDIST_from_dataset`; new `_check_dataset_channels`.
- What was wrong: only `dataset` was checked against the transformer channels; a test dataset in another channel order
  was transformed positionally (test loss 0.009 instead of 1.62 in the reproducer).
- Now: both datasets are checked at construction, and every dataset passed to `get_DDIST_from_dataset`.
- Effect on results: none for valid inputs; mismatched ones raise ValueError.
- Affected analyses: none (no `dataset_test` use found). Action: nothing. Test: repro_core.py test_dataset_order.

### R2d-D4 PCA null components and NaN through unused components
- Commit: R2d, 2026-11-07. Approval: D4 option (a) (codex3-04; incomplete R2b-G2-15 / R2-C14).
- Where: spill.pca.SingleSpectrumPCAFitter (`__init__`, new `get_numerical_rank`); ref.panel.PanelModule.get_spectra
  (mask_unused); spill.panel.PanelConfiguration.get_spectra_from_embedding.
- What was wrong: normalised spectra (one channel fixed to 1) have rank <= n_channels-1, so with few channels a fitted PC
  could have zero variance -> std 0 -> NaN embedding column; NaN * 0 kept that NaN in reconstructions even for unused PCs.
- Now: N is also capped at the numerical rank of the centred spectra (s**2 > 1e-10 * s_max**2; rank 0 -> dummy fit);
  masked reconstructions zero the unused components' embedding first (pandas and torch).
- Affected when: stains whose spectra have rank < requested N (few channels / dependent channels); reconstructions with a
  non-finite unused embedding.
- Effect on results: NaN -> finite where it occurred; such stains get fewer components. Full-rank fits identical.
- Affected analyses: none. The saved HD7 panels (panel_HD7_{Identity,TCell}{,_refined}.pt, 22 stains x 5 PCs) have a
  smallest PC variance of 3.4e-4 x the largest per stain (BUV563), far above the 1e-10 cutoff: the cap changes nothing
  (jobs/r2d_verify/hd7_panels.py). Their stored embeddings contain NaN entries (not investigated; most likely runs
  without that dye, not null PCs).
- Action: nothing. Test: v_r2d.py "D4"; repro_numerics.py pca_rank_deficiency.

### R2d-D5 variance-corrected CVs accept error 0
- Commit: R2d, 2026-11-07. Approval: D5 option (a) (codex3-05).
- Where: utils.evl.get_asinh_paired_CV_varcorr, get_asinh_single_CV_varcorr.
- What was wrong: the usable-value mask required error > 0, so zero errors (e.g. R2b-G2-20's zero-Jacobian events) were
  dropped; all-zero errors gave NaN instead of the uncorrected CV.
- Now: finite errors >= 0 are used; negative or non-finite ones are dropped.
- Effect on results: CVs with zero errors change (NaN -> value; mixed zeros: those events now count).
- Affected analyses: none (only an unused alias, figures/plots_batchintHD7/lib/cv.py:35).
- Action: nothing. Test: v_r2d.py "D5"; repro_numerics.py varcorr_zero_errors.

### R2d-D6 PMT fit with a fixed laser exponent
- Commit: R2d, 2026-11-07. Approval: D6 (codex3-06).
- Where: rainbow.pmt.PMTExponentFitter.fit_single.
- What was wrong: with `fit_laser_exponent=False` curve_fit still got 3 parameters (the laser one unused), so 2 voltages
  raised TypeError.
- Now: a 2-parameter fit (intercept, PMT exponent) in that case; exponent 1 appended.
- Effect on results: 2 voltages: crash -> works; >= 3 voltages: same least-squares solution (verified against polyfit).
- Affected analyses: none (figures/plots_calib/lib/calib.py:1238 fits the laser exponent). Action: nothing.
- Test: v_r2d.py "D6"; repro_core.py two_voltage_fixed_laser.

### R2d-D7 `UnmxErrorExclusionMultiplier` in float64
- Commit: R2d, 2026-11-07. Approval: D7 (codex3-07).
- Where: ref.model_transformer_smp.UnmxErrorExclusionMultiplier.forward.
- What was wrong: the boolean mask was cast to float32 before multiplying with `block_mask`, which is float64 after
  `.double()` -> RuntimeError.
- Now: cast to `block_mask.dtype`.
- Effect on results: none in float32; float64 works.
- Affected analyses: cytovanni_reproducibility processing/batchint_data/lib/validation.py:186-190 patches `block_mask`
  back to float32 by hand; that workaround is no longer needed (still works). Action: optional cleanup (R7).
- Test: repro_followup.py exclusion_multiplier_float64.

### R2d-D8 all-fixed PCA panel embedding
- Commit: R2d, 2026-11-07. Approval: D8 (codex3-08).
- Where: spill.panel.PanelConfiguration.get_spectra_embedding.
- What was wrong: `pd.concat([])` when no stain has components -> ValueError.
- Now: an empty float Series. Effect on results: none (crash -> works). Affected analyses: none.
- Test: repro_numerics.py dummy_pca_spectrum_embedding.

## R3b (bugs found while writing the R3 docstrings, branch review/R3b-bugs; approvals in todo/R3_bugs_approval.md)

All approved by the user 2026-11-07 ("B7, extend it so it functions as the identity. The other Bs sound fine." /
"U3 a"; B11 from the user's answer "fsc ssc is blue, the (violet) ones are violet"). Tests: jobs/r3b_verify/v_r3b.py
in the review project (11/11 PASS, GPU job; B9 not run, code only).

### R3b-B1 `fix_sample_bug` keeps the $PnS labels
- Where: io.utils.fix_sample_bug. What was wrong: written with `overwrite_pns=False`, so the repaired file had no $PnS.
- Now: every channel gets its original $PnS ("" where none). Effect on results: none for `readfcs_sample` (uses
  $PnN); `readfcs_dataframe` column names of repaired files are the stain labels again.
- Affected analyses: files repaired with `fix_A3_HW_bug` before this fix lack $PnS (re-run the repair if needed).

### R3b-B2 `read_folder_timestamps` order without UTC
- Where: io.utils.read_folder_timestamps. What was wrong: if any file had no UTC, the whole table was sorted by the
  $DATE string ("01-FEB-2024" before "05-JAN-2023").
- Now: sorted by UTC; files without UTC last (by $DATE/$BTIM strings). Effect: row order only. Affected: none known.

### R3b-B3 `CovarianceTrilExtendableParameter` on the GPU
- Where: torch.parameter.CovarianceTrilExtendableParameter. Now the init helpers create tensors on `device`.
- Effect: crash -> works. Affected: none (class not used by default).

### R3b-B4 `plot_eval_NxN` returns the datashader image
- Where: ref.plot.plot_eval_NxN. Now returns `base_plot_NxN_ds(...)` (the image, None with savepath). Plot only.

### R3b-B5 `plot_GMM_llh` draws the cutoffs on `ax`
- Where: rainbow.plot.plot_GMM_llh (`ax.axvline` instead of `plt.axvline`). Plot only.
- Affected: cytovanni_reproducibility processing/gosling/lib/gosling.py:295 `plt.sca` workaround no longer needed (R7).

### R3b-B6 voltage setting with a saturated channel
- Where: rainbow.setvolt.get_config_from_rainbow_reference, get_voltages_from_rainbow.
- What was wrong: if no peak of a channel was below the linear limit, its target became 0 (peak 1) silently; the
  voltage step then gave voltage 0 and error inf.
- Now: CytovanniWarning naming the channels, target NaN; 'target' and 'voltage_target' are nullable integers (Int64)
  instead of int, so that channel gets NaN factor/error/voltage. Effect: only for such channels; dtype change of
  the two columns (values unchanged otherwise). Affected: none known.

### R3b-B7 `DummyTransformer` as working identity
- Where: ref.model_transformer.DummyTransformer. Now: `from_data(channels, stains=None)` (stains default to the
  channels, one per channel, else ValueError), `stain_marker_name`, `loss_use_stain_mask`, `export(include_history)`,
  `standardise(..., evl)`. Works with `OverlapFitModel` for distances, `add_ad_standardised` (not `name_by='marker'`),
  save/load and extend; nothing to fit. Breaking: `from_data` no longer accepts different stain and channel counts.
- Effect: crash -> works. Affected: none (unused).

### R3b-B8 `SpilloverCollection` warns about a stain under several dye names
- Where: spill.collection.SpilloverCollection.__init__. Collection unchanged (first dye name wins); a SpilloverWarning
  lists the measurements that are not included for that stain. Full support -> R5. Affected: none for HD7.

### R3b-B9 `LinearSingleStainSpectrumFit.plot_NxN` with a channel subset
- Where: spill.stainfit.LinearSingleStainSpectrumFit.plot_NxN: data and fitted lines are looked up by channel name.
  Plot only.

### R3b-B10 subsampling by position
- Where: utils.misc.anndata_concat(permute=True), anndata_subsample, df_subsample.
- What was wrong: events/rows selected by name: non-unique names raised (AnnData) or returned extra rows (DataFrame).
- Now: by position. Effect: identical draws for unique names (same numpy random calls); works for non-unique.
- Affected: none (the author's calls use make_unique=True).

### R3b-B11 S8 scatter channels' laser
- Where: utils.config.CytometerConfiguration_S8 `laserdict`: FSC and SSC (Imaging) -> Blue, LightLoss (Violet) ->
  Violet; the separate "Imaging" entry is gone; LightLoss (Imaging) stays unassigned ('unknown', to confirm).
- Effect: `channel_laser_key` (plot colours, per-laser grouping). The S8 calibration figure (figures/plots_calib
  `s8_prepare`/`plot_s8`) plots only channels with a wavelength, so the scatter channels are not in it.

### R3b-U3 panel `*_fromad` diagnostics apply the fitted factors
- Where: ref.panel.PanelModule.get_unmixing_resistance_fromad, get_grouped_jac_fromad, get_grouped_jac_single_fromad,
  get_clustermeans_withz (new helper `_get_x_stainscale_fromad`; get_compmean_errorbars gains `stainscale_np`).
- What was wrong: they unmixed `layers[key_layer]` without the channel factors (`var['sfm_calibration_factor']`) and,
  except get_grouped_jac_single_fromad, without the stain factors (`varm['sfm_stain_factor']`); the author's own
  add_unmixing_stability (hd7-03 cell 317) applies both.
- Now: `apply_factors=True` (default) applies both where present in the AnnData; `apply_factors=False` = old behaviour.
  Verified equal to the manual computation (R, Jacobian, cluster means).
- Effect on results: yes, these diagnostics on AnnData objects that carry the standardisation parameters.
- Affected analyses: hd7-03-spectra-integration.ipynb cells 94, 96 (R, grouped Jacobian) and 120-257
  (`get_grouped_jac_single_fromad`): values change if re-run (diagnostic plots only, no downstream results).

### R3b-U2 unmixing-resistance defaults
- Where: ref.panel.PanelModule.get_unmixing_resistance (was single_step=True, gradstep_maxnorm=True) and
  get_unmixing_resistance_fromad (was gradstep_maxnorm=True): both now single_step=False, gradstep_maxnorm=False,
  the settings of every call in the usage evidence (hd7-03 cells 5, 96, 317). `include_unused` stays False.
- Approval: user 2026-11-07, "Yes, change the defaults".
- Effect on results: yes, for calls that rely on the defaults (different step direction/normalisation of R).
- Affected analyses: none known (all known calls pass both arguments explicitly).

---

## R4b (bugs found by the R4 tests, branch review/R4b-bugs; approval: user 2026-11-07, "Fix on branch, correct the foxlog")

Every fix has the reproducer that found it (previously a strict xfail) as its test; findings in todo/R4_findings/.

### R4-B1 `CovarianceTrilExtendableParameter` with fewer than two axes
- Where: torch.parameter. The documented ValueError was only raised in `get_initial_trainable`; the trainable mask
  (`torch.tril`) is built first and raised a RuntimeError. Now `get_initial_trainable_mask` checks the shape too.
- Effect: error type only. Affected: none. Test: test_torch_parameter.py::test_covtril_one_axis_raises.

### R4-B2 `RBF_batched` with a float bandwidth
- Where: torch.mmd.RBF_batched.get_bandwidth. A float (documented) was indexed like a (..., 1, 1) tensor (TypeError).
  Now a fixed bandwidth is converted to a tensor on the distances' device, a scalar to shape (1, 1).
- Effect: crash -> works; tensor bandwidths unchanged. Affected: none. Test: test_torch_mmd.py::test_rbf_batched_float_bandwidth.

### R4-B3 single-stain file names: channel that ends another channel's name
- Where: spill.utils.collect_bead_controls_regularflow and spill.autospill.make_AutoSpill_fcs_config (filename regex).
- What was wrong: `Compensation Controls_(.*){channel} Stained Control.fcs` also matched a file of a channel whose name
  ends with `channel`, e.g. the UV540 control for V540 (and UV1 for V1). Then V540 either got the UV540 file (if it has
  no own control) or the search failed with "multiple possible matches".
- Now: `Compensation Controls_(?:(.*) )?{channel} Stained Control\.fcs$`: the antigen (if any) is separated by a space,
  so the channel name must be a whole word. Antigens are read as before (stripped).
- Effect on results: yes where such channel pairs exist (S6, Aurora configs) and a channel had no own control (it now
  gets none / the pad file instead of the other channel's control), or the search failed (now works). Unchanged
  otherwise; files without a space between antigen and channel (not the BD convention) no longer match.
- Affected analyses: none known (the usage evidence is A3/S8/Fusion data). Action: on S6/Aurora single stains, check
  `panel_meta.csv` / the AutoSpill config of earlier runs for a V-channel that got a UV file.
- Tests: test_spill_utils.py::test_regularflow_channel_that_ends_another_channel_name,
  test_spill_autospill.py::test_channel_that_ends_another_channel_name_not_matched.

### R4-B4 `HistogramTesselator` with Nx != Ny
- Where: gating.autospill.HistogramTesselator. The index <-> data maps used `np.vstack` of the x and y bin centres,
  which fails for different lengths. Now they use the first and last bin centres (`bins_center_corners`; the
  `bins_center` attribute is gone). Effect: crash -> works; square grids give identical results.
- Affected: none (all calls use Nx == Ny). Test: test_gating_autospill.py::test_tesselator_nonsquare_grid.

### R4-B5 duplicate exception classes
- Where: exceptions.py; the second (identical) definitions of `SpilloverFitException` and `MissingGPUWarning`
  removed. Effect: none. Test: test_package.py::test_no_duplicate_class_definitions_in_exceptions.

### R4-B6 `EventRateEstimator` on quantised event times
- Where: utils.rate.EventRateEstimator (used by `estimate_event_rate`, i.e. by arges).
- What was wrong: event times are stored with a finite resolution ($TIMESTEP, e.g. 0.01 s). At rates of the order of
  1 / $TIMESTEP or above most consecutive differences are 0, so median = 0 and rate = inf; below that the median is
  rounded to a few time steps, so the rate was biased (e.g. 50 events/s at 0.01 s gave about 69).
- Now: if the median difference is below 20 resolution steps (smallest positive difference), the differences over k
  events (k = 2, 4, 8, ...) are used, with the Gamma(k) median: rate = gammaincinv(k, 1/2) / median(t[i+k] - t[i]).
  New attribute `k`. k = 1 is the old estimator (unchanged results when the median spans >= 20 steps).
- Check (numpy, 20000 events): 200/s at 0.01 s -> k 64, 199; 5000/s at 0.01 s -> 4875; 50/s at 0.01 s -> 50.5
  (k 16); unquantised and 1e-5 s resolution: unchanged (k 1).
- Effect on results: yes, rates of recordings with coarse time resolution relative to their event rate.
- Affected analyses: arges event rates (not in this package; logged in todo/ARGES_LUCY.md). Test:
  test_utils_oe_rate.py::test_event_rate_with_quantised_times.

### R4-B7 `check_pulse_width_independence` with a height of 0
- Where: utils.misc. One event with -H = 0 made A / H inf, the correlation NaN, and the result False ("computed").
  Now events with -H <= 0 are left out. Effect: NaN -> the correct answer; unchanged otherwise.
- Affected: none known. Test: test_utils_misc.py::test_check_pulse_width_independence_with_zero_height.

### R4-B8 `anndata_concat(permute=True)` warning
- Where: utils.misc.anndata_concat. Indexing for the permutation happened outside the warning filter, so the
  duplicate-name warning the docstring says is silenced appeared. Now inside. Effect: warning only.
- Test: test_utils_misc.py::test_anndata_concat_permute_silences_duplicate_name_warning.

### R4-B9 `get_voltages_from_rainbow` with a targetconfig read from .csv
- Where: rainbow.setvolt. '_N_peak' comes back as a float (3.0), which `GaussianMixture` rejects. Now `int()`.
- Effect: crash -> works. Affected: none (the pipeline passes the in-memory targetconfig). Test:
  test_rainbow_setvolt.py::test_voltages_from_csv_round_trip.

### R4-B10 `ExtendableParameter.extend` with mean normalisation: start of new entries
- Where: torch.parameter.ExtendableParameter.extend (default `keep_prior_trainable_structure=False`); shared by
  `ScaleFactor` (ScalingTransformer, SpectralFitTransformer scales), the rainbow fluorescence and scatter models, and
  the `fit_factor_*` log factors.
- What was wrong: the old entries' un-normalised values were moved into the fixed part. The mean (over the old
  entries) is subtracted from all entries, so new entries (init 0) started at -mean(old raw values), not at
  `init_value` (documented: "New batches start at factor 1"). The raw mean is 0 under plain SGD, but not under Adam
  (`OverlapFitModel.fit`, rainbow fits), so the start of new batches was off by the drift. Old entries were unchanged.
- Now: the old entries are stored normalised (`forward()`), so their masked mean is 0, they keep their values, and new
  entries start at `init_value`. `freeze()` (= `extend([])`) gives the same values as before.
- Effect on results: only the starting point of new (trainable) entries; after `extend` and a fit to convergence the
  same optimum. With `freeze` and no further fit, new entries now have factor 1 instead of exp(-mean). Exception:
  the exact scatter fit (`RainbowScatterCalibrationModule.fit`) takes the minimum-norm change from the start where the
  problem is degenerate, e.g. new batches measured only with a new bead type (common offset of the new position and the
  new batch shifts); there the split, and so the new batches' scatter factors, follows the new start.
- Affected analyses: hd7-01 rainbow integration (`extend_fit`): extended batches' factors may differ within the fit's
  convergence tolerance on a re-run. ScalingTransformer.extend: no calls found.
- Tests: test_ref_transformer.py::test_scaling_extend_new_batch_starts_at_factor_one,
  test_torch_parameter.py::test_extend_with_mean_norm_new_entries_start_at_init_value.

### R4-P4d rainbow fluorescence fit: covariance penalty (from the reproducibility session, P4c/P4d)
- Source: ~/hd7_myeloma/wuest/cytovanni_reproducibility/jobs/dev/p4d/cov_penalty.patch (written against ~/Cytovanni,
  old names RainbowIntegrator / ...IntegrationTrainer), documented in its todo/P4c_reassign_fix.md (P4d). Ported to
  `RainbowCalibrator.fit_fluorescence` / `RainbowFluorescenceGMMCalibrationTrainer`, numpydoc; code otherwise as in the
  patch. Applied on the user's request 2026-11-07 ("The fix from the other session, falls under p4c in its
  documentation, apply and come up with a regression test for it").
- What was wrong: the per-measurement peak precisions are free parameters. For calibration beads with few peaks per
  laser (Aria III clbr2pks), a peak broadened over its neighbours during the full fit and the measurement's factors
  then drifted (Green laser CV 21.5% instead of 6.5%); freezing the precisions after `fit_init_cov` stops the drift
  but breaks Fusion; a symmetric KL penalty has no lambda that works on both.
- Now: `fit_fluorescence(cov_penalty=0., cov_penalty_broaden_only=True)`, passed to the trainer: adds
  `cov_penalty * get_cov_penalty()` in `fit`, the KL divergence (zero-mean normals) between the covariances after
  `fit_init` (`set_cov_reference`, done at the first `fit` call) and the current ones, summed over peaks, averaged over
  measurements; with `cov_penalty_broaden_only` only broadening directions (eigenvalues of P Sigma0 < 1). The loss
  history includes the weighted penalty. `extend_fit` passes the options through `**fitkwargs`.
- Effect on results: none by default (0: same code path; test: identical factors and loss history). With the
  reproducibility setting (.3, broadening only): Aria III clbr2pks laser CVs 8.82/6.72/7.18/8.39 (orig
  9.01/6.54/7.87/9.35), Fusion/Aria III v2 at the P4c level (numbers from the reproducibility session, its jobs
  61613/61626/61652).
- Affected analyses: reproducibility processing/instrument_rainbow clbr2pks fits (already refitted there with the
  patched copy; identical to the package option per its test 61651). Action: switch that pipeline to the package option.
- Tests: tests/test_rainbow_covpenalty.py (penalty 0 at the reference, isotropic broadening = analytic KL for both
  variants, narrowing free with broadening only, gradient direction, option reaches the trainer, 0 = unpenalised fit;
  slow: penalised fit stays closer to the reference covariances, factors unchanged within 5% on synthetic beads).


## R5 (improvements)

### R5-A1 arcsinh axis scale; `label_axis_ArcSinh` replaced by `set_arcsinh_ticks`
- New: `ax.set_xscale("arcsinh", cofactor=...)` for data in its own units (`cytovanni.utils.scale`, registered on
  import). Major ticks 0 and +-10^p in log-axis notation (tick label font); powers near 0 left out as far as needed
  for the labels to fit (gap `label_gap` .5 em, next to the 0 `zero_gap` 1 em on x, 2 em on y); m x 10^p labels when
  fewer than two labels; minor ticks 2..9 x 10^p continued below the labels while >= 2.5 pt apart; the 0 label sits on
  the baseline of the power labels. Recomputed at every draw.
- `set_arcsinh_ticks(ax, cofactor, axis="x", minpower=None, mark_cofactor=False)` for already transformed data
  replaces `label_axis_ArcSinh(ax, cofactor, onx=True, minpower=2, mark_cofactor=False)` (removed).
- Affected: plots only (tick positions and labels; label text now `$\mathdefault{10^{3}}$`). No numerical results.
- Action: port calls of `label_axis_ArcSinh` (reproducibility code, ~50 calls): new name, `onx=False` -> `axis="y"`;
  `minpower` is now automatic by default (pass `minpower=2` for the earlier labels).

### R5-B7 rainbow fluorescence `fit`: `lr` of a later call with `warmup=0`
- Before: `RainbowFluorescenceGMMCalibrationTrainer.fit` set the learning rate only during the warm-up, so a later call
  with `warmup=0` kept the learning rate of the previous call and ignored `lr`.
- Now: the learning rate is set in every step (`lr * min(1, (step+1)/warmup)`, or `lr` without warm-up).
- Effect on results: none for a single `fit` call and for later calls with `warmup > 0` (identical learning rates).
  Changes only repeated calls with `warmup=0` and a different `lr`.
- Affected analyses: none known (all usages call `fit` once, via `fit_fluorescence` / `extend_fit`).
- Tests: tests/test_rainbow_covpenalty.py::test_refit_uses_the_given_lr.

### R5-B6 random states
- `get_config_from_rainbow_reference` and `get_voltages_from_rainbow` take `random_state` (passed to
  `fit_rainbow_GMM`); `PosNegSingleStainSpectrumFit(adata, random_state=None)` passes it to its mixture. Default
  None: as before (numpy's global random state).
- `get_hex_distinct` (N > 28) uses its own seeded generator instead of `np.random.seed(0)`: same colours, numpy's
  global random state is no longer reset.
- Everything else draws from numpy's / torch's global random state, so `np.random.seed` + `torch.manual_seed` before
  a call makes it repeatable (as done in the reproducibility code); no further arguments.
- Effect on results: none by default, except that random draws after a `get_hex_distinct(N > 28)` call no longer
  start from seed 0 (code that relied on this implicit reseed draws different subsamples there).
- Tests: test_rainbow_setvolt.py::test_random_state_gives_identical_results,
  test_spill_stainfit.py::test_posneg_random_state_reaches_the_mixture,
  test_utils_misc.py::test_get_hex_distinct_reproducible_and_leaves_global_state.

### R5-C API changes before the 1.2.0 pin (no numerical results change)
- C3: `cytovanni.rainbow.plot_eval_RBInt_fit(rbint, ...)` -> `plot_eval_calibration(calibrator, ...)`;
  `SpilloverCollection(rbint=...)`, `.from_custom(rbint, ...)`, `.from_AutoSpill(rbint, ...)` -> `calibrator`
  (attribute `.rbint` -> `.calibrator`); `SpectralFitTransformer.add_standardisation_parameters(rbint_key=...)` ->
  `calibration_factor_key`.
- C4: `SpilloverCollection.normalize_spectra(dye_channel_key=...)` -> `norm_channels` (Series stain -> channel).
- C5: `CutoffUnmxGate.apply` and `UnmxMinGate.apply` return a numpy bool array (were pandas Series, like the other
  gates now); `CutoffUnmxGate.apply` takes `addkey`; `PolygonGate` converts `vertices` to an array (a list works for
  `plot`).
- C2: `OverlapFitModel.export` and `.save` default to `include_history=False` (export was True). The loss per step
  (`losshistory['train']`, `['test']`) is now always saved; the per-step loss matrices and the transformer's parameter
  history only with `include_history=True`. The transformers' `export` default to False as well.
- C6: `PanelModule.get_grouped_jac_fromad(include_unused=True)` (was fixed True; default unchanged);
  `get_grouped_jac_single_fromad(s_dig=...)` -> `decimals`.
- C7: `ScalingTransformer.get_standardisationparams_fromad` returns 5 values like the SpectralFitTransformer one:
  `(None, stainscale, spectra, unmx_int_matrix, None)` (was 3).
- C11: `write/read_bead_h5ad` and `write/read_rainbow_h5ad` share one implementation (`cytovanni.io.h5ad`); names
  and behaviour unchanged.
- C12: `unrolled_subplots` always returns a 1-D numpy array of the N axes (was a list for 2-D grids);
  `sum(loss_containers)` works (adding 0 returns the container).
- Affected analyses: code calling the renamed functions/arguments must be ported (reproducibility: 2 files
  `plot_eval_RBInt_fit`, 2 files `rbint=`; tutorials ported); code that used the Series index of a CutoffUnmxGate /
  UnmxMinGate mask (none found: all usages index AnnData or combine masks); models saved without history now also
  contain the loss per step (larger by that list only).

### R5-D1 new modules `cytovanni.utils.alignment` and `cytovanni.utils.batcheval` (batch-integration evaluation)
- New: k-nearest-neighbour batch mixing and the Seurat alignment score (`alignment`: `knn_same_batch_counts`,
  `knn_same_batch_counts_grouped`, `alignment_design`, `alignment_scores`, `pair_alignment_scores`,
  `weighted_celltype_score`, ...) and the replicate-pair evaluation of the HD7 batch-integration paper (`batcheval`:
  per-cell-type summaries, MFI / sigmaFI / proportion CV tables, counting-noise corrections, Gaussian-negative
  control, replicate correlations). Ported from the reproducibility project's own evaluation code; built on
  `cytovanni.utils.evl` (unchanged).
- Fix during the port: `gaussian_negative` on a DataFrame uses `DataFrame.mask` (same values; the staged version
  assigned float64 noise into float32 columns, a FutureWarning under pandas 2 and a TypeError under pandas 3).
- Changes against the staged version (user 2026-11-08): `alignment_scores(scope="global")` gives NaN for cell types
  that fail the minimum-cell rule (were scored); `celltype_marker_summary` without the float32 special case of the
  mean (values may differ from the project's by one unit in the last place); `proportion_cv_table` ``log_std`` with
  ddof=1 also for pairs (was ddof=0; descriptive column only, the CVs are unchanged); `subtract_counting_noise`
  checks that the marker / cell type of every noise row matches the table.
- Effect on results: none for existing package code. For the reproducibility project: same numbers as its own code
  (checked there, bit-identical on the CPU and GPU) except that the single-sample (`samples=`, Ref0) proportion and
  sigmaFI CVs use `evl.get_positive_single_CV` with ddof=1 (project: ddof=0), larger by about sqrt(n / (n - 1)).
- Tests: tests/test_utils_alignment.py, tests/test_utils_batcheval.py (synthetic data, CPU and CUDA).
