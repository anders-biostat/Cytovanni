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
| R2-C15b | dependencies: `pandas<3` pinned, `joblib` declared | no | installation |
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
  copy. No use of `fit_factor_Sinkhorn_LBFGS` was found.
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
  samples that showed the "trying again" warning. `processing/instrument_rainbow/lib/pipeline.py:53` passes a
  `RandomState` instance: unchanged.
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
