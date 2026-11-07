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
