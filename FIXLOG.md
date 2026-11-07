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
