# Snapshots

Stored results of the package on fixed synthetic inputs (small `.npz` files), compared by `tests/test_snapshots.py`
(marker `snapshot`, part of the default run, about 1 minute) with the tolerance in `cases.py`. They catch result
changes that are not intended. When a change is intended (a fix or an approved improvement), regenerate the affected
files and record the change in CHANGELOG.md:

    python tests/snapshots/generate.py [case ...]    # default: all cases

Cases (`cases.py`): `rainbow_gmm`, `rainbow_calibration`, `rainbow_calibration_covpenalty` (factors, shifts, peak
precisions; shortened schedule), `spill_stainfit` (linear and outlier-safe spectrum fits), `ref_scaling` and
`ref_spectralfit` (reference model, short fits), `utils_rate` (event rate, exact and coarse time stamps). Everything
runs on the CPU with one thread and fixed seeds.
