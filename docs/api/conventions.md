# Conventions

Names, shapes, scales and keys used throughout the API reference.

## Terms
- **channel**: a detector channel of the cytometer (an .fcs parameter, e.g. `BV421-A` or `V1 (420)-A`).
- **dye**: a fluorophore as named in a single-stain measurement: a spectrum column of `SpilloverMeasurement.spectra`
  and the index of its `panel_meta` (for conventional flow typically the channel name without `-A`).
- **stain**: what a spectrum or a factor belongs to in an analysis. In `cytovanni.spill` it is the value of
  `panel_meta[register_on]` (the keys of `SpilloverCollection`, `PanelConfiguration.stains`); in `cytovanni.ref` the
  dye names of the panel (the index of the spectra).
- **marker**: the antigen a stain is bound to; `stain_marker_name` is the combined label `"stain | marker"`.
- **component**: a PCA component of the spectrum of one stain, named `'{stain}_PC{i}'` (i from 0).
- **batch**: a group of measurements that share one set of correction parameters (e.g. a measurement day); see
  [Batch keys](#batch-keys).

## Axis names
Shapes in the docstrings use these names: `events`, `channel`, `stain` / `dye`, `marker`, `batch`, `peak` (rainbow
bead peaks), `component`, `measurement` (single-stain runs).

## Spectra orientation
- **(stain, channel)**: spectra used for unmixing and by the panel modules: `cytovanni.ref`, the unmixing functions of
  `cytovanni.torch` and `cytovanni.utils`, `PanelConfiguration` (`get_run_spectra`, `get_spectra_from_embedding`,
  `pca_offset`), `cytovanni.io.readfcs_spill`. Torch functions take batches of spectra, shape (..., stain, channel).
- **(channel, stain)**: inverted spectra (unmixing matrices), e.g. `cytovanni.utils.invert_spectra`; unmixing is
  `x @ inverse` for `x` of shape (events, channel).
- **(channel, dye)**: spectra of one single-stain measurement: `SpilloverMeasurement.spectra` /
  `spectra_calibrated`, the `spectra_*.csv` files of a single-stain folder, `cytovanni.io.read_spill_S8`,
  `cytovanni.rainbow.scale_spectra_wfactors`.
- **(measurement, channel)**: the spectra of one stain collected over many measurements:
  `SingleStainSpilloverCollection.spectra`, `SingleSpectrumPCAFitter`, `PanelConfiguration.spectra_full`; the
  corresponding PCA embeddings are (measurement, component).

Every docstring states the orientation it uses.

## Data scales
- **raw**: intensities as read from the .fcs file (`adata.layers['raw']`).
- **calibrated**: raw intensities multiplied by the rainbow calibration factors (`adata.layers['calibrated']`), see
  [Calibration factors](#calibration-factors).
- **unmixed**: calibrated (or standardised) intensities unmixed into stains, linear scale, e.g. `adata.obsm['unmx']`.
- **arcsinh**: `arcsinh(x / cofactor)`, with the cofactor in the units of `x` (raw intensity units). The default
  cofactor depends on the purpose: 5000 for the losses of the fits (Sinkhorn factors, transformers), 1500 for
  outputs, evaluation and plots (`add_ad_standardised`, `apply_arcsinh`, `add_adata_unmixed`, the panel diagnostics,
  `cytovanni.utils.evl`). Gate thresholds are on the scale of the data they gate.
- **embedding units**: panel embeddings are in units of the standard deviation of each component over the
  single-stain measurements, so `embedding_step=1` or `max_abs_embedding=6` are in standard deviations.
- **CVs** (`cytovanni.utils.evl`) are fractions, not percent.

## Calibration factors
- Calibrated = raw × factor, one factor per (batch, channel); the instrument gain relative to the reference is
  1 / factor.
- Log factors are natural logs. Rainbow calibration factors have geometric mean 1 per channel over the batches of the
  first fit; to go from batch 1 to batch 2, multiply by factors₁ / factors₂.
- The scale factors of the reference model (`cytovanni.ref`: stain and channel factors, `learned_stainlogscale`)
  likewise multiply the linear data.
- Rainbow shifts are in raw units and added before the factors, (raw + shift) × factor; they apply to rainbow bead
  measurements only.

## AnnData layout
`cytovanni.io.readfcs_sample` returns an AnnData with
- `X` empty, `layers['raw']` (events, channel);
- `var`: one row per channel, columns `pnn`, `pns`, `png`, `pnv`, `pnr` (the .fcs parameter keywords);
- `obs`: the meta channels (e.g. `Time`, in seconds);
- `uns`: `date`, `UTC` (seconds since the epoch), `cytometer`, `uid`, `cytoconfig`, `set_hash` (hash of the channel
  settings), `Nevents`. Rainbow bead samples (`readfcs_rainbow_sample`) add `name`, `name_uid`, `N_rainbow_peak`,
  `rainbow_type` and `rainbow_batch`.

Later steps add layers (`calibrated`), `obsm` entries (unmixed data and their arcsinh versions) and the fitted
parameters (`var`, `varm`, `uns`); each function documents what it writes.

`fit_rainbow_GMM` stores its results in `uns` as arrays of shape (peak, channel) or (peak, channel, channel), with
row k belonging to `obs['GMM_label'] == k` and the channels `cytoconfig.channels_fluorescence`; `GMM_means` is on
the raw scale, the `GMM_asinh_*` entries on `arcsinh(raw / GMM_asinh_cofactor)`.

## Batch keys
Batch labels are stored per sample in `adata.uns`:
- rainbow calibration: `cytometer`, `rainbow_batch` (a period with unchanged settings on one cytometer),
  `rainbow_type` (bead type or lot) and `uid` (one measurement); samples are assigned to a rainbow batch from
  cytometer, date/UTC and settings hash;
- reference model (`cytovanni.ref`): any keys chosen by the user (e.g. `batch`, `date`), listed when creating the
  `MultiOrdinalEncoder`.

`MultiOrdinalEncoder` maps the labels of every key to 0-based batch indices (labels as strings, sorted). Two index
forms are used: `bidx`, a dict of key to `torch.LongTensor` of shape (events,) or (1,), and `anchor_bidx`, a dict of
key to an int or int array. For nested keys, `bidx` also holds `key + "_imputed"` (bool).

## Gates
`gate.apply(adata)` and `gate.apply_df(df)` return boolean masks (True = kept). `apply_df` works on the raw event
DataFrame of an .fcs file (columns: channel names) and ignores the gate's `layer` / `obsm` settings.
