# Conventions

*Draft, filled in during the docstring review (R3).*

## Axis names
Shapes in the docstrings use these names: `events`, `channel` (detector channels), `stain` / `dye`, `batch`,
`component` (PCA components of a dye spectrum), `embedding`.

## Spectra orientation
- **(stain, channel)**: spectra used for unmixing and by the panel modules (`cytovanni.ref`, `cytovanni.torch`
  unmixing functions, `cytovanni.utils.invert_spectra`).
- **(channel, dye)**: spectra in the spillover collection and calibrated spectra (`cytovanni.spill`,
  `cytovanni.rainbow.scale_spectra_wfactors`).

## Data scales
- **raw**: intensities as read from the .fcs file (`adata.layers['raw']`).
- **calibrated**: raw intensities multiplied by the rainbow calibration factors (`adata.layers['calibrated']`).
- **arcsinh**: `arcsinh(x / cofactor)`, with the cofactor in raw intensity units.

## Batch keys
To be written.
