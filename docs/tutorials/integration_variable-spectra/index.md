# Standardization with Inferred Dye Spectra

In this tutorial, we showcase our full pipeline, including inferring appropriate dye spectra for every batch.

As in the simpler case, we start with the [rainbow bead calibration](tutorials/integration_variable-spectra/integration_variable-spectra-rainbow.ipynb) and apply it to the samples and single stain measurements. Afterwards, we [model the dye spectra variability](tutorials/integration_variable-spectra/integration_variable-spectra-dye.ipynb) to compress the full spectral matrix into a smaller embedding. This then allows us to [fit a full standardisation from overlapping samples](tutorials/integration_variable-spectra/integration_variable-spectra-model.ipynb), including both marker normalization factors and inferred dye spectra for every batch.

```{toctree}
:hidden:
:maxdepth: 2

Rainbow <tutorials/integration_variable-spectra/integration_variable-spectra-rainbow>
Dye <tutorials/integration_variable-spectra/integration_variable-spectra-dye>
Full Model <tutorials/integration_variable-spectra/integration_variable-spectra-model>
```