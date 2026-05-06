# Standardisation with Inferred Dye Spectra

In this tutorial, we showcase our full pipeline, including inferring appropriate dye spectra for every batch.

As in the simpler case, we start with the [rainbow bead calibration](integration_variable-spectra-rainbow) and apply it to the samples and single stain measurements. Afterwards, we [model the dye spectra variability](integration_variable-spectra-dye) to compress the full spectral matrix into a smaller embedding. This then allows us to [fit a full standardisation from overlapping samples](integration_variable-spectra-model), including both marker normalization factors and inferred dye spectra for every batch.

```{toctree}
:hidden:
:maxdepth: 2

Rainbow <integration_variable-spectra-rainbow>
Dye <integration_variable-spectra-dye>
Full Model <integration_variable-spectra-model>
```