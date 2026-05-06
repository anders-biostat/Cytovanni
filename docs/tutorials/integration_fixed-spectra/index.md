# Standardisation with Fixed Dye Spectra

In this tutorial, we demonstrate how to standardise data while using only measured dye spectra.

First, we use [rainbow beads to calibrate the flow cytometer](tutorials/integration_fixed-spectra/integration_fixed-spectra-rainbow.ipynb), and apply this to the spectra and raw samples. Afterwards, we [fit a marker normalization using reference samples](tutorials/integration_fixed-spectra/integration_fixed-spectra-ref.ipynb) to normalize the unmixed marker intensities. While we strongly encourage the use of rainbow beads, the marker normalization can also be used independently from the cytometer standardization, and will be the dominant source of batch effects in many setups.

```{toctree}
:hidden:
:maxdepth: 2

Rainbow <integration_fixed-spectra-rainbow>
Model <integration_fixed-spectra-ref>
```