# Cytovanni

Computational Standardization of Flow Cytometry Data

---

## Tutorials

The first tutorial shows how to use [rainbow beads to standardize the flow cytometer](tutorials/integration_fixed-spectra/integration_fixed-spectra-rainbow.ipynb), and [fit scaling factors for each marker using reference samples](tutorials/integration_fixed-spectra/integration_fixed-spectra-ref.ipynb). While we strongly encourage the use of rainbow beads, the scaling factors can also be used independently from the cytometer standardization, and will be the dominant source of batch effects in many setups.

The second tutorial also uses the [rainbow bead standardization](tutorials/integration_variable-spectra/integration_variable-spectra-rainbow.ipynb), which is applied to the samples, as well as to single stain spectra. We then [model the dye variability](tutorials/integration_variable-spectra/integration_variable-spectra-dye.ipynb), which allows us to [fit a full integration from overlapping samples](tutorials/integration_variable-spectra/integration_variable-spectra-model.ipynb) including both marker scaling factors, as well as to extract dye spectra for every batch from the overlapping samples.
