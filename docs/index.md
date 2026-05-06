# Cytovanni
## Computational Standardization of Flow Cytometry Data

---

cite paper

## Installation

Cytovanni can be installed directly from pypi:
```bash
pip install cytovanni[all]
```

It requires a pytorch installation, preferrably on a machine with a GPU and CUDA for the GPU speedup. For the Sinkhorn divergence, we also require geomloss.

As setting up a working CUDA installation can be tiresome, we strongly suggest [using an apptainer container as a Jupyter kernel](tutorials/container) to run the package, which otherwise only requires installed Nvidia drivers.

## Tutorials
We provide a set of tutorials on how to use Cytovanni. First, a [cytometer configuration](tutorials/configuration/index) needs to be set up.
We then provide example workflows using either [only the rainbow calibration and marker normalization](tutorials/integration_fixed-spectra/index), or our [full pipeline](tutorials/integration_variable-spectra/index) including inferred dye spectra for every batch.





```{toctree}
:hidden:
:maxdepth: 2

Container <tutorials/container>
Configuration <tutorials/configuration/index>
Fixed Spectra Workflow <tutorials/integration_fixed-spectra/index>
Inferred Spectra Workflow <tutorials/integration_variable-spectra/index>
```
