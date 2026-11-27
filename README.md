<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/anders-biostat/Cytovanni/main/docs/_static/banner_dark.png">
  <img alt="Cytovanni - Computational Standardisation of Flow Cytometry Data" src="https://raw.githubusercontent.com/anders-biostat/Cytovanni/main/docs/_static/banner_light.png">
</picture>

**Cytovanni** removes batch effects from flow cytometry data, so that samples recorded months or years apart can be
analysed together. Unlike common integration tools, it does not warp the unmixed marker intensities. It models the
physical causes of batch effects and corrects them with linear transformations of the raw fluorescence intensities, as
part of the unmixing.

It assumes batch effects have two sources:

- **Instrument variability:** laser intensity, detector sensitivity and gain. Cytovanni calibrates each batch
  computationally from multipeak rainbow beads measured alongside the samples. This works even when the cytometer
  settings change between batches.
- **Dye variability:** dye spectra and staining intensity. Cytovanni infers each batch's dye spectra from samples
  measured in more than one batch, so single-stain controls are needed for only one batch. It then normalises each
  marker's staining intensity.

Because every correction is linear and has few parameters, technical variation is removed while biological differences
between samples are kept. The method also stays compatible with unstained and FMO controls.

The documentation is still a work in progress; a current draft of the associated paper [is available in this git.](https://raw.githubusercontent.com/anders-biostat/Cytovanni/main/docs/paper_draft.pdf)
