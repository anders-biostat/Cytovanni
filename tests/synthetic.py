"""Synthetic data with known ground truth for the Cytovanni tests.

One small simulated cytometer (4 scatter + 3 fluorescence channels + Time) with three batches. Everything a fit
should recover is defined here as a module constant:

- ``PEAKS``: rainbow bead peak intensities (raw units) of the reference batch, the same in every fluorescence channel.
- ``GAINS``: detector gain per batch and fluorescence channel; a batch measures ``true * gain``. The calibration
  factor that maps a batch onto the reference batch "A" is therefore ``1 / gain`` (relative to "A").
- ``SCATTER_GAINS``: the same for the scatter channels (one value per batch, all scatter channels).
- ``SPECTRA``: dye spectra, shape (stain, channel), each normalised to 1 in its main channel.
- ``STAIN_SCALES``: staining intensity per batch and stain (what the reference model's stain factors undo).

Measured events are ``abundance @ SPECTRA * gain`` plus noise. Files are written with `cytovanni.io.writefcs`, so the
tests read them through the same code path as real data.

The last sections hold helpers shared by several test modules: rainbow fits with a short training schedule and
hand-made panel configurations (with the spectra they imply).
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import torch

from cytovanni.utils import CytometerConfiguration
from cytovanni.io import readfcs_rainbow_sample, writefcs
import cytovanni.rainbow.rbim as rbim
from cytovanni.rainbow import RainbowFluorescenceGMMCalibrationTrainer, fit_rainbow_GMM
from cytovanni.spill import PanelConfiguration

CHANNELS_SCATTER = ["FSC-A", "FSC-H", "SSC-A", "SSC-H"]
CHANNELS_FLUORESCENCE = ["FL1-A", "FL2-A", "FL3-A"]
CHANNELS = CHANNELS_SCATTER + CHANNELS_FLUORESCENCE + ["Time"]
BATCHES = ["A", "B", "C"]
STAINS = ["FL1", "FL2", "FL3"]
MARKERS = ["CD1", "CD2", "CD3"]

PEAKS = np.array([300., 4000., 40000.])
GAINS = pd.DataFrame([[1., 1., 1.], [1.5, .7, 1.2], [.8, 1.3, 1.]], index=BATCHES, columns=CHANNELS_FLUORESCENCE)
SCATTER_GAINS = pd.Series([1., 1.3, .9], index=BATCHES)
SPECTRA = pd.DataFrame([[1., .3, .05], [.1, 1., .4], [.02, .15, 1.]], index=STAINS, columns=CHANNELS_FLUORESCENCE)
STAIN_SCALES = pd.DataFrame([[1., 1., 1.], [1.4, .8, 1.], [.7, 1.2, 1.1]], index=BATCHES, columns=STAINS)
VOLTAGES = pd.DataFrame([[300, 310, 320], [350, 360, 370], [400, 410, 420]], index=BATCHES, columns=CHANNELS_FLUORESCENCE)
DATES = {"A": "01-JAN-2024", "B": "02-JAN-2024", "C": "03-JAN-2024"}


def make_config(**kwargs):
    """CytometerConfiguration of the simulated cytometer (keyword arguments override the defaults)."""
    kw = dict(channels_scatter=CHANNELS_SCATTER, channels_meta=["Time"], max_linear_range=2e5, cytometer="SIM",
              laserdict={"Blue": ["FL1-A", "FL2-A", "FSC-A", "SSC-A"], "Red": ["FL3-A"]})
    kw.update(kwargs)
    return CytometerConfiguration(CHANNELS_FLUORESCENCE, **kw)


def fcs_metadata(batch, btim="10:00:00"):
    """Text keywords of a file of `batch`: $DATE, $BTIM and the voltages ($PnV, used for the settings hash)."""
    meta = {"date": DATES[batch], "btim": btim}
    for i, c in enumerate(CHANNELS):
        if c in CHANNELS_FLUORESCENCE:
            meta[f"p{i+1}v"] = str(VOLTAGES.loc[batch, c])
        elif c != "Time":
            meta[f"p{i+1}v"] = "250"
    return meta


def scatter_events(n, rng, gain=1., debris=.1):
    """FSC-A, FSC-H, SSC-A, SSC-H of `n` events: a compact bead/cell cloud plus a fraction `debris` of small events."""
    nd = int(n * debris); nb = n - nd
    fsc = np.concatenate([1e4 * (1 + .08 * rng.normal(size=nb)), rng.uniform(500, 4000, nd)]) * gain
    ssc = np.concatenate([5e3 * (1 + .08 * rng.normal(size=nb)), rng.uniform(200, 2000, nd)]) * gain
    return {"FSC-A": fsc, "FSC-H": .9 * fsc, "SSC-A": ssc, "SSC-H": .9 * ssc}


def rainbow_events(batch, n=3000, seed=0, cv=.06, shift=0.):
    """Rainbow bead events of `batch` as a DataFrame (columns `CHANNELS`) and the true peak label of every event."""
    rng = np.random.default_rng(seed)
    cols = scatter_events(n, rng, SCATTER_GAINS[batch])
    label = rng.integers(0, len(PEAKS), n)
    for c in CHANNELS_FLUORESCENCE:
        mu = PEAKS[label] * GAINS.loc[batch, c] + shift
        cols[c] = mu * (1 + cv * rng.normal(size=n)) + 30 * rng.normal(size=n)
    cols["Time"] = np.sort(rng.uniform(0, 60, n))
    return pd.DataFrame(cols)[CHANNELS].astype(np.float32), label


def cell_abundances(n, rng, positive_fraction=.4):
    """True stain abundances (events, stain): lognormal positives and a negative population around 100."""
    pos = rng.random((n, len(STAINS))) < positive_fraction
    return np.where(pos, rng.lognormal(9.5, .4, (n, len(STAINS))), rng.normal(100, 60, (n, len(STAINS))))


def cell_events(batch, n=4000, seed=0, abundances=None, spectra=SPECTRA, stain_scale=True):
    """Stained cell events of `batch` (DataFrame, columns `CHANNELS`) and the true abundances before batch effects.

    Measured fluorescence = (abundance * STAIN_SCALES[batch]) @ spectra * GAINS[batch] + noise.
    """
    rng = np.random.default_rng(seed)
    ab = cell_abundances(n, rng) if abundances is None else np.asarray(abundances)
    scale = STAIN_SCALES.loc[batch].to_numpy()[None] if stain_scale else 1.
    fl = (ab * scale) @ spectra.to_numpy() * GAINS.loc[batch].to_numpy()[None] + 15 * rng.normal(size=(len(ab), 3))
    cols = scatter_events(len(ab), rng, SCATTER_GAINS[batch], debris=0.)
    cols.update({c: fl[:, i] for i, c in enumerate(CHANNELS_FLUORESCENCE)})
    cols["Time"] = np.sort(rng.uniform(0, 60, len(ab)))
    return pd.DataFrame(cols)[CHANNELS].astype(np.float32), ab


def single_stain_events(dye, batch="A", n=3000, seed=0, spectra=SPECTRA):
    """Single-stain control of `dye`: half the events stained (lognormal), half unstained."""
    rng = np.random.default_rng(seed)
    stain = np.where(rng.random(n) < .5, rng.lognormal(9, .4, n), rng.normal(50, 30, n))
    fl = stain[:, None] * spectra.loc[dye].to_numpy()[None] * GAINS.loc[batch].to_numpy()[None] + 20 * rng.normal(size=(n, 3))
    cols = scatter_events(n, rng, SCATTER_GAINS[batch])
    cols.update({c: fl[:, i] for i, c in enumerate(CHANNELS_FLUORESCENCE)})
    cols["Time"] = np.sort(rng.uniform(0, 60, n))
    return pd.DataFrame(cols)[CHANNELS].astype(np.float32)


def write_rainbow_fcs(path, batch, btim="09:00:00", **kwargs):
    """Write rainbow beads of `batch` to `path` (.fcs); returns `path`."""
    df, _ = rainbow_events(batch, **kwargs)
    writefcs(path, df, add_metadata=fcs_metadata(batch, btim))
    return path


def write_cell_fcs(path, batch, btim="10:30:00", **kwargs):
    """Write stained cells of `batch` to `path` (.fcs); returns `path`."""
    df, _ = cell_events(batch, **kwargs)
    writefcs(path, df, add_metadata=fcs_metadata(batch, btim))
    return path


def write_single_stains(folder, batch="A", seed=0, n=3000):
    """Write one single-stain control per stain plus an unstained control into `folder` (BD naming)."""
    os.makedirs(folder, exist_ok=True)
    for i, (dye, marker) in enumerate(zip(STAINS, MARKERS)):
        df = single_stain_events(dye, batch, n=n, seed=seed + i)
        writefcs(os.path.join(folder, f"Compensation Controls_{marker} {dye} Stained Control.fcs"), df,
                 add_metadata=fcs_metadata(batch, "11:00:00"))
    df, _ = cell_events(batch, n=n, seed=seed + 99, abundances=np.random.default_rng(seed).normal(0, 30, (n, 3)),
                        stain_scale=False)
    writefcs(os.path.join(folder, "Unstained Control.fcs"), df, add_metadata=fcs_metadata(batch, "11:05:00"))
    return folder


def make_adata(x, columns, layer="calibrated", obs_index=None, **uns):
    """AnnData with `x` (events, channel) in ``X`` and ``layers[layer]``, and the given ``uns`` entries."""
    x = np.asarray(x, dtype=np.float32)
    obs = pd.DataFrame(index=[str(i) for i in range(len(x))] if obs_index is None else obs_index)
    adata = ad.AnnData(x, var=pd.DataFrame(index=list(columns)), obs=obs)
    adata.layers[layer] = adata.X.copy()
    adata.uns.update(uns)
    return adata


def cell_adata(batch, n=2000, seed=0, abundances=None, uid=None, **uns):
    """Cells of `batch` already on the calibrated scale (gains removed), fluorescence channels only.

    ``layers['calibrated']`` = (abundance * STAIN_SCALES[batch]) @ SPECTRA + noise; ``uns`` holds ``batch`` and
    ``uid`` (default: the batch), plus `uns`.
    """
    rng = np.random.default_rng(seed)
    ab = cell_abundances(n, rng) if abundances is None else np.asarray(abundances)
    x = (ab * STAIN_SCALES.loc[batch].to_numpy()[None]) @ SPECTRA.to_numpy() + 15 * rng.normal(size=(len(ab), 3))
    return make_adata(x, CHANNELS_FLUORESCENCE, batch=batch, uid=batch if uid is None else uid, type="same", **uns)


# ======================================================================== rainbow fits (shared by test modules)
class ShortTrainer(RainbowFluorescenceGMMCalibrationTrainer):
    """The trainer with a short default schedule, for `fit_fluorescence` in tests."""
    def fit_init_peak(self, dataset, Niter=200, **kw):
        return super().fit_init_peak(dataset, Niter=Niter, **kw)

    def fit_init_cov(self, dataset, Niter=5, **kw):
        return super().fit_init_cov(dataset, Niter=Niter, **kw)

    def fit(self, dataset, Niter=10, **kw):
        return super().fit(dataset, Niter=Niter, **kw)


def short_fit_fluorescence(rb, trainer=ShortTrainer, **kwargs):
    """`rb.fit_fluorescence(**kwargs)` with `trainer` instead of the full training schedule."""
    kwargs = {"gpu": False, "verbosity": 0, **kwargs}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rbim, "RainbowFluorescenceGMMCalibrationTrainer", trainer)
        rb.fit_fluorescence(**kwargs)


def expected_rainbow_factors(batches=BATCHES):
    """True fluorescence factors 1 / gain, geometric mean 1 over `batches`."""
    f = 1 / GAINS.loc[batches]
    return f / np.exp(np.log(f).mean())


def read_rainbow_fit(path, config, batch, rainbow_type="T", seed=0):
    """Read a rainbow bead file of `batch` and fit its GMM (all peaks of `PEAKS`)."""
    ad = readfcs_rainbow_sample(path, config, len(PEAKS), name=os.path.basename(path), rainbow_batch=batch,
                                rainbow_type=rainbow_type)
    fit_rainbow_GMM(ad, random_state=seed)
    return ad


# ======================================================================== panel configurations (shared by test modules)
# Panel read from a csv, with hand-made PCA data (used by the transformer and model tests)
# PCA components of the test panel (component, channel): FL1 spills more or less into FL2, FL2 into FL3; the FL3
# component exists but is not used by the panel
PANEL_COMPONENTS = pd.DataFrame([[0., .1, 0.], [0., 0., .1], [.05, 0., 0.]], index=["FL1_PC0", "FL2_PC0", "FL3_PC0"], columns=CHANNELS_FLUORESCENCE)
PANEL_COMPONENTS_USE = pd.Series([True, True, False], index=PANEL_COMPONENTS.index)


def make_panelconfig_csv(folder):
    """PanelConfiguration of the simulated panel with hand-made PCA data (offset = SPECTRA)."""
    path = folder / "panel.csv"
    pd.DataFrame({"stain": STAINS, "marker": MARKERS, "components": ["0", "0", ""]}).to_csv(path, index=False)
    pc = PanelConfiguration.from_csv(str(path), "sim")
    pc.pca_components = PANEL_COMPONENTS.copy()
    pc.pca_offset = SPECTRA.copy()
    pc.pca_components_stain = pd.Series(STAINS, index=PANEL_COMPONENTS.index)
    pc.pca_components_use = PANEL_COMPONENTS_USE.copy()
    pc.pca_explained_variance = pd.Series([.02, .01, .005], index=PANEL_COMPONENTS.index)
    pc.pca_embedding = pd.DataFrame([[1., -1., 0.]], index=["comp_run"], columns=PANEL_COMPONENTS.index)
    pc.channels = CHANNELS_FLUORESCENCE
    pc.has_PCA_data = True
    return pc


def panel_spectra_of(embedding):
    """Spectra (stain, channel) of the test panel for an embedding (component,), unused components masked."""
    e = np.asarray(embedding, dtype=float) * PANEL_COMPONENTS_USE.to_numpy()
    sp = SPECTRA.to_numpy().copy()
    for k, comp in enumerate(PANEL_COMPONENTS.index):
        sp[STAINS.index(comp.split("_")[0])] += e[k] * PANEL_COMPONENTS.loc[comp].to_numpy()
    return np.clip(sp, 0, None)


def set_transformer_params(tr, emb=None, stain=None, channel=None):
    """Set the trainable parts of a transformer (values per batch, before masks)."""
    with torch.no_grad():
        if emb is not None:
            tr.pem.param_embedding["batch"].param_trainable.data[:] = torch.as_tensor(emb, dtype=torch.float32)
        if stain is not None:
            tr.param_stainscale.parameter.param_trainable.data[:] = torch.as_tensor(stain, dtype=torch.float32)
        if channel is not None:
            tr.param_channelscale.parameter.param_trainable.data[:] = torch.as_tensor(channel, dtype=torch.float32)


# Panel built directly (used by the PanelModule and panel-embedding tests)
# components of the square panel (3 stains x 3 channels): zero in each stain's main channel (normalised spectra)
COMPONENTS_SQUARE = {
    "FL1": [[0., .05, -.02], [0., .01, .03]],
    "FL2": [[.04, 0., .03]],
    "FL3": [[.01, .03, 0.]],
}
USE_SQUARE = {"FL1_PC0": True, "FL1_PC1": False, "FL2_PC0": True, "FL3_PC0": True}


def make_panelconfig_pca(offset=None, components=None, use=None, markers=None, n_measurements=4, seed=0):
    """PanelConfiguration with PCA data: `offset` (stain, channel) DataFrame, `components` stain -> list of vectors."""
    offset = SPECTRA.copy() if offset is None else offset
    components = COMPONENTS_SQUARE if components is None else components
    use = USE_SQUARE if use is None else use
    stains = offset.index.to_numpy()
    markers = np.array(MARKERS[:len(stains)] if markers is None else markers)
    names, rows, comp_stain = [], [], []
    for s in stains:
        for i, v in enumerate(components.get(s, [])):
            names.append(f"{s}_PC{i}"); rows.append(v); comp_stain.append(s)
    pc = PanelConfiguration()
    pc.name = "test_panel"
    pc.stains, pc.markers = stains, markers
    pc.stain_marker_name = np.array([f"{s} | {m}" for s, m in zip(stains, markers)])
    pc.pca_components = pd.DataFrame(np.asarray(rows, dtype=float).reshape(len(rows), offset.shape[1]), index=names,
                                     columns=offset.columns)
    pc.pca_offset = offset
    pc.pca_components_stain = pd.Series(comp_stain, index=names)
    pc.pca_components_use = pd.Series([bool(use[n]) for n in names], index=names)
    pc.components = np.array([np.array([i for i, n in enumerate(n for n in names if n.startswith(s + "_"))
                                        if use[f"{s}_PC{i}"]], dtype=int) for s in stains], dtype=object)
    pc.total_components = int(pc.pca_components_use.sum())
    rng = np.random.default_rng(seed)
    pc.pca_embedding = pd.DataFrame(rng.normal(size=(n_measurements, len(names))), index=[f"run{i}" for i in range(n_measurements)],
                                    columns=names)
    pc.channels = list(offset.columns)
    pc.has_PCA_data = True
    return pc
