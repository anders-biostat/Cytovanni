"""Snapshot cases: fixed synthetic inputs -> stored results (``<name>.npz`` in this folder).

Each case is a function returning a dict of numpy arrays. Everything runs on the CPU with fixed seeds, so the
results are reproducible up to floating point differences between machines; the comparison (tests/test_snapshots.py)
uses `RTOL` / `ATOL`. Regenerate after an intended change with ``python tests/snapshots/generate.py [case ...]`` and
record the change in CHANGELOG.md.

The cases cover the main numerical steps: rainbow GMM, rainbow fluorescence + scatter calibration (also with the
covariance penalty), single-stain spectrum fits, the reference model with a scaling and with a spectral-fit
transformer, and the event-rate estimate.
"""
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import synthetic  # noqa: E402
from cytovanni.io import readfcs_rainbow_sample  # noqa: E402
from cytovanni.rainbow import (RainbowCalibrator, RainbowFluorescenceGMMCalibrationTrainer,  # noqa: E402
                               fit_rainbow_GMM)
from cytovanni.ref import (MultiOrdinalEncoder, OverlapFitModel, PanelEmbeddingModule, SampleOverlapDataset,  # noqa: E402
                           ScalingTransformer, SpectralFitTransformer)
from cytovanni.spill.stainfit import LinearSingleStainSpectrumFit, OutliersafeLinearSingleStainSpectrumFit  # noqa: E402
from cytovanni.utils import EventRateEstimator  # noqa: E402

FOLDER = os.path.dirname(os.path.abspath(__file__))
RTOL, ATOL = 1e-3, 1e-5
FL = synthetic.CHANNELS_FLUORESCENCE


def seed(s=0):
    np.random.seed(s)
    torch.manual_seed(s)


class SnapshotTrainer(RainbowFluorescenceGMMCalibrationTrainer):
    """Shortened schedule (the default 12000 steps are too slow for the suite), longer than the unit-test one."""
    def fit_init_peak(self, dataset, Niter=200, **kw):
        return super().fit_init_peak(dataset, Niter=Niter, **kw)

    def fit_init_cov(self, dataset, Niter=5, **kw):
        return super().fit_init_cov(dataset, Niter=Niter, **kw)

    def fit(self, dataset, Niter=150, **kw):
        return super().fit(dataset, Niter=Niter, **kw)


def _rainbow_ads(folder):
    config = synthetic.make_config()
    ads = []
    for i, b in enumerate(synthetic.BATCHES):
        for k in range(2):
            p = synthetic.write_rainbow_fcs(os.path.join(folder, f"rainbow_{b}_{k}.fcs"), b, btim=f"0{8+k}:00:00",
                                            seed=10 * i + k)
            ads.append(synthetic.read_rainbow_fit(p, config, b))
    return ads


def rainbow_gmm():
    config = synthetic.make_config()
    with tempfile.TemporaryDirectory() as folder:
        p = synthetic.write_rainbow_fcs(os.path.join(folder, "rb.fcs"), "B", seed=3)
        ad = readfcs_rainbow_sample(p, config, len(synthetic.PEAKS), name="rb", rainbow_batch="B")
    fit_rainbow_GMM(ad, random_state=0)
    return {"means": ad.uns["GMM_means"], "asinh_stds": ad.uns["GMM_asinh_stds"],
            "cofactor": np.asarray(ad.uns["GMM_asinh_cofactor"]),
            "label_counts": np.bincount(ad.obs["GMM_label"].to_numpy().astype(int), minlength=len(synthetic.PEAKS))}


def _rainbow_calibration(**fitkw):
    with tempfile.TemporaryDirectory() as folder:
        ads = _rainbow_ads(folder)
    seed()
    rb = RainbowCalibrator(ads)
    import cytovanni.rainbow.rbim as rbim
    orig = rbim.RainbowFluorescenceGMMCalibrationTrainer
    rbim.RainbowFluorescenceGMMCalibrationTrainer = SnapshotTrainer
    try:
        rb.fit_fluorescence(gpu=False, verbosity=0, **fitkw)
    finally:
        rbim.RainbowFluorescenceGMMCalibrationTrainer = orig
    factors = rb.get_calibration_factors_all()
    shifts = rb.get_rainbow_shifts_all()["T"][FL]
    with torch.no_grad():
        precision = rb.rfim.param_cov["T"]()[0].diagonal(dim1=-2, dim2=-1).numpy()   # (measurement, peak, channel)
    out = {"factors_fluorescence": factors[FL].to_numpy(), "factors_scatter": factors[synthetic.CHANNELS_SCATTER].to_numpy(),
           "shifts": shifts.to_numpy(), "precision_diag": precision,
           "loss_last": np.asarray(rb.rfim_trainer.fit_losshistory[-10:])}
    if rb.rfim_trainer.cov_reference is not None:
        with torch.no_grad():
            out["cov_penalty"] = np.asarray(rb.rfim_trainer.get_cov_penalty().item())
    return out


def rainbow_calibration():
    return _rainbow_calibration()


def rainbow_calibration_covpenalty():
    return _rainbow_calibration(cov_penalty=.3)


def _ss_adata(dye, n=3000, seed_=0, offset=(100., 200., -50.)):
    import anndata as ad
    df = synthetic.single_stain_events(dye, "A", n=n, seed=seed_)
    x = df[FL].to_numpy(np.float64) + np.asarray(offset)[None]
    a = ad.AnnData(x.astype(np.float32), var=pd.DataFrame(index=FL), obs=pd.DataFrame(index=[str(i) for i in range(n)]))
    a.layers["raw"] = x
    a.uns.update({"cytoconfig": synthetic.make_config(), "dye": dye, "channel": dye + "-A"})
    return a


def spill_stainfit():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dye in synthetic.STAINS:
            seed()
            out[f"linear_{dye}"] = LinearSingleStainSpectrumFit(_ss_adata(dye), robust=True).spectrum.to_numpy()
            seed()
            out[f"outliersafe_{dye}"] = OutliersafeLinearSingleStainSpectrumFit(_ss_adata(dye)).spectrum.to_numpy()
    return out


def _dataset(ads, moe, anchors=("A",), sN_cells=400):
    isanchor = [ad.uns["batch"] in anchors for ad in ads] if anchors else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SampleOverlapDataset(ads, moe, FL, data_isanchor=isanchor, sN_cells=sN_cells, min_sample_cells=100)


def ref_scaling():
    ads = synthetic.overlap_adatas(n=1000)
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    seed()
    ds = _dataset(ads, moe, anchors=None, sN_cells=500)
    tr = ScalingTransformer.from_data(moe, synthetic.SPECTRA, scale_stains=True, anchor_bidx={"batch": 0})
    m = OverlapFitModel.from_data(moe, tr, ds, DDIST_separate=True)
    seed()
    m.fit(Niter=60, lr=5e-2)
    return {"stainlogscale": tr.learned_stainlogscale.to_numpy(), "loss": np.asarray([h["loss"] for h in m.losshistory["train"][-10:]])}


def ref_spectralfit():
    true_emb = np.array([1.5, -1., 0.])
    ab = synthetic.cell_abundances(1500, np.random.default_rng(3))
    ads = []
    for i, b in enumerate(["A", "B"]):
        sp = synthetic.panel_spectra_of(true_emb if b == "B" else np.zeros(3))
        x = ab @ sp + 15 * np.random.default_rng(i).normal(size=ab.shape)
        ads.append(synthetic.make_adata(x, FL, batch=b, uid=f"ref_{b}", type="ref"))
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    with tempfile.TemporaryDirectory() as folder:
        import pathlib
        pc = synthetic.make_panelconfig_csv(pathlib.Path(folder))
    pem = PanelEmbeddingModule.from_data(pc, moe, pd.Series("batch", index=synthetic.PANEL_COMPONENTS.index),
                                         anchor_bidx={"batch": [0]})
    tr = SpectralFitTransformer.from_data(moe, pem, anchor_bidx={"batch": [0]}, arcsinh_cofactor=1500)
    seed()
    ds = _dataset(ads, moe, sN_cells=600)
    m = OverlapFitModel(moe, tr, dataset=ds)
    seed()
    m.fit(Niter=80, lr=5e-2)
    return {"embedding": pem.get_learned_embedding()["batch"].to_numpy(), "stainlogscale": tr.learned_stainlogscale.to_numpy()}


def utils_rate():
    rng = np.random.default_rng(5)
    t = np.cumsum(rng.exponential(1 / 200., 20000))
    t_coarse = np.round(t / .01) * .01
    return {"rate": np.asarray([EventRateEstimator(t).rate, EventRateEstimator(t_coarse).rate])}


CASES = {f.__name__: f for f in [rainbow_gmm, rainbow_calibration, rainbow_calibration_covpenalty, spill_stainfit,
                                 ref_scaling, ref_spectralfit, utils_rate]}


def path(name):
    return os.path.join(FOLDER, f"{name}.npz")
