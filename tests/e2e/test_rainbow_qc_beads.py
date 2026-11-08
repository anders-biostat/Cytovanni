"""Regression test of the covariance penalty (FIXLOG R4-P4d) on real QC beads: Aria III calibration beads, which do
not fluoresce in every channel (few bright peaks per laser).

Without the penalty, the free per-measurement peak precisions let a peak broaden during the full fit, and the laser
factors of some measurements drift away from the others (here the five known cases, e.g. the Green laser by a factor
of ~3.7 after 2000 steps); with ``cov_penalty=.3`` they stay at the level of the other measurements.

Data: ``CYTOVANNI_TEST_QC_BEADS``, a folder with the GMM-fitted calibration-bead measurements (.h5ad, written by
`write_rainbow_h5ad`); the test uses 24 of them, including the five. Skipped if not set. GPU, a few minutes.
"""
import glob
import os

import numpy as np
import pandas as pd
import pytest
import torch

from cytovanni.io import read_rainbow_h5ad
from cytovanni.rainbow import (RainbowCalibrator, RainbowFluorescenceGMMCalibrationDataset,
                               RainbowFluorescenceGMMCalibrationTrainer)

OUTLIERS = ["1633684819", "1638967469", "1639055911", "1639381690", "1639660868"]
ANCHORS = ["B1 (525)-A", "G1 (586)-A", "R1 (670)-A", "V1 (450)-A"]   # one channel per laser
UV_OFF = 1000          # UV laser counted as off below this GMM mean in UV2
NITER = 2000

pytestmark = [pytest.mark.slow, pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")]


@pytest.fixture(scope="module")
def files():
    folder = os.environ.get("CYTOVANNI_TEST_QC_BEADS")
    if not folder or not os.path.isdir(folder):
        pytest.skip("QC bead data not available (set CYTOVANNI_TEST_QC_BEADS)")
    allf = sorted(glob.glob(os.path.join(folder, "*.h5ad")))
    pick = [f for f in allf if any(o in os.path.basename(f) for o in OUTLIERS)] + allf[::max(1, len(allf) // 19)][:19]
    pick = sorted(set(pick))
    assert sum(any(o in os.path.basename(f) for o in OUTLIERS) for f in pick) == len(OUTLIERS)
    return pick


def load(files):
    """Each measurement its own batch; UV1 and switched-off UV lasers excluded (as in the instrument pipeline)."""
    ads = [read_rainbow_h5ad(p) for p in files]
    ads.sort(key=lambda ad: ad.uns["UTC"])
    for ad in ads:
        ad.uns["rainbow_batch"] = ad.uns["uid"]
        ch = pd.Index(ad.uns["cytoconfig"].channels_fluorescence)
        ad.uns["GMM_means"][:, ch.get_indexer(["UV1 (379)-A"])] = 0
        if ad.uns["GMM_means"][:, ch.get_loc("UV2 (515)-A")].max() < UV_OFF:
            ad.uns["GMM_means"][:, ch.str.startswith("UV")] = 0
    return ads


def anchor_deviation(files, cov_penalty):
    """Largest |log factor - median over measurements| per anchor channel: (outliers, others)."""
    torch.manual_seed(0); np.random.seed(0)
    rb = RainbowCalibrator(load(files))
    rb.channel_missing_cutoff = 5
    ds = RainbowFluorescenceGMMCalibrationDataset(rb.rboe, rb.adatas, device="cuda:0", channel_missing_cutoff=5)
    rb.dataset = ds
    rb.rfim.to("cuda:0")
    tr = RainbowFluorescenceGMMCalibrationTrainer(rb.rfim, cov_penalty=cov_penalty)
    rb.rfim_trainer = tr
    tr.fit_init(ds, showprogress=False)
    tr.fit(ds, Niter=NITER, showprogress=False)
    f = np.log(rb.get_calibration_factors_all(warn_untrained=False))[ANCHORS]
    dev = (f - f.median(0)).abs()
    isout = dev.index.to_series().apply(lambda u: any(o in u for o in OUTLIERS))
    return dev[isout].max(), dev[~isout].max()


def test_without_penalty_the_known_measurements_drift(files):
    """The failure the penalty fixes is present in this setup (otherwise the next test proves nothing)."""
    out, others = anchor_deviation(files, 0.)
    assert out["G1 (586)-A"] > 1., out


def test_cov_penalty_keeps_the_factors_together(files):
    out, others = anchor_deviation(files, .3)
    assert (out < .6).all(), pd.DataFrame({"outliers": out, "others": others})
