"""cytovanni.rainbow.gmm and cytovanni.rainbow.plot: per-measurement GMM peak assignment, likelihood gate, plots."""
import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

import cytovanni.rainbow.gmm as gmm
from cytovanni.exceptions import GMMAssignmentWarning
from cytovanni.io import readfcs_rainbow_sample
from cytovanni.rainbow import (fit_rainbow_GMM, get_rainbow_GMM_gate, plot_rainbow_channel_GMM_single, plot_GMM_llh)
from cytovanni.rainbow.gmm import _GMM_choose_asinh_cofactor, _get_rainbow_GMM_cutoff
from cytovanni.rainbow.plot import _get_max_separation_channel

import synthetic

FL = synthetic.CHANNELS_FLUORESCENCE
NPEAK = len(synthetic.PEAKS)


@pytest.fixture(scope="module")
def rainbow_raw(rainbow_files, config):
    """One unfitted rainbow measurement per batch (first file), read with readfcs_rainbow_sample."""
    return {b: readfcs_rainbow_sample(paths[0], config, NPEAK, name=f"rb_{b}", rainbow_batch=b, rainbow_type="T")
            for b, paths in rainbow_files.items()}


@pytest.fixture(scope="module")
def rainbow_fitted(rainbow_raw):
    """The measurements of `rainbow_raw` after fit_rainbow_GMM(random_state=0) (do not modify; copy first)."""
    out = {}
    for b, ad in rainbow_raw.items():
        ad = ad.copy()
        fit_rainbow_GMM(ad, random_state=0)
        out[b] = ad
    return out


def _true_labels(batch, k=0):
    """True peak label of every event of the k-th rainbow file of `batch` (the generator is deterministic)."""
    i = synthetic.BATCHES.index(batch)
    return synthetic.rainbow_events(batch, seed=10 * i + k)[1]


# ---------------------------------------------------------------- cofactor
def test_choose_cofactor_maps_geometric_mean_percentile_to_target(rainbow_raw):
    ad = rainbow_raw["B"]
    p90 = np.percentile(ad[:, FL].layers["raw"], 90, axis=0)
    expected = np.exp(np.log(p90).mean()) / np.sinh(7)
    assert _GMM_choose_asinh_cofactor(ad) == pytest.approx(expected)
    assert _GMM_choose_asinh_cofactor(ad, percentile=50, target=5) == pytest.approx(
        np.exp(np.log(np.percentile(ad[:, FL].layers["raw"], 50, axis=0)).mean()) / np.sinh(5))


def test_choose_cofactor_default_without_positive_percentile(rainbow_raw):
    ad = rainbow_raw["A"].copy()
    ad.layers["raw"] = -np.abs(ad.layers["raw"])
    assert _GMM_choose_asinh_cofactor(ad, default=1234.) == 1234.


# ---------------------------------------------------------------- fit_rainbow_GMM: outputs and recovery
def test_fit_writes_documented_outputs(rainbow_fitted):
    ad = rainbow_fitted["A"]
    assert ad.obs["GMM_label"].dtype.kind == "i" and set(ad.obs["GMM_label"]) == set(range(NPEAK))
    assert np.isfinite(ad.obs["GMM_llh"]).all()
    assert ad.uns["GMM_means"].shape == (NPEAK, len(FL))
    assert ad.uns["GMM_asinh_means"].shape == (NPEAK, len(FL))
    assert ad.uns["GMM_asinh_covariances"].shape == (NPEAK, len(FL), len(FL))
    assert ad.uns["GMM_asinh_stds"].shape == (NPEAK, len(FL))
    assert ad.uns["GMM_order_ok"] is True
    cof = ad.uns["GMM_asinh_cofactor"]
    assert cof == pytest.approx(_GMM_choose_asinh_cofactor(ad))
    # raw means are sinh(asinh means) * cofactor; stds are the sqrt of the covariance diagonals
    np.testing.assert_allclose(ad.uns["GMM_means"], np.sinh(ad.uns["GMM_asinh_means"]) * cof)
    np.testing.assert_allclose(ad.uns["GMM_asinh_stds"] ** 2,
                               np.diagonal(ad.uns["GMM_asinh_covariances"], axis1=1, axis2=2))


@pytest.mark.parametrize("batch", synthetic.BATCHES)
def test_fit_recovers_peaks_dim_to_bright(rainbow_fitted, batch):
    ad = rainbow_fitted[batch]
    expected = synthetic.PEAKS[:, None] * synthetic.GAINS.loc[batch].to_numpy()[None]
    np.testing.assert_allclose(ad.uns["GMM_means"], expected, rtol=.03)
    # labels match the true peaks (peaks are well separated)
    assert (ad.obs["GMM_label"].to_numpy() == _true_labels(batch)).mean() > .99
    # row k belongs to GMM_label == k: the raw median of the label-k beads is near row k
    med = pd.DataFrame(ad[:, FL].layers["raw"]).groupby(ad.obs["GMM_label"].to_numpy()).median().to_numpy()
    np.testing.assert_allclose(med, ad.uns["GMM_means"], rtol=.05)


def test_fit_explicit_cofactor_and_N(rainbow_raw):
    ad = rainbow_raw["A"].copy()
    fit_rainbow_GMM(ad, N=NPEAK, cofactor=50., random_state=0)
    assert ad.uns["GMM_asinh_cofactor"] == 50.
    np.testing.assert_allclose(ad.uns["GMM_means"][:, 0], synthetic.PEAKS, rtol=.03)


def test_fit_is_deterministic_with_int_seed(rainbow_raw):
    a, b = rainbow_raw["C"].copy(), rainbow_raw["C"].copy()
    fit_rainbow_GMM(a, random_state=3)
    fit_rainbow_GMM(b, random_state=3)
    np.testing.assert_array_equal(a.obs["GMM_label"], b.obs["GMM_label"])
    np.testing.assert_allclose(a.uns["GMM_means"], b.uns["GMM_means"])


def test_fit_custom_uniform_recovers_peaks(rainbow_raw):
    ad = rainbow_raw["B"].copy()
    fit_rainbow_GMM(ad, init_params="custom-uniform", random_state=0, n_init=1)
    expected = synthetic.PEAKS[:, None] * synthetic.GAINS.loc["B"].to_numpy()[None]
    np.testing.assert_allclose(ad.uns["GMM_means"], expected, rtol=.03)


def test_custom_uniform_precision_init_per_peak_R2_D3(rainbow_raw, monkeypatch):
    """R2-D3: the 'custom-uniform' precisions are initialised from the beads of one peak each (slices of the mean rank
    over channels), not from a mix of all peaks."""
    captured = {}
    real = gmm.GaussianMixture

    def spy(**kw):
        captured.update(kw)
        return real(**kw)
    monkeypatch.setattr(gmm, "GaussianMixture", spy)
    ad = rainbow_raw["A"].copy()
    fit_rainbow_GMM(ad, init_params="custom-uniform", random_state=0, n_init=1)
    prec = captured["precisions_init"]
    assert prec.shape == (NPEAK, len(FL), len(FL))
    # within-peak variance on the arcsinh scale is small (CV 6 %, var ~0.004-0.01); a mix of all peaks (the old bug:
    # slices of argsort indices) has the variance of the whole sample (~4). The middle slice also catches a few beads
    # of the neighbouring peaks, since the synthetic peak shares are only approximately equal.
    init_var = np.array([np.diag(np.linalg.inv(p)) for p in prec])
    total_var = np.var(np.arcsinh(ad[:, FL].layers["raw"] / ad.uns["GMM_asinh_cofactor"]), axis=0)
    assert (init_var < .1 * total_var[None]).all(), init_var
    assert (np.sort(init_var.max(axis=1))[:2] < .05).all(), init_var
    np.testing.assert_allclose(captured["weights_init"], np.full(NPEAK, 1 / NPEAK))


# ---------------------------------------------------------------- retries (R2-E7, R2b-G2-09)
class _StubGM:
    """Stand-in for sklearn's GaussianMixture with fixed labels; records the random_state of every fit."""
    calls = []
    assign = None

    def __init__(self, n_components, random_state=None, **kw):
        self.n = n_components
        _StubGM.calls.append(random_state)

    def fit(self, X):
        self.means_ = np.arange(self.n)[:, None] * np.ones((1, X.shape[1]))
        self.covariances_ = np.stack([np.eye(X.shape[1])] * self.n)
        return self

    def predict(self, X):
        return _StubGM.assign(X.shape[0])

    def score_samples(self, X):
        return np.zeros(X.shape[0])


def _run_stub(monkeypatch, ad, N, assign, **kw):
    _StubGM.calls, _StubGM.assign = [], staticmethod(assign)
    monkeypatch.setattr(gmm, "GaussianMixture", _StubGM)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fit_rainbow_GMM(ad, N=N, cofactor=100., **kw)
    msgs = [str(w.message) for w in rec if issubclass(w.category, GMMAssignmentWarning)]
    return list(_StubGM.calls), msgs


def test_retry_increments_int_seed_R2_E7(rainbow_raw, monkeypatch):
    """R2-E7: a failed fit is retried with random_state + 1 per retry (an int seed would repeat the identical fit),
    and the retry warning works without uns['name_uid']."""
    ad = rainbow_raw["A"].copy()
    del ad.uns["name_uid"]
    uneven = lambda n: np.where(np.arange(n) < int(.8 * n), 0, np.arange(n) % 3)
    calls, msgs = _run_stub(monkeypatch, ad, 3, uneven, random_state=7, Nretry=2)
    assert calls == [7, 8, 9]
    assert len(msgs) == 3 and "trying again" in msgs[0] and "Smallest peak class" in msgs[-1]


def test_retry_keeps_randomstate_instance_R2_E7(rainbow_raw, monkeypatch):
    ad = rainbow_raw["A"].copy()
    rs = np.random.RandomState(0)
    uneven = lambda n: np.where(np.arange(n) < int(.8 * n), 0, np.arange(n) % 3)
    calls, _ = _run_stub(monkeypatch, ad, 3, uneven, random_state=rs, Nretry=1)
    assert len(calls) == 2 and all(c is rs for c in calls)


def test_empty_component_triggers_retry_R2b_G2_09(rainbow_raw, monkeypatch):
    """R2b-G2-09: an empty GMM component counts as a failed fit (retries, then the final warning), also when the
    remaining shares are even; with Nretry=0 there is only the final warning."""
    ad = rainbow_raw["A"].copy()
    calls, msgs = _run_stub(monkeypatch, ad, 8, lambda n: np.arange(n) % 7, random_state=0, Nretry=2)
    assert len(calls) == 3 and len(msgs) == 3 and "share 0.00" in msgs[-1]
    calls, msgs = _run_stub(monkeypatch, ad, 3, lambda n: np.arange(n) % 2, random_state=0, Nretry=0)
    assert len(calls) == 1 and len(msgs) == 1


def test_even_fit_does_not_warn(rainbow_raw, monkeypatch):
    ad = rainbow_raw["A"].copy()
    calls, msgs = _run_stub(monkeypatch, ad, 3, lambda n: np.arange(n) % 3, random_state=0)
    assert len(calls) == 1 and msgs == []


# ---------------------------------------------------------------- order_channels (R1 a113432)
def test_order_channels_puts_channel_maxima_first(rainbow_raw):
    """R1-other (a113432): with order_channels, peak k is the component with the highest mean in order_channels[k];
    the other peaks follow in the default (dim to bright) order."""
    ad = rainbow_raw["B"].copy()
    fit_rainbow_GMM(ad, random_state=0, order_channels=["FL1-A"])
    assert ad.uns["GMM_order_ok"] is True
    expected = synthetic.PEAKS[[2, 0, 1], None] * synthetic.GAINS.loc["B"].to_numpy()[None]
    np.testing.assert_allclose(ad.uns["GMM_means"], expected, rtol=.03)
    true = _true_labels("B")
    assert (ad.obs["GMM_label"].to_numpy() == np.array([1, 2, 0])[true]).mean() > .99


def test_order_channels_not_distinct_falls_back_with_warning(rainbow_raw):
    ad = rainbow_raw["B"].copy()
    with pytest.warns(GMMAssignmentWarning, match="order_channels"):
        fit_rainbow_GMM(ad, random_state=0, order_channels=["FL1-A", "FL2-A"])   # both pick the brightest peak
    assert ad.uns["GMM_order_ok"] is False
    expected = synthetic.PEAKS[:, None] * synthetic.GAINS.loc["B"].to_numpy()[None]
    np.testing.assert_allclose(ad.uns["GMM_means"], expected, rtol=.03)


def test_order_channels_unknown_channel_raises(rainbow_raw):
    with pytest.raises(ValueError, match="order_channels"):
        fit_rainbow_GMM(rainbow_raw["A"].copy(), random_state=0, order_channels=["FSC-A"])


# ---------------------------------------------------------------- likelihood gate
def test_gate_returns_boolean_mask_and_keeps_adata(rainbow_fitted):
    ad = rainbow_fitted["A"]
    before = ad.obs.copy()
    mask = get_rainbow_GMM_gate(ad)
    assert isinstance(mask, pd.Series) and mask.dtype == bool
    assert list(mask.index) == list(ad.obs_names)
    pd.testing.assert_frame_equal(ad.obs, before)
    assert .85 < mask.mean() < 1.


def test_gate_drops_at_most_dropmaxpercent_per_peak(rainbow_fitted):
    ad = rainbow_fitted["C"]
    labels = ad.obs["GMM_label"].to_numpy()
    strict = get_rainbow_GMM_gate(ad, cutoff=100., dropmaxpercent=10)   # cutoff above every bead: capped
    dropped = pd.Series(~strict.to_numpy()).groupby(labels).mean()
    np.testing.assert_allclose(dropped, .10, atol=.01)
    lenient = get_rainbow_GMM_gate(ad, cutoff=-100.)                     # cutoff below every bead
    assert lenient.all()


def test_gate_cutoff_is_mean_plus_cutoff_std_of_clipped_llh(rainbow_fitted):
    ad = rainbow_fitted["A"]
    cut = _get_rainbow_GMM_cutoff(ad, cutoff=-2., clippercent=5, dropmaxpercent=100)
    for k in range(NPEAK):
        x = ad.obs["GMM_llh"].to_numpy()[ad.obs["GMM_label"].to_numpy() == k]
        x = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
        assert cut.loc[k] == pytest.approx(x.mean() - 2 * x.std())
    mask = get_rainbow_GMM_gate(ad, cutoff=-2., dropmaxpercent=100)
    expected = ad.obs["GMM_llh"].to_numpy() > cut.loc[ad.obs["GMM_label"]].to_numpy()   # strict
    np.testing.assert_array_equal(mask.to_numpy(), expected)


def test_gate_then_refit_workflow(rainbow_fitted):
    ad = rainbow_fitted["B"].copy()
    ad = ad[get_rainbow_GMM_gate(ad, cutoff=-2, dropmaxpercent=10)].copy()
    fit_rainbow_GMM(ad, random_state=0)
    expected = synthetic.PEAKS[:, None] * synthetic.GAINS.loc["B"].to_numpy()[None]
    np.testing.assert_allclose(ad.uns["GMM_means"], expected, rtol=.03)


# ---------------------------------------------------------------- plots
def test_max_separation_channel_is_valid_index(rainbow_fitted):
    i = _get_max_separation_channel(rainbow_fitted["B"])
    assert 0 <= i < len(FL)


def test_max_separation_channel_picks_best_separated():
    ad = synthetic.make_adata(np.zeros((3, 3)), FL, layer="raw")
    ad.uns["GMM_asinh_means"] = np.array([[1., 1., 1.], [2., 4., 2.], [3., 7., 3.]])
    ad.uns["GMM_asinh_stds"] = np.full((3, 3), .1)
    assert _get_max_separation_channel(ad) == 1


def test_plot_channel_histogram(rainbow_fitted, tmp_path):
    ad = rainbow_fitted["A"]
    fig, ax = plt.subplots()
    plot_rainbow_channel_GMM_single(ad, i=1, ax=ax)
    assert ax.get_xlabel() == FL[1] and ax.get_title() == ad.uns["name_uid"]
    path = tmp_path / "hist.png"
    plot_rainbow_channel_GMM_single(ad, savepath=str(path))
    assert path.exists()


def test_plot_GMM_llh_draws_cutoffs_on_ax_R3b_B5(rainbow_fitted):
    """R3b-B5: the cutoff lines (one per peak, at the gate cutoffs) are drawn on `ax`, not on the current axes."""
    ad = rainbow_fitted["A"]
    fig, (ax0, ax1) = plt.subplots(1, 2)
    plt.sca(ax1)
    plot_GMM_llh(ad, ax=ax0)
    vertical = lambda ax: [l.get_xdata()[0] for l in ax.lines if len(set(np.asarray(l.get_xdata(), float))) == 1]
    assert len(vertical(ax0)) == NPEAK and vertical(ax1) == []
    np.testing.assert_allclose(sorted(vertical(ax0)), sorted(_get_rainbow_GMM_cutoff(ad)))


def test_plot_GMM_llh_new_figure(rainbow_fitted):
    n = len(plt.get_fignums())
    plot_GMM_llh(rainbow_fitted["C"], cutoff=-3, plotminstd=2)
    assert len(plt.get_fignums()) == n + 1
