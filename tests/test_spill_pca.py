"""Tests of cytovanni.spill.pca.SingleSpectrumPCAFitter."""
import warnings

import numpy as np
import pandas as pd
import pytest

from cytovanni.spill.pca import SingleSpectrumPCAFitter

CH = [f"c{i}" for i in range(6)]
MEAN = np.array([1., .4, .2, .1, .05, .02])
DIRS = np.array([[0., .3, -.2, .1, 0., .05], [0., -.1, .1, .3, .2, 0.]])  # two true variation directions


def spectra_with_dirs(n=20, seed=0, scales=(.2, .05), noise=0.):
    """n spectra = MEAN + latent @ DIRS (+ noise), the first channel stays exactly 1 (normalised)."""
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, 2)) * np.asarray(scales)[None]
    x = MEAN[None] + lat @ DIRS + noise * rng.normal(size=(n, len(CH))) * (np.arange(len(CH)) > 0)[None]
    return pd.DataFrame(x, columns=CH, index=[f"m{i}" for i in range(n)])


def test_fit_attributes_shapes_and_reconstruction():
    sp = spectra_with_dirs(noise=1e-3)
    f = SingleSpectrumPCAFitter(sp, N=4, name="FL1")
    assert f.N == 4 and not f.is_dummy
    assert list(f.channels) == CH and list(f.obs) == list(sp.index)
    assert f.embedding.shape == (20, 4) and f.components.shape == (4, 6)
    assert f.spectra_means.shape == (1, 6)
    assert np.allclose(np.linalg.norm(f.components, axis=1), 1.)
    assert f.explained_variance.shape == (4,) and f.explained_variance_ratio.shape == (4,)
    assert f.explained_variance_bychannel.shape == (4, 6) and f.explained_variance_bychannel_ratio.shape == (4, 6)
    assert f.variance_bychannel.shape == (6,) and f.component_max_effect.shape == (4,)
    assert np.allclose(f.spectra_means[0], sp.mean().to_numpy())
    # spectra ~ mean + embedding @ components
    assert np.allclose(f.get_smoothed(4).to_numpy(), f.spectra_means + f.embedding @ f.components)
    assert np.allclose(f.get_smoothed(0).to_numpy(), np.repeat(f.spectra_means, 20, 0))
    assert np.allclose(f.get_real().to_numpy(), sp.to_numpy())


def test_variances_ddof1_and_totals():
    sp = spectra_with_dirs(noise=1e-3)
    f = SingleSpectrumPCAFitter(sp, N=5)
    var = np.var(sp.to_numpy(), axis=0, ddof=1)
    assert f.total_variance == pytest.approx(var.sum())
    assert np.allclose(f.variance_bychannel, var)
    assert f.max_channel_variance == pytest.approx(var.max())
    assert np.allclose(f.explained_variance, np.var(f.embedding, axis=0, ddof=1))
    assert np.allclose(f.explained_variance_ratio, f.explained_variance / var.sum())
    # constant channel c0 -> NaN ratio
    assert np.isnan(f.explained_variance_bychannel_ratio[:, 0]).all()
    assert np.allclose(f.component_max_effect, np.abs(f.embedding[..., None] * f.components[None]).max((0, 2)))


def test_recovers_variation_subspace():
    """C: two true directions -> the first two components span them; their variance dominates."""
    sp = spectra_with_dirs(n=40, noise=1e-4)
    f = SingleSpectrumPCAFitter(sp, N=5)
    assert f.explained_variance_ratio[:2].sum() > .999
    q, _ = np.linalg.qr(DIRS.T)
    proj = f.components[:2] @ q  # rows: components in the basis of the true subspace
    assert np.allclose(np.linalg.norm(proj, axis=1), 1., atol=1e-3)
    assert np.abs(f.get_smoothed(2).to_numpy() - sp.to_numpy()).max() < 1e-3


def test_rows_with_nan_are_dropped():
    sp = spectra_with_dirs(n=10, noise=1e-3)
    sp.iloc[3, 2] = np.nan
    f = SingleSpectrumPCAFitter(sp)
    assert "m3" not in f.obs and len(f.obs) == 9 and f.embedding.shape[0] == 9


def test_N_capped_by_channels():
    """R2-C14: more spectra than channels, fewer channels than N."""
    sp = pd.DataFrame(np.random.default_rng(0).random((10, 2)), columns=["c1", "c2"])
    f = SingleSpectrumPCAFitter(sp)
    assert f.N == 2 and f.embedding.shape == (10, 2)


@pytest.mark.parametrize("n, expected", [(3, 2), (2, 1), (6, 5), (7, 5)])
def test_N_capped_by_spectra_minus_one(n, expected):
    """R2b-G2-15: n spectra span n-1 dimensions; no zero-variance last component."""
    sp = pd.DataFrame(np.random.default_rng(1).random((n, 8)), columns=[f"c{i}" for i in range(8)])
    f = SingleSpectrumPCAFitter(sp)
    assert f.N == expected
    assert (f.explained_variance > 1e-8).all()
    assert np.isfinite(f.embedding / f.embedding.std(0)).all()


def test_N_capped_by_numerical_rank():
    """R2d-D4: normalised 3-channel spectra have rank <= 2 -> at most 2 components, none with zero variance."""
    rng = np.random.default_rng(12)
    sp = pd.DataFrame(np.column_stack([np.ones(8), .12 + .02 * rng.normal(size=8), .1 + .02 * rng.normal(size=8)]),
                      columns=["c1", "c2", "c3"])
    f = SingleSpectrumPCAFitter(sp, N=5)
    assert f.N == 2
    assert (np.nanstd(f.embedding, axis=0) > 0).all()
    # exactly one variation direction -> one component
    lat = rng.normal(size=(8, 1))
    sp1 = pd.DataFrame(MEAN[None] + lat * DIRS[:1], columns=CH)
    assert SingleSpectrumPCAFitter(sp1).N == 1


@pytest.mark.parametrize("x, rank", [(np.zeros((3, 4)), 0), (np.zeros((0, 4)), 0), (np.eye(3), 3),
                                     (np.outer([1., 2.], [1., 0., 1.]), 1)])
def test_get_numerical_rank(x, rank):
    assert SingleSpectrumPCAFitter.get_numerical_rank(x) == rank


def test_get_numerical_rank_rtol():
    x = np.diag([1., 1e-6])
    assert SingleSpectrumPCAFitter.get_numerical_rank(x) == 1   # s**2 = 1e-12 <= 1e-10 * 1
    assert SingleSpectrumPCAFitter.get_numerical_rank(x, rtol=1e-13) == 2


@pytest.mark.parametrize("sp", [pd.DataFrame([[1., .2, .1]], columns=["a", "b", "c"]),
                                pd.DataFrame([[1., .2, .1]] * 4, columns=["a", "b", "c"])],
                         ids=["one_spectrum", "identical_spectra"])
def test_dummy_fit(sp):
    f = SingleSpectrumPCAFitter(sp)
    assert f.is_dummy and f.N == 0
    assert f.embedding.shape == (len(sp), 0) and f.components.shape == (0, 3)
    assert f.total_variance == 0. and f.explained_variance.shape == (0,)
    assert np.allclose(f.get_smoothed(0).to_numpy(), sp.to_numpy())
    f.plot_fullscatter()  # empty plot
    f.add_explained_share_above_cutoff()
    assert f.total_SAC == 0 and f.explained_SAC.shape == (0,)


def test_explained_share_above_cutoff():
    sp = spectra_with_dirs(n=30, noise=1e-4)
    f = SingleSpectrumPCAFitter(sp, N=3)
    f.add_explained_share_above_cutoff(cutoff=.01)
    sac = [(np.abs(f.get_real() - f.get_smoothed(i)) > .01).mean().sum() for i in range(4)]
    assert f.total_SAC == pytest.approx(sac[0])
    assert np.allclose(f.explained_SAC, np.array(sac[:-1]) - np.array(sac[1:]))
    assert f.explained_SAC.shape == (3,)
    assert f.explained_SAC[:2].sum() == pytest.approx(sac[0], abs=1e-12)  # two components explain everything


def test_plots_run():
    f = SingleSpectrumPCAFitter(spectra_with_dirs(noise=1e-3), N=3, name="FL1")
    f.plot_fullscatter()
    f.plot_expvar()
