"""Tests of cytovanni.utils.unmx: arcsinh transform and linear unmixing with DataFrames (spectra (stain, channel))."""
import inspect

import numpy as np
import pandas as pd
import pytest

import synthetic
from cytovanni.exceptions import SpilloverInversionException
from cytovanni.utils import (add_adata_unmixed, apply_arcsinh, apply_unmixing_inv, apply_unmixing_lstsq,
                             invert_spectra, unmixed_cluster_means)


@pytest.fixture
def spectral():
    """Spectral case: 2 stains, 4 channels, (stain, channel)."""
    return pd.DataFrame([[1., .3, .1, .05], [.1, .2, 1., .4]], index=["s1", "s2"], columns=["c1", "c2", "c3", "c4"])


def mixed(spectra, rng, n=50):
    """Mixed data (events, channel) with known abundances (events, stain)."""
    ab = pd.DataFrame(rng.lognormal(8, 1, (n, len(spectra))), columns=spectra.index)
    return ab @ spectra, ab


# ------------------------------------------------------------------ apply_arcsinh
def test_apply_arcsinh_values_and_type():
    x = np.array([-1500., 0., 1500., 15000.])
    assert np.allclose(apply_arcsinh(x), np.arcsinh(x / 1500))
    assert np.allclose(apply_arcsinh(x, cofactor=150), np.arcsinh(x / 150))
    df = pd.DataFrame({"a": x}, index=list("wxyz"))
    out = apply_arcsinh(df)
    assert isinstance(out, pd.DataFrame) and out.index.equals(df.index) and list(out.columns) == ["a"]


# ------------------------------------------------------------------ invert_spectra
def test_invert_spectra_square_is_exact_inverse():
    s = synthetic.SPECTRA
    inv = invert_spectra(s)
    assert list(inv.index) == list(s.columns) and list(inv.columns) == list(s.index)  # (channel, stain)
    assert np.allclose(s.to_numpy() @ inv.to_numpy(), np.eye(3))


def test_invert_spectra_spectral_is_right_pseudoinverse(spectral):
    inv = invert_spectra(spectral)
    S = spectral.to_numpy()
    assert inv.shape == (4, 2)
    assert np.allclose(S @ inv.to_numpy(), np.eye(2))
    assert np.allclose(inv.to_numpy(), S.T @ np.linalg.inv(S @ S.T))
    assert np.allclose(invert_spectra(spectral, use_pinv=True).to_numpy(), inv.to_numpy())


def test_invert_spectra_fewer_channels_than_stains(spectral):
    with pytest.raises(SpilloverInversionException):
        invert_spectra(spectral.T)
    inv = invert_spectra(spectral.T, use_pinv=True)  # no error, minimum-norm solution
    assert inv.shape == (2, 4)


def test_invert_spectra_requires_dataframe():
    with pytest.raises(ValueError):
        invert_spectra(np.eye(3))


# ------------------------------------------------------------------ apply_unmixing_inv / _lstsq
def test_unmixing_recovers_abundances_with_shuffled_columns(spectral, rng):
    x, ab = mixed(spectral, rng)
    x["extra"] = 1.
    x = x[["c3", "extra", "c1", "c4", "c2"]]  # matched by name, extra columns ignored
    u_inv = apply_unmixing_inv(invert_spectra(spectral), x)
    u_lsq = apply_unmixing_lstsq(spectral, x)
    for u in (u_inv, u_lsq):
        assert list(u.columns) == ["s1", "s2"] and u.index.equals(x.index)
        assert np.allclose(u.to_numpy(), ab.to_numpy(), rtol=1e-8)


def test_apply_unmixing_inv_type_checks(spectral):
    """R2-E2: a non-DataFrame inverse raises ValueError (was a NameError in the message)."""
    x = pd.DataFrame(np.ones((2, 4)), columns=spectral.columns)
    with pytest.raises(ValueError):
        apply_unmixing_inv(invert_spectra(spectral).to_numpy(), x)
    with pytest.raises(ValueError):
        apply_unmixing_inv(invert_spectra(spectral), x.to_numpy())


def test_apply_unmixing_lstsq_type_checks(spectral):
    x = pd.DataFrame(np.ones((2, 4)), columns=spectral.columns)
    with pytest.raises(ValueError):
        apply_unmixing_lstsq(spectral.to_numpy(), x)
    with pytest.raises(ValueError):
        apply_unmixing_lstsq(spectral, x.to_numpy())


# ------------------------------------------------------------------ add_adata_unmixed
def test_add_adata_unmixed_writes_obsm():
    adata = synthetic.cell_adata("A", n=200)
    assert add_adata_unmixed(adata, synthetic.SPECTRA, arcsinh_cofactor=1000) is None
    expected = adata.to_df(layer="calibrated") @ invert_spectra(synthetic.SPECTRA)
    u = adata.obsm["unmx"]
    assert list(u.columns) == synthetic.STAINS and u.index.equals(adata.obs_names)
    assert np.allclose(u.to_numpy(), expected.to_numpy(), rtol=1e-5, atol=1e-3)
    assert np.allclose(adata.obsm["unmx_arcsinh"].to_numpy(), np.arcsinh(u.to_numpy() / 1000))


def test_add_adata_unmixed_no_arcsinh_and_key():
    adata = synthetic.cell_adata("A", n=50)
    add_adata_unmixed(adata, synthetic.SPECTRA, addkey="u2", add_arcsinh=False)
    assert "u2" in adata.obsm and "u2_arcsinh" not in adata.obsm


def test_add_adata_unmixed_honours_layer():
    """R2-K5: the data come from `layer` (was always 'calibrated'); the default stays 'calibrated'."""
    assert inspect.signature(add_adata_unmixed).parameters["layer"].default == "calibrated"
    adata = synthetic.cell_adata("A", n=50)
    adata.layers["raw"] = adata.layers["calibrated"] * 2
    add_adata_unmixed(adata, synthetic.SPECTRA, addkey="cal")
    add_adata_unmixed(adata, synthetic.SPECTRA, addkey="raw", layer="raw")
    assert np.allclose(adata.obsm["raw"].to_numpy(), 2 * adata.obsm["cal"].to_numpy(), rtol=1e-5, atol=1e-3)


def test_add_adata_unmixed_uses_channels_by_name():
    adata = synthetic.cell_adata("A", n=50)
    reordered = adata[:, ["FL3-A", "FL1-A", "FL2-A"]].copy()
    add_adata_unmixed(adata, synthetic.SPECTRA)
    add_adata_unmixed(reordered, synthetic.SPECTRA)
    assert np.allclose(adata.obsm["unmx"].to_numpy(), reordered.obsm["unmx"].to_numpy(), rtol=1e-5, atol=1e-3)


def test_add_adata_unmixed_too_few_channels():
    adata = synthetic.make_adata(np.ones((5, 2)), ["c1", "c2"])
    spectra = pd.DataFrame(np.eye(3)[:, :2], index=["a", "b", "c"], columns=["c1", "c2"])
    with pytest.raises(SpilloverInversionException):
        add_adata_unmixed(adata, spectra)


# ------------------------------------------------------------------ unmixed_cluster_means
def test_unmixed_cluster_means():
    u = pd.DataFrame({"a": [1., 2, 3, 4, 5], "b": [0., 0, 10, 10, 10]}, index=list("vwxyz"))
    m = unmixed_cluster_means(u, ["y", "x", "y", "x", "x"])
    assert list(m.index) == ["x", "y"] and list(m.columns) == ["a", "b"]
    assert np.allclose(m.loc["x"], [(2 + 4 + 5) / 3, 20 / 3]) and np.allclose(m.loc["y"], [2, 5])
