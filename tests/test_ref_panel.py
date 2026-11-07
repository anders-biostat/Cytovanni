"""cytovanni.ref.panel.PanelModule: spectra from a panel embedding, differentiable unmixing and its diagnostics.

Group-specific generator: `make_panelconfig` builds a `PanelConfiguration` with PCA data directly (offsets = the
synthetic spectra, a few small components per stain), without fitting single-stain controls.
"""
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.spill import PanelConfiguration
from cytovanni.ref import PanelModule
from cytovanni.torch import torch_invert_spectra, torch_apply_unmixing_inv

# components of the square panel (3 stains x 3 channels): zero in each stain's main channel (normalised spectra)
COMPONENTS_SQUARE = {
    "FL1": [[0., .05, -.02], [0., .01, .03]],
    "FL2": [[.04, 0., .03]],
    "FL3": [[.01, .03, 0.]],
}
USE_SQUARE = {"FL1_PC0": True, "FL1_PC1": False, "FL2_PC0": True, "FL3_PC0": True}


def make_panelconfig(offset=None, components=None, use=None, markers=None, n_measurements=4, seed=0):
    """PanelConfiguration with PCA data: `offset` (stain, channel) DataFrame, `components` stain -> list of vectors."""
    offset = synthetic.SPECTRA.copy() if offset is None else offset
    components = COMPONENTS_SQUARE if components is None else components
    use = USE_SQUARE if use is None else use
    stains = offset.index.to_numpy()
    markers = np.array(synthetic.MARKERS[:len(stains)] if markers is None else markers)
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


def make_spectral_panelconfig():
    """2 stains x 4 channels, one used component per stain."""
    ch = ["c0", "c1", "c2", "c3"]
    offset = pd.DataFrame([[1., .4, .1, .05], [.1, .3, 1., .5]], index=["A", "B"], columns=ch)
    comps = {"A": [[0., .05, .02, .01]], "B": [[.02, 0.03, 0., .04]]}
    return make_panelconfig(offset, comps, {"A_PC0": True, "B_PC0": True}, markers=["m1", "m2"])


@pytest.fixture
def pc():
    return make_panelconfig()


@pytest.fixture
def pm(pc):
    return PanelModule.from_panelconfig(pc)


def _events(n=50, seed=0):
    rng = np.random.default_rng(seed)
    ab = np.abs(rng.lognormal(8, 1, (n, 3)))
    return (ab @ synthetic.SPECTRA.to_numpy()).astype(np.float32), ab


def _adata(x, **uns):
    return synthetic.make_adata(x, synthetic.CHANNELS_FLUORESCENCE, **uns)


# ------------------------------------------------------------------ construction, forward
def test_from_panelconfig_names_and_shapes(pm, pc):
    assert list(pm.stains) == synthetic.STAINS and list(pm.channels) == synthetic.CHANNELS_FLUORESCENCE
    assert list(pm.components) == list(pc.pca_components.index)
    assert pm.param_offset.shape == (3, 3)
    assert pm.param_components.shape == (3, 3, 4)  # (stain, channel, component)
    assert pm.param_components_mask.tolist() == [1., 0., 1., 1.]
    assert not any(p.requires_grad for p in pm.parameters())
    # every component sits in its own stain's row only
    assert torch.allclose(pm.param_components[0, :, 1], torch.tensor([0., .01, .03]))
    assert pm.param_components[1:, :, :2].abs().sum() == 0


def test_spectral_panel_component_shape():
    """R2-B1: components are allocated (stain, channel, component); spectral panels (stains < channels) crashed."""
    pm = PanelModule.from_panelconfig(make_spectral_panelconfig())
    assert pm.param_components.shape == (2, 4, 2)
    S = pm(torch.zeros(2))
    assert S.shape == (2, 4)
    assert torch.allclose(S, torch.tensor(make_spectral_panelconfig().pca_offset.to_numpy(), dtype=torch.float32))


def test_forward_formula(pm, pc):
    e = torch.tensor([1., 2., -1., .5])
    S = pm(e, mask_unused=False)
    expected = pc.pca_offset.to_numpy().copy()
    for k, name in enumerate(pc.pca_components.index):
        expected[pm.stains.tolist().index(pc.pca_components_stain[name])] += pc.pca_components.loc[name].to_numpy() * e[k].item()
    assert torch.allclose(S, torch.tensor(np.clip(expected, 0, None), dtype=torch.float32), atol=1e-6)


def test_forward_masks_unused_components(pm):
    e = torch.tensor([0., 5., 0., 0.])  # only the unused FL1_PC1
    assert torch.allclose(pm(e), pm(torch.zeros(4)))
    assert not torch.allclose(pm(e, mask_unused=False), pm(torch.zeros(4)))


def test_forward_nonfinite_unused_embedding_stays_finite(pm):
    """R2d-D4: a NaN embedding of an unused component no longer spreads into the spectra (NaN * 0 = NaN)."""
    e = torch.tensor([0., float("nan"), 0., 0.])
    assert torch.isfinite(pm(e)).all()
    assert torch.isnan(pm(e, mask_unused=False)).any()


def test_forward_batched_embedding_axes(pm):
    e = torch.zeros(4, 5, 6)
    assert pm(e).shape == (5, 6, 3, 3)
    e2 = torch.randn(4, 7)
    S = pm(e2)
    assert S.shape == (7, 3, 3)
    assert torch.allclose(S[2], pm(e2[:, 2]))


def test_forward_clips_negative_entries_with_straight_through_gradient(pm):
    e = torch.tensor([-10., 0., 0., 0.], requires_grad=True)  # FL1 / FL2-A entry: .3 - .5 -> clipped
    S = pm(e)
    assert (S >= 0).all() and S[0, 1] == 0
    S[0, 1].backward()
    assert e.grad[0] == pytest.approx(.05)  # straight-through
    e2 = e.detach().clone().requires_grad_(True)
    pm(e2, clamp_differentiable=False)[0, 1].backward()
    assert e2.grad[0] == 0  # plain clamp


def test_apply_component_mask_broadcast(pm):
    """R2b-G2-23: inputs (component, k) with k > 1 are masked (failed before)."""
    out = pm.apply_component_mask(torch.ones(4, 3))
    assert out.tolist() == [[1.] * 3, [0.] * 3, [1.] * 3, [1.] * 3]
    assert pm.apply_component_mask(torch.ones(4)).tolist() == [1., 0., 1., 1.]


def test_requires_grad_keeps_parameters_fixed(pm):
    """R2-E1: requires_grad_ returns self and keeps the module non-trainable."""
    assert pm.requires_grad_(True) is pm
    assert not any(p.requires_grad for p in pm.parameters())


def test_export_roundtrip(pm, tmp_path):
    torch.save(pm.export(), tmp_path / "pm.pt")
    pm2 = PanelModule.from_exported(torch.load(tmp_path / "pm.pt", weights_only=False))
    e = torch.randn(4)
    assert torch.equal(pm2(e), pm(e))
    assert list(pm2.stains) == list(pm.stains) and list(pm2.components) == list(pm.components)


def test_device_property(pm, device):
    pm.to(device)
    assert pm.device.type == device
    x, _ = _events(5)
    out = pm.apply_ashunmx(torch.tensor(x), torch.zeros(4))  # inputs are moved to the module's device
    assert out.device.type == device


# ------------------------------------------------------------------ unmixing, Jacobians
def test_apply_ashunmx_formula_and_recovery(pm):
    x, ab = _events()
    scale = torch.tensor([2., 1., .5])
    out = pm.apply_ashunmx(torch.tensor(x), torch.zeros(4), stainscale=scale, arcsinh_cofactor=1000)
    assert out.shape == (50, 3)
    assert np.allclose(out.numpy(), np.arcsinh(ab * scale.numpy() / 1000), atol=1e-3)


def test_apply_ashunmx_one_embedding_per_event(pm):
    x, _ = _events(6)
    E = torch.randn(4, 6) * .5
    out = pm.apply_ashunmx(torch.tensor(x), E)
    for i in range(6):
        assert torch.allclose(out[i], pm.apply_ashunmx(torch.tensor(x[i]), E[:, i]), atol=1e-5)


@pytest.mark.parametrize("include_unused", [True, False])
def test_jacobian_matches_jacrev_reference(pm, include_unused):
    x, _ = _events(8)
    e = torch.tensor([.3, -.2, .5, .1])
    j = pm.get_jacobian_ashunmx(torch.tensor(x), e, include_unused=include_unused)
    jr = pm.get_jacobian_ashunmx_jacrev(torch.tensor(x), e, include_unused=include_unused)
    assert j.shape == (8, 3, 4)
    assert torch.allclose(j, jr, atol=1e-5)
    assert (j[..., 1].abs().sum() > 0) == include_unused


def test_jacobian_with_stainscale_matches_autograd(pm):
    x, _ = _events(4)
    e = torch.tensor([.3, -.2, .5, .1])
    scale = torch.tensor([2., 1., .5])
    j = pm.get_jacobian_ashunmx(torch.tensor(x), e, stainscale=scale)
    ref = torch.func.jacrev(lambda emb: pm.apply_ashunmx(torch.tensor(x), emb, stainscale=scale))(e)
    assert torch.allclose(j, ref, atol=1e-5)


# ------------------------------------------------------------------ unmixing resistance
def test_resistance_defaults_are_gradient_step_sum_norm():
    """R3b-U2: defaults single_step=False, gradstep_maxnorm=False (the author's settings)."""
    for f in (PanelModule.get_unmixing_resistance, PanelModule.get_unmixing_resistance_fromad):
        sig = inspect.signature(f).parameters
        assert sig["single_step"].default is False and sig["gradstep_maxnorm"].default is False
        assert sig["include_unused"].default is False


@pytest.mark.parametrize("single_step,maxnorm", [(False, False), (False, True), (True, False)])
def test_resistance_outputs(pm, single_step, maxnorm):
    x, _ = _events(30)
    e = np.array([.3, 0., -.4, .2])
    val, val_std, R = pm.get_unmixing_resistance(x, e, single_step=single_step, gradstep_maxnorm=maxnorm)
    assert val.shape == val_std.shape == R.shape == (30, 3)
    ref = pm.apply_ashunmx(torch.tensor(x), torch.tensor(e, dtype=torch.float32)).numpy()
    assert np.allclose(val, ref, atol=1e-5)
    assert (val_std > 0).all() and np.isfinite(val_std).all()
    assert np.allclose(R, np.arcsinh(np.abs(val) / val_std / 20), atol=1e-5)


def test_resistance_error_grows_with_step(pm):
    x, _ = _events(20)
    e = np.zeros(4)
    s1 = pm.get_unmixing_resistance(x, e, embedding_step=.5)[1]
    s2 = pm.get_unmixing_resistance(x, e, embedding_step=2.)[1]
    assert (s2 > s1).all()


def test_resistance_chunks_do_not_change_result(pm):
    x, _ = _events(25)
    e = np.array([.1, 0., .2, -.1])
    a = pm.get_unmixing_resistance(x, e)
    b = pm.get_unmixing_resistance(x, e, batch_size=7)
    for u, v in zip(a, b):
        assert np.allclose(u, v, atol=1e-6)


def test_resistance_single_step_adds_step():
    """R2-D2: with single_step the chosen component is shifted by +-step (it was set to +-step)."""
    pm = PanelModule.from_panelconfig(make_panelconfig(use={k: True for k in USE_SQUARE}))
    x, _ = _events(10)
    e = np.array([3., -2., 1.5, .5])
    rec = []
    orig = pm.apply_ashunmx

    def spy(x, embedding, **kw):
        rec.append(embedding.detach().clone())
        return orig(x, embedding, **kw)

    pm.apply_ashunmx = spy
    pm.get_unmixing_resistance(x, e, embedding_step=1., single_step=True)
    m = [r for r in rec if r.dim() == 4][0]  # (component, events, stain, 2)
    diff = (m - torch.tensor(e, dtype=torch.float32)[:, None, None, None])
    # exactly one component changes per (event, stain, sign), by +1 / -1
    assert torch.allclose(diff.abs().sum(0), torch.ones(diff.shape[1:]))
    assert torch.allclose(diff.sum(0)[..., 0], torch.ones(diff.shape[1:3]))
    assert torch.allclose(diff.sum(0)[..., 1], -torch.ones(diff.shape[1:3]))


@pytest.mark.parametrize("single_step", [True, False])
def test_resistance_zero_events(pm, single_step):
    """R2b-G2-20: all-zero events have error 0 and R 0 (NaN before)."""
    x, _ = _events(5)
    x[2] = 0.
    val, val_std, R = pm.get_unmixing_resistance(x, np.zeros(4), single_step=single_step)
    assert np.isfinite(val_std).all() and np.isfinite(R[[0, 1, 3, 4]]).all()
    assert (val[2] == 0).all() and (val_std[2] == 0).all() and (R[2] == 0).all()


def test_resistance_stainscale(pm):
    x, ab = _events(10)
    val = pm.get_unmixing_resistance(x, np.zeros(4), stainscale_np=np.array([2., 1., 1.]), arcsinh_cofactor=1000)[0]
    assert np.allclose(val[:, 0], np.arcsinh(2 * ab[:, 0] / 1000), atol=1e-3)


# ------------------------------------------------------------------ *_fromad diagnostics
def _adata_with_factors(x, channel_factor, stain_factor):
    ad = _adata(x)
    ad.var["sfm_calibration_factor"] = channel_factor
    ad.varm["sfm_stain_factor"] = np.tile(np.asarray(stain_factor, dtype=float), (ad.n_vars, 1))
    return ad


def test_resistance_fromad_applies_stored_factors(pm):
    """R3b-U3: channel factors (var) and stain factors (varm) of a fitted SpectralFitTransformer are applied."""
    x, _ = _events(15)
    e = np.array([.2, 0., -.3, .1])
    cf, sf = np.array([1.2, .9, 1.1]), np.array([1.5, 1., .7])
    ad = _adata_with_factors(x, cf, sf)
    R = pm.get_unmixing_resistance_fromad(ad, e)
    R_manual = pm.get_unmixing_resistance(x * cf[None], e, stainscale_np=sf)[-1]
    assert isinstance(R, pd.DataFrame) and list(R.index) == list(ad.obs.index)
    assert list(R.columns) == ["FL1 | CD1", "FL2 | CD2", "FL3 | CD3"]
    assert np.allclose(R.to_numpy(), R_manual, atol=1e-5)
    R_raw = pm.get_unmixing_resistance_fromad(ad, e, apply_factors=False, label_marker=False)
    assert list(R_raw.columns) == synthetic.STAINS
    assert np.allclose(R_raw.to_numpy(), pm.get_unmixing_resistance(x, e)[-1], atol=1e-5)
    assert not np.allclose(R_raw.to_numpy(), R.to_numpy())


def test_resistance_fromad_without_stored_factors_equals_plain(pm):
    x, _ = _events(10)
    R = pm.get_unmixing_resistance_fromad(_adata(x), np.zeros(4))
    assert np.allclose(R.to_numpy(), pm.get_unmixing_resistance(x, np.zeros(4))[-1], atol=1e-6)


def test_fromad_nan_channel_factor_raises(pm):
    x, _ = _events(5)
    ad = _adata_with_factors(x, [1., np.nan, 1.], [1., 1., 1.])
    with pytest.raises(ValueError):
        pm.get_unmixing_resistance_fromad(ad, np.zeros(4))


def test_fromad_wrong_stain_factor_count_raises(pm):
    x, _ = _events(5)
    ad = _adata(x)
    ad.varm["sfm_stain_factor"] = np.ones((3, 2))
    with pytest.raises(ValueError):
        pm.get_grouped_jac_fromad(ad, np.zeros(4))


def test_fromad_uses_only_panel_channels_in_panel_order(pm):
    x, _ = _events(8)
    extra = np.hstack([np.ones((8, 1)) * 1e5, x[:, ::-1]])
    ad = synthetic.make_adata(extra, ["FSC-A"] + synthetic.CHANNELS_FLUORESCENCE[::-1])
    R = pm.get_unmixing_resistance_fromad(ad, np.zeros(4))
    assert np.allclose(R.to_numpy(), pm.get_unmixing_resistance(x, np.zeros(4))[-1], atol=1e-6)


def test_grouped_jac_fromad_is_mean_jacobian(pm):
    x, _ = _events(12)
    e = np.array([.2, .1, -.3, .1])
    cf, sf = np.array([1.2, .9, 1.1]), np.array([1.5, 1., .7])
    ad = _adata_with_factors(x, cf, sf)
    jac = pm.get_grouped_jac_fromad(ad, e, batch_size=5)
    ref = pm.get_jacobian_ashunmx(torch.tensor(x * cf[None], dtype=torch.float32), torch.tensor(e, dtype=torch.float32),
                                  stainscale=torch.tensor(sf, dtype=torch.float32)[None], include_unused=True).mean(0)
    assert list(jac.index) == list(pm.components) and list(jac.columns) == list(pm.panelconfig.stain_marker_name)
    assert np.allclose(jac.to_numpy(), ref.numpy().T, atol=1e-6)
    assert (jac.loc["FL1_PC1"].abs() > 0).any()  # unused components are included
    # an explicit stain scale overrides the stored one
    jac1 = pm.get_grouped_jac_fromad(ad, e, stainscale_np=np.ones(3), label_marker=False)
    assert list(jac1.columns) == synthetic.STAINS and not np.allclose(jac1.to_numpy(), jac.to_numpy())


def test_grouped_jac_single_fromad(pm):
    x, _ = _events(12)
    e = np.array([.2, .1, -.3, .1])
    ad = _adata_with_factors(x, [1.2, .9, 1.1], [1.5, 1., .7])
    ad.uns["panel_embedding"] = e
    df, style = pm.get_grouped_jac_single_fromad(ad, "FL2 | CD2", include=3)
    assert list(df.columns) == ["abs. Jacobian", "Jacobian", "in Panel"] and len(df) == 3
    assert df["in Panel"].dtype == bool
    assert (np.diff(df["abs. Jacobian"].to_numpy(dtype=float)) <= 0).all()
    full = pm.get_grouped_jac_fromad(ad, e)["FL2 | CD2"]
    assert np.allclose(df["Jacobian"].to_numpy(dtype=float), np.round(full.loc[df.index], 3))
    assert df.loc[df.index, "in Panel"].tolist() == pm.panelconfig.pca_components_use.loc[df.index].tolist()
    pytest.importorskip("jinja2")
    html = style(df).to_html()
    assert "lightgreen" in html
    assert len(pm.get_grouped_jac_single_fromad(ad, "FL2 | CD2", include=None)[0]) == 4


def test_spectra_fromembedding(pm):
    e = np.array([0., 5., 0., 0.])
    s = pm.get_spectra_fromembedding(e)
    assert list(s.index) == list(pm.panelconfig.stain_marker_name) and list(s.columns) == list(pm.channels)
    assert np.allclose(s.to_numpy(), synthetic.SPECTRA.to_numpy(), atol=1e-6)
    assert list(pm.get_spectra_fromembedding(e, label_marker=False).index) == synthetic.STAINS


def test_spectra_fromembedding_include_unused():
    """R2b-G2-22: include_unused=True returns the unmasked spectra (it was ignored)."""
    pm = PanelModule.from_panelconfig(make_panelconfig())
    e = np.array([0., 5., 0., 0.])
    s = pm.get_spectra_fromembedding(e, include_unused=True)
    assert np.allclose(s.iloc[0].to_numpy(), synthetic.SPECTRA.iloc[0].to_numpy() + 5 * np.array([0., .01, .03]), atol=1e-6)


def test_compmean_errorbars(pm):
    x, _ = _events(40)
    e = np.array([.2, 0., -.3, .1])
    df = pm.get_compmean_errorbars(x, e)
    assert list(df.columns) == ["mean", "std", "z"] and list(df.index) == list(pm.panelconfig.stain_marker_name)
    vals = pm.apply_ashunmx(torch.tensor(x), torch.tensor(e, dtype=torch.float32)).numpy()
    assert np.allclose(df["mean"], vals.mean(0), atol=1e-5)
    assert (df["std"] > 0).all() and np.allclose(df["z"], df["mean"] / df["std"])
    med = pm.get_compmean_errorbars(x, e, use_median=True, label_marker=False)
    # torch's median: the lower of the two middle values for an even count (numpy averages them)
    lower_median = np.sort(vals, 0)[(len(vals) - 1) // 2]
    assert np.allclose(med["mean"], lower_median, atol=1e-5) and list(med.index) == synthetic.STAINS
    sc = pm.get_compmean_errorbars(x, e, stainscale_np=np.array([2., 1., 1.]))
    assert sc["mean"].iloc[0] > df["mean"].iloc[0] and np.isclose(sc["mean"].iloc[1], df["mean"].iloc[1])


def test_compmean_errorbars_step_is_added(pm):
    """The perturbation is +-step around the embedding (as R2-D2 for the per-event version)."""
    x, _ = _events(20)
    e = np.array([.5, 0., -.5, .2])
    a = pm.get_compmean_errorbars(x, e, embedding_step=1.)
    b = pm.get_compmean_errorbars(x, e, embedding_step=1e-3)
    assert (b["std"] < a["std"] / 100).all()


def test_clustermeans_withz(pm):
    x, _ = _events(60)
    ad = _adata(x)
    ad.obs["cl"] = np.where(x[:, 0] > np.median(x[:, 0]), "hi", "lo")
    means, zs = pm.get_clustermeans_withz(np.zeros(4), ad, "cl", cluster_reorder=False, include_unused=False)
    assert list(means.index) == ["hi", "lo"] and list(means.columns) == list(pm.panelconfig.stain_marker_name)
    assert means.shape == zs.shape == (2, 3)
    ref = pm.get_compmean_errorbars(x[ad.obs["cl"].to_numpy() == "lo"], np.zeros(4))
    assert np.allclose(means.loc["lo"], ref["mean"], atol=1e-5) and np.allclose(zs.loc["lo"], ref["z"], rtol=1e-4)
    assert means.loc["hi"].iloc[0] > means.loc["lo"].iloc[0]
    m2, z2 = pm.get_clustermeans_withz(np.zeros(4), ad, "cl", cluster_reorder=True, include_unused=False)
    assert set(m2.index) == {"hi", "lo"}
    assert np.allclose(m2.loc[means.index, means.columns], means) and np.allclose(z2.loc[zs.index, zs.columns], zs)


def test_clustermeans_withz_applies_factors(pm):
    """R3b-U3: get_clustermeans_withz applies the stored factors (default apply_factors=True)."""
    x, _ = _events(30)
    ad = _adata_with_factors(x, [1.2, .9, 1.1], [2., 1., 1.])
    ad.obs["cl"] = "a"
    means, _ = pm.get_clustermeans_withz(np.zeros(4), ad, "cl", cluster_reorder=False)
    ref = pm.get_compmean_errorbars(x * np.array([1.2, .9, 1.1])[None], np.zeros(4), include_unused=True,
                                    stainscale_np=np.array([2., 1., 1.]))
    assert np.allclose(means.loc["a"], ref["mean"], atol=1e-5)
    raw, _ = pm.get_clustermeans_withz(np.zeros(4), ad, "cl", cluster_reorder=False, apply_factors=False)
    assert not np.allclose(raw.to_numpy(), means.to_numpy())


def test_spectral_panel_unmixing_recovers_abundances():
    pc = make_spectral_panelconfig()
    pm = PanelModule.from_panelconfig(pc)
    rng = np.random.default_rng(0)
    ab = rng.lognormal(8, 1, (20, 2))
    e = torch.tensor([.5, -.5])
    S = pm(e).numpy()
    x = (ab @ S).astype(np.float32)
    out = pm.apply_ashunmx(torch.tensor(x), e, arcsinh_cofactor=1.)
    assert np.allclose(np.sinh(out.numpy()), ab, rtol=1e-3)
    val, val_std, R = pm.get_unmixing_resistance(x, e.numpy())
    assert val.shape == (20, 2) and np.isfinite(R).all()
