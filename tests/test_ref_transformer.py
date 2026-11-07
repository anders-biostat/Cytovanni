"""Tests of cytovanni.ref.model_transformer: ScaleFactor, DummyTransformer, SpectralFitTransformer, ScalingTransformer."""
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.exceptions import OverlapStandardisationException, OverlapStandardisationWarning
from cytovanni.ref import (DummyTransformer, FixedPanelModule, MultiOrdinalEncoder, PanelEmbeddingModule,
                           ScalingTransformer, SpectralFitTransformer, UnmxErrorExclusionMultiplier)
from cytovanni.ref.model_transformer import ScaleFactor, pick_panel_cls, pick_transformer_cls
from cytovanni.spill import PanelConfiguration
from cytovanni.utils import invert_spectra

CH = synthetic.CHANNELS_FLUORESCENCE
ST = synthetic.STAINS
SP = synthetic.SPECTRA
# PCA components of the test panel (component, channel): FL1 spills more or less into FL2, FL2 into FL3; the FL3
# component exists but is not used by the panel
COMPONENTS = pd.DataFrame([[0., .1, 0.], [0., 0., .1], [.05, 0., 0.]], index=["FL1_PC0", "FL2_PC0", "FL3_PC0"], columns=CH)
COMPONENTS_USE = pd.Series([True, True, False], index=COMPONENTS.index)


def make_panelconfig(folder):
    """PanelConfiguration of the simulated panel with hand-made PCA data (offset = SPECTRA)."""
    path = folder / "panel.csv"
    pd.DataFrame({"stain": ST, "marker": synthetic.MARKERS, "components": ["0", "0", ""]}).to_csv(path, index=False)
    pc = PanelConfiguration.from_csv(str(path), "sim")
    pc.pca_components = COMPONENTS.copy()
    pc.pca_offset = SP.copy()
    pc.pca_components_stain = pd.Series(ST, index=COMPONENTS.index)
    pc.pca_components_use = COMPONENTS_USE.copy()
    pc.pca_explained_variance = pd.Series([.02, .01, .005], index=COMPONENTS.index)
    pc.pca_embedding = pd.DataFrame([[1., -1., 0.]], index=["comp_run"], columns=COMPONENTS.index)
    pc.channels = CH
    pc.has_PCA_data = True
    return pc


def spectra_of(embedding):
    """Spectra (stain, channel) of the test panel for an embedding (component,), unused components masked."""
    e = np.asarray(embedding, dtype=float) * COMPONENTS_USE.to_numpy()
    sp = SP.to_numpy().copy()
    for k, comp in enumerate(COMPONENTS.index):
        sp[ST.index(comp.split("_")[0])] += e[k] * COMPONENTS.loc[comp].to_numpy()
    return np.clip(sp, 0, None)


def moe_abc():
    return MultiOrdinalEncoder.from_data(pd.DataFrame({"batch": synthetic.BATCHES}), ["batch"])


def adata_of(batch, n=5, seed=0, channels=CH):
    return synthetic.make_adata(np.random.default_rng(seed).uniform(1e3, 1e4, (n, len(channels))), channels, batch=batch)


@pytest.fixture
def panelconfig(tmp_path):
    return make_panelconfig(tmp_path)


def spectral_transformer(pc, moe=None, **kw):
    """SpectralFitTransformer of the test panel with batch A anchored."""
    moe = moe_abc() if moe is None else moe
    pem = PanelEmbeddingModule.from_data(pc, moe, pd.Series("batch", index=COMPONENTS.index), anchor_bidx={"batch": [0]})
    kw = dict(dict(scale_stains=True, scale_channels=True, anchor_bidx={"batch": [0]}), **kw)
    return SpectralFitTransformer.from_data(moe, pem, **kw)


def set_params(tr, emb=None, stain=None, channel=None):
    """Set the trainable parts of a transformer (values per batch, before masks)."""
    with torch.no_grad():
        if emb is not None:
            tr.pem.param_embedding["batch"].param_trainable.data[:] = torch.as_tensor(emb, dtype=torch.float32)
        if stain is not None:
            tr.param_stainscale.parameter.param_trainable.data[:] = torch.as_tensor(stain, dtype=torch.float32)
        if channel is not None:
            tr.param_channelscale.parameter.param_trainable.data[:] = torch.as_tensor(channel, dtype=torch.float32)


# ------------------------------------------------------------------ pick functions
def test_pick_transformer_and_panel_cls():
    assert pick_transformer_cls("dummy") is DummyTransformer
    assert pick_transformer_cls("spectralfit") is SpectralFitTransformer
    assert pick_transformer_cls("scaling") is ScalingTransformer
    assert pick_panel_cls("embedding") is PanelEmbeddingModule
    assert pick_panel_cls("fixed") is FixedPanelModule
    with pytest.raises(NotImplementedError):
        pick_transformer_cls("x")
    with pytest.raises(NotImplementedError):
        pick_panel_cls("x")


# ------------------------------------------------------------------ ScaleFactor
def test_scale_factor_starts_at_one_and_anchors_stay_fixed():
    sf = ScaleFactor(3, 2, anchoridxs=[0], trainable=True)
    assert sf.shape == (3, 2)
    assert torch.equal(sf.factor, torch.ones(3, 2)) and torch.equal(sf.logfactor, torch.zeros(3, 2))
    sf.parameter.param_trainable.data[:] = 1.
    assert torch.equal(sf.logfactor[0], torch.zeros(2)) and torch.equal(sf.logfactor[1:], torch.ones(2, 2))
    assert torch.allclose(sf.factor, sf.logfactor.exp())


def test_scale_factor_channel_mask_and_l1():
    sf = ScaleFactor(2, 3, use_channel_mask=[True, False, True], trainable=True)
    sf.parameter.param_trainable.data[:] = torch.tensor([[1., 1., -2.], [-1., 1., 0.]])
    assert sf.logfactor[:, 1].tolist() == [0., 0.]
    assert sf.L1.item() == pytest.approx(1. + 0. + 1.)  # mean |log| over batches, summed over features


def test_scale_factor_mean_norm():
    sf = ScaleFactor(3, 1, trainable=True, mean_norm=True)
    sf.parameter.param_trainable.data[:] = torch.tensor([[1.], [2.], [6.]])
    assert sf.logfactor[:, 0].tolist() == pytest.approx([-2., -1., 3.])


def test_scale_factor_max_abs_value_is_straight_through():
    sf = ScaleFactor(1, 2, trainable=True, max_abs_value=.5)
    sf.parameter.param_trainable.data[:] = torch.tensor([[2., -.1]])
    out = sf()
    assert np.allclose(out.tolist(), [[.5, -.1]])
    out.sum().backward()
    assert sf.parameter.param_trainable.grad.tolist() == [[1., 1.]]


def test_scale_factor_add_to_fixed_and_clip_warning():
    """R2-K2: adding to the fixed part warns when the result exceeds max_abs_value."""
    sf = ScaleFactor(2, 2, trainable=True, max_abs_value=1.)
    sf.add_to_fixed(torch.tensor([.3, .4]), bidx=[1])
    assert np.allclose(sf.logfactor.tolist(), [[0., 0.], [.3, .4]])
    sf.add_to_fixed(torch.tensor([[0.], [0.]]), cidx=[0])
    with pytest.warns(OverlapStandardisationWarning, match="clipped"):
        sf.add_to_fixed(torch.full((2, 2), 2.))


def test_scale_factor_anchor_bidx_and_freeze():
    sf = ScaleFactor(3, 1, trainable=True)
    sf.parameter.param_trainable.data[:] = .7
    sf.anchor_bidx([1])
    assert sf.parameter.param_mask_trainable[:, 0].tolist() == [1., 0., 1.]
    assert sf.logfactor[:, 0].tolist() == pytest.approx([.7, .7, .7])
    sf.freeze()
    assert (sf.parameter.param_mask_trainable == 0).all()


def test_scale_factor_extend_keeps_masked_features_fixed():
    """R2d-D2: features excluded by use_channel_mask stay untrainable in rows added by extend."""
    sf = ScaleFactor(2, 3, use_channel_mask=[True, False, True], trainable=True)
    sf.extend([2, 3])
    assert sf.shape == (4, 3)
    m = sf.parameter.param_mask_trainable
    assert m[:2].sum() == 0  # existing rows frozen
    assert m[2:].tolist() == [[1., 0., 1.], [1., 0., 1.]]
    assert torch.equal(sf.factor[2:], torch.ones(2, 3))


# ------------------------------------------------------------------ DummyTransformer
def test_dummy_transformer_is_identity_and_roundtrips():
    """R3b-B7: DummyTransformer is a working identity transformer (stains default to the channels)."""
    tr = DummyTransformer.from_data(CH)
    assert tr.stains == CH and tr.stain_marker_name == CH and tr.NAME == "dummy"
    x = torch.rand(4, 3)
    assert torch.equal(tr(x, {}), x) and torch.equal(tr.standardise(x, {}, evl=True), x)
    assert tr.loss_use_stain_mask.tolist() == [True] * 3
    assert tr.get_aux_loss()["loss_transformer"] == 0.
    assert tr.device.type == "cpu"
    tr.append_history(); tr.freeze(); tr.extend(None, {})
    tr2 = DummyTransformer.from_exported(tr.export(include_history=False))
    assert tr2.channels == CH and tr2.stains == CH
    tr3 = DummyTransformer.from_data(CH, stains=["a", "b", "c"])
    assert tr3.stains == ["a", "b", "c"]
    with pytest.raises(ValueError, match="one stain per channel"):
        DummyTransformer.from_data(CH, ["s1"])


# ------------------------------------------------------------------ SpectralFitTransformer
def test_spectralfit_from_data_config(panelconfig):
    tr = spectral_transformer(panelconfig, scale_stains=pd.Series([False, True, True], index=ST[::-1]),
                              scale_channels=False, key_scale_stains="batch")
    assert tr.config["scale_stains"].index.tolist() == ST
    assert tr.config["scale_stains"].tolist() == [True, True, False]
    assert tr.config["scale_channels"].tolist() == [False] * 3
    assert tr.config["key_scale_channels"] == "batch"
    assert tr.stains.tolist() == ST and tr.channels.tolist() == CH
    assert set(tr.plot_auxloss_dct) == {"l1_embedding", "l1_stainlogscale"}
    assert tr.loss_use_stain_mask.tolist() == [True] * 3
    # unscaled stain / channels: not trainable
    assert tr.param_stainscale.parameter.param_mask_trainable[1:, 2].sum() == 0
    assert tr.param_channelscale.parameter.param_mask_trainable.sum() == 0


def test_spectralfit_anchor_batch_is_frozen(panelconfig):
    tr = spectral_transformer(panelconfig)
    for p in [tr.param_stainscale.parameter, tr.param_channelscale.parameter]:
        assert p.param_mask_trainable[0].sum() == 0 and p.param_mask_trainable[1:].all()
    m = tr.pem.param_embedding["batch"].param_mask_trainable
    assert m[:, 0].sum() == 0 and m[:, 1:].all()


def test_spectralfit_standardise_formula(panelconfig):
    tr = spectral_transformer(panelconfig)
    emb = np.array([[0., 1.5, -2.], [0., -.5, 1.], [0., 3., 0.]])  # (component, batch); FL3_PC0 is unused
    stain = np.array([[0., 0., 0.], [.2, -.1, .3], [-.3, .4, 0.]])
    channel = np.array([[0., 0., 0.], [.1, .2, -.2], [0., -.1, .1]])
    set_params(tr, emb, stain, channel)
    x = torch.tensor(np.random.default_rng(0).uniform(1e3, 1e5, (6, 3)), dtype=torch.float32)
    for b in range(3):
        bidx = {"batch": torch.full((6,), b, dtype=torch.long)}
        got = tr.standardise(x, bidx).detach().numpy()
        sp = spectra_of(emb[:, b])
        expected = (x.numpy() * np.exp(channel[b])[None]) @ invert_spectra(pd.DataFrame(sp)).to_numpy() * np.exp(stain[b])[None]
        assert np.allclose(got, expected, rtol=1e-4), b
        assert torch.allclose(tr(x, bidx), torch.arcsinh(tr.standardise(x, bidx) / 5000))
        # (1,) batch indices broadcast over the events
        assert torch.allclose(tr.standardise(x, {"batch": torch.tensor([b])}), tr.standardise(x, bidx))


def test_spectralfit_learned_parameters_and_aux_loss(panelconfig):
    tr = spectral_transformer(panelconfig, L1weight_embedding=2., L1weight_stainscale=3., L1weight_channelscale=5.)
    set_params(tr, emb=np.full((3, 3), .5), stain=np.full((3, 3), .1), channel=np.full((3, 3), -.2))
    ls, lc = tr.learned_stainlogscale, tr.learned_channellogscale
    assert ls.index.tolist() == synthetic.BATCHES and ls.columns.tolist() == ST
    assert lc.columns.tolist() == CH and np.allclose(lc.loc["B"], -.2) and np.allclose(lc.loc["A"], 0.)
    aux = tr.get_aux_loss()
    assert aux["l1_stainlogscale"].item() == pytest.approx(3. * tr.param_stainscale.L1.item())
    assert aux["l1_channellogscale"].item() == pytest.approx(5. * tr.param_channelscale.L1.item())
    assert aux["l1_embedding"].item() == pytest.approx(2. * tr.pem.get_L1().item())
    assert aux["loss_transformer"].item() == pytest.approx(aux["l1_stainlogscale"].item() + aux["l1_channellogscale"].item()
                                                           + aux["l1_embedding"].item())


def test_spectralfit_history_and_plots(panelconfig):
    tr = spectral_transformer(panelconfig)
    for i in range(3):
        set_params(tr, emb=np.full((3, 3), .1 * i))
        tr.append_history()
    assert len(tr.paramhistory["embedding"]) == 3 and len(tr.paramhistory["stainlogscale"]) == 3
    tr.plot_history_embedding(legend=True)
    tr.plot_history_embedding_byexpvar()
    tr.plot_logscale_history(legend=True)


def test_spectralfit_export_roundtrip_with_multiplier(panelconfig):
    tr = spectral_transformer(panelconfig)
    tr.set_smp(UnmxErrorExclusionMultiplier(tr, groups=[("FL1", 4e3, ["FL2"])]))
    set_params(tr, emb=np.random.default_rng(1).normal(size=(3, 3)), stain=np.full((3, 3), .2), channel=np.full((3, 3), .1))
    tr.append_history()
    tr2 = SpectralFitTransformer.from_exported(tr.export())
    assert isinstance(tr2.smp, UnmxErrorExclusionMultiplier) and tr2.smp.groups == tr.smp.groups
    assert len(tr2.paramhistory["embedding"]) == 1
    assert "paramhistory" not in tr.export(include_history=False)
    x = torch.rand(4, 3) * 1e4
    bidx = {"batch": torch.tensor([2])}
    assert torch.allclose(tr.standardise(x, bidx), tr2.standardise(x, bidx))
    # anchors are still frozen after loading
    assert tr2.param_stainscale.parameter.param_mask_trainable[0].sum() == 0


def test_spectralfit_standardisation_params_reproduce_standardise(panelconfig):
    tr = spectral_transformer(panelconfig)
    set_params(tr, emb=np.array([[0., 1., 2.], [0., -1., .5], [0., 0., 0.]]), stain=np.full((3, 3), .3),
               channel=np.array([[0.] * 3, [.1, -.1, .2]] * 1 + [[0.] * 3]))
    ad = adata_of("B")
    cs, ss, spectra, unmx, emb = tr.get_standardisationparams_fromad(ad, label_include_stain=True)
    assert spectra.shape == (3, 3) and spectra.index.tolist() == CH
    assert ss.index.tolist() == list(tr.stain_marker_name)
    assert np.allclose(spectra.T.to_numpy(), spectra_of([1., -1., 0.]), atol=1e-6)
    assert emb.shape == (3,) and emb[2] == 0.  # unused component masked
    x = ad.layers["calibrated"]
    std = tr.standardise(torch.tensor(x), tr.moe.transform_ad(ad)).detach().numpy()
    assert np.allclose(x @ unmx.to_numpy(), std, rtol=1e-4)
    cs2, ss2, sp2, _, _ = tr.get_standardisationparams_fromad(ad)
    assert ss2.index.tolist() == synthetic.MARKERS


def test_spectralfit_add_standardisation_parameters(panelconfig):
    tr = spectral_transformer(panelconfig)
    set_params(tr, stain=np.full((3, 3), .3), channel=np.full((3, 3), .1))
    ad = adata_of("C", channels=["FSC-A"] + CH)
    ad.var["rainbow_calibration_factor"] = [np.nan, 2., 3., 4.]
    tr.add_standardisation_parameters(ad)
    assert np.isnan(ad.var.loc["FSC-A", "sfm_calibration_factor"])
    assert np.allclose(ad.var.loc[CH, "sfm_calibration_factor"], np.exp(.1))
    assert np.isnan(ad.varm["sfm_spectra"].loc["FSC-A"]).all()
    assert (ad.varm["sfm_standardise_unmx"].loc["FSC-A"] == 0).all()
    assert np.allclose(ad.varm["sfm_stain_factor"].to_numpy(), np.exp(.3))
    std = tr.standardise(torch.tensor(ad.layers["calibrated"][:, 1:]), tr.moe.transform_ad(ad)).detach().numpy()
    assert np.allclose((ad.to_df(layer="calibrated") @ ad.varm["sfm_standardise_unmx"]).to_numpy(), std, rtol=1e-4)
    rb = ad.varm["rb_sfm_standardise_unmx"]
    assert np.allclose(rb.to_numpy(), ad.var["rainbow_calibration_factor"].fillna(0).to_numpy()[:, None]
                       * ad.varm["sfm_standardise_unmx"].to_numpy())
    assert ad.uns["panel_embedding"].shape == (3,)


def test_spectralfit_with_fixed_panel_module():
    """R2b-G2-18: SpectralFitTransformer works with a FixedPanelModule (full spectra, empty embedding)."""
    for sp in [SP.iloc[:2], SP]:
        moe = moe_abc()
        tr = SpectralFitTransformer.from_data(moe, FixedPanelModule.from_data(sp), scale_stains=True)
        ad = adata_of("B")
        cs, ss, spectra, unmx, emb = tr.get_standardisationparams_fromad(ad)
        assert np.allclose(spectra.to_numpy(), sp.T.to_numpy()) and emb.shape == (0,)
        assert np.allclose(np.asarray(unmx), cs.to_numpy()[:, None] * invert_spectra(sp).to_numpy() * ss.to_numpy()[None])
        tr.add_standardisation_parameters(ad)
        assert ad.uns["panel_embedding"].shape == (0,)
        assert "l1_embedding" not in tr.plot_auxloss_dct


def test_spectralfit_freeparam_functions_match_forward(panelconfig):
    tr = spectral_transformer(panelconfig)
    set_params(tr, emb=np.full((3, 3), .4), stain=np.full((3, 3), .2), channel=np.full((3, 3), -.1))
    x = torch.rand(10, 3) * 1e4
    plist = []
    for b in [1, 2]:
        bidx = {"batch": torch.tensor([b])}
        params, fct = tr.get_freeparam_standardisetransf(bidx)
        assert set(params) == {"embedding", "channellogscale", "stainlogscale"}
        out = fct(x)
        assert torch.allclose(out, tr(x, bidx), atol=1e-6)
        out.sum().backward()
        plist.append(params)
    grads = tr.get_freeparam_collect(plist)
    assert grads["channellogscale"].shape == (2, 3) and grads["channellogscale"].columns.tolist() == CH
    assert grads["stainlogscale"].columns.tolist() == ST
    assert grads["embedding"].shape == (2, 3)
    assert np.abs(grads["stainlogscale"].to_numpy()).sum() > 0
    fixed = SpectralFitTransformer.from_data(moe_abc(), FixedPanelModule.from_data(SP))
    with pytest.raises(NotImplementedError):
        fixed.get_freeparam_standardisetransf({"batch": torch.tensor([0])})


def test_spectralfit_freeze_and_extend(panelconfig):
    """R2d-D2: after extend, the new batches' unscaled stains/channels stay untrainable."""
    moe = moe_abc()
    tr = spectral_transformer(panelconfig, moe=moe, scale_stains=pd.Series([True, False, True], index=ST),
                              scale_channels=pd.Series([False, True, True], index=CH))
    new = moe.extend(pd.DataFrame({"batch": ["D"]}))
    tr.extend(moe, new)
    assert tr.param_stainscale.shape == (4, 3) and tr.param_channelscale.shape == (4, 3)
    assert tr.param_stainscale.parameter.param_mask_trainable.tolist() == [[0.] * 3] * 3 + [[1., 0., 1.]]
    assert tr.param_channelscale.parameter.param_mask_trainable.tolist() == [[0.] * 3] * 3 + [[0., 1., 1.]]
    assert tr.pem.param_embedding["batch"].shape[1] == 4
    assert tr.learned_stainlogscale.index.tolist() == synthetic.BATCHES + ["D"]
    tr.freeze()
    assert not any((p.param_mask_trainable != 0).any() for p in [tr.param_stainscale.parameter, tr.param_channelscale.parameter])


def test_spectralfit_add_stain_logscale(panelconfig):
    tr = spectral_transformer(panelconfig)
    ls = pd.DataFrame([[.1], [.2], [.3]], index=["C", "A", "B"], columns=["FL2"])
    tr.add_stain_logscale_toself(ls)
    assert tr.learned_stainlogscale["FL2"].tolist() == pytest.approx([.2, .3, .1])
    assert tr.learned_stainlogscale["FL1"].tolist() == [0., 0., 0.]
    with pytest.raises(OverlapStandardisationException, match="different batches"):
        tr.add_stain_logscale_toself(ls.iloc[:2])
    with pytest.raises(OverlapStandardisationException, match="not currently present"):
        tr.add_stain_logscale_toself(ls.rename(columns={"FL2": "XX"}))


# ------------------------------------------------------------------ ScalingTransformer
def scaling_transformer(anchor=0, spectra=SP, **kw):
    kw = dict(dict(scale_stains=True), **kw)
    if anchor is not None:
        kw["anchor_bidx"] = {"batch": anchor}
    return ScalingTransformer.from_data(moe_abc(), spectra, **kw)


def test_scaling_no_scaled_stain_warns():
    """R2-C2: no scaled stain gives an OverlapStandardisationWarning (crashed before: warnings not imported)."""
    with pytest.warns(OverlapStandardisationWarning, match="no stains"):
        tr = ScalingTransformer.from_data(moe_abc(), SP, scale_stains=False)
    assert tr.loss_use_stain_mask.tolist() == [False] * 3


@pytest.mark.parametrize("anchor", [1, [1], np.int64(1)])
def test_scaling_anchor_int_or_list(anchor):
    """R2-E17: an int or a one-element list anchors one batch; several anchors raise ValueError."""
    assert scaling_transformer(anchor=anchor).anchor_idx == [1]


def test_scaling_several_anchors_raise():
    with pytest.raises(ValueError, match="single batch"):
        scaling_transformer(anchor=[0, 1])
    assert scaling_transformer(anchor=None).anchor_idx is None
    # other batch keys are ignored
    assert ScalingTransformer.from_data(moe_abc(), SP, scale_stains=True, anchor_bidx={"other": [0, 1]}).anchor_idx is None


def test_scaling_mean_norm_and_anchor_relative_factors():
    tr = scaling_transformer(anchor=1, scale_stains=pd.Series([True, True, False], index=ST))
    assert tr.loss_use_stain_mask.tolist() == [True, True, False]
    set_params(tr, stain=[[.1, .3, 0.], [.4, -.3, 0.], [.7, 0., 0.]])
    raw = tr.param_stainscale.logfactor.detach()
    assert torch.allclose(raw.mean(0), torch.zeros(3), atol=1e-6)  # mean 0 over batches during the fit
    ls = tr.learned_stainlogscale
    assert np.allclose(ls.loc["B"], 0.)
    assert np.allclose(ls.to_numpy(), (raw - raw[1]).numpy())
    assert np.allclose(tr.learned_stainscale.to_numpy(), np.exp(ls.to_numpy()))
    assert ls.columns.tolist() == ST and ls.index.tolist() == synthetic.BATCHES
    assert np.allclose(ls["FL3"], 0.)
    x = torch.rand(5, 3) * 1e4
    bidx = {"batch": torch.tensor([2])}
    unmixed = x.numpy() @ invert_spectra(SP).to_numpy()
    assert np.allclose(tr.standardise(x, bidx, evl=True).detach().numpy(), unmixed * np.exp(ls.loc["C"].to_numpy())[None], rtol=1e-4)
    assert np.allclose(tr.standardise(x, bidx).detach().numpy(), unmixed * np.exp(raw[2].numpy())[None], rtol=1e-4)
    assert torch.allclose(tr(x, bidx), torch.arcsinh(tr.standardise(x, bidx) / 5000))


def test_scaling_without_anchor_learned_has_mean_zero():
    tr = scaling_transformer(anchor=None)
    set_params(tr, stain=np.random.default_rng(0).normal(size=(3, 3)))
    assert np.allclose(tr.learned_stainlogscale.mean(0), 0., atol=1e-6)


def test_scaling_standardisation_params_spectral_panel():
    """R2-B2 + R2-A3: spectral panel (fewer stains than channels) unmixes exactly; the stain factor goes to varm."""
    sp = SP.iloc[:2]
    tr = scaling_transformer(anchor=0, spectra=sp)
    set_params(tr, stain=[[.5, -.5], [.1, .2], [0., 0.]])
    ad = adata_of("B", channels=CH + ["FSC-A"])
    scale, spectra, unmx = tr.get_standardisationparams_fromad(ad)
    assert unmx.index.tolist() == CH and spectra.shape == (3, 2)
    A = np.array([[1., 2.], [3., -1.]])
    assert np.allclose((A @ sp.to_numpy()) @ unmx.to_numpy(), A * scale.to_numpy()[None], atol=1e-5)
    assert np.allclose(scale.to_numpy(), tr.learned_stainscale.loc["B"].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        tr.add_standardisation_parameters(ad)
    assert "sfm_stain_factor" not in ad.var
    assert np.allclose(ad.varm["sfm_stain_factor"].to_numpy(), np.repeat(scale.to_numpy()[None], 4, 0))
    assert np.isnan(ad.varm["sfm_spectra"].loc["FSC-A"]).all()
    assert (ad.varm["sfm_standardise_unmx"].loc["FSC-A"] == 0).all()
    std = tr.standardise(torch.tensor(ad.layers["calibrated"][:, :3]), tr.moe.transform_ad(ad), evl=True).detach().numpy()
    assert np.allclose((ad.to_df(layer="calibrated") @ ad.varm["sfm_standardise_unmx"]).to_numpy(), std, rtol=1e-4)


def test_scaling_add_stain_logscale_only_differences_take_effect():
    tr = scaling_transformer(anchor=0)
    before = tr.learned_stainlogscale.copy()
    ls = pd.DataFrame({"FL1": [1., 1.5, 3.]}, index=synthetic.BATCHES)
    tr.add_stain_logscale_toself(ls)
    after = tr.learned_stainlogscale
    assert np.allclose(after["FL1"] - before["FL1"], ls["FL1"] - ls.loc["A", "FL1"])
    assert np.allclose(after[["FL2", "FL3"]], before[["FL2", "FL3"]])
    with pytest.raises(OverlapStandardisationException):
        tr.add_stain_logscale_toself(ls.iloc[1:])


def test_scaling_export_roundtrip_and_aux_loss():
    tr = scaling_transformer(anchor=2, L1weight_stainscale=2., max_abslogstain=1.)
    set_params(tr, stain=np.random.default_rng(0).normal(size=(3, 3)))
    tr.append_history()
    tr2 = ScalingTransformer.from_exported(tr.export())
    assert tr2.anchor_idx == [2] and len(tr2.paramhistory["stainlogscale"]) == 1
    assert np.allclose(tr2.learned_stainlogscale, tr.learned_stainlogscale)
    aux = tr.get_aux_loss()
    assert aux["loss_transformer"].item() == pytest.approx(2. * tr.param_stainscale.L1.item())
    assert set(aux.keys()) == {"l1_stainlogscale", "loss_transformer"}
    tr.plot_logscale_history(legend=True)


def test_scaling_extend_masks_and_freeze():
    """R2d-D2: unscaled stains stay untrainable for new batches."""
    tr = scaling_transformer(anchor=0, scale_stains=pd.Series([True, False, True], index=ST))
    moe = tr.moe
    new = moe.extend(pd.DataFrame({"batch": ["D"]}))
    tr.extend(moe, new)
    assert tr.param_stainscale.parameter.param_mask_trainable[3].tolist() == [1., 0., 1.]
    tr.freeze()
    assert (tr.param_stainscale.parameter.param_mask_trainable == 0).all()


@pytest.mark.xfail(strict=True, reason="R4 finding ref_model #1: new batches of an extended ScalingTransformer start at "
                                       "exp(-mean of the old un-normalised log factors), not at factor 1")
def test_scaling_extend_new_batch_starts_at_factor_one():
    tr = scaling_transformer(anchor=None)
    # un-normalised log factors with mean 1.5 for FL1 (reachable with Adam, whose updates are not mean-free)
    set_params(tr, stain=[[1., 0., 0.], [2., 0., 0.], [1.5, 0., 0.]])
    before = tr.learned_stainlogscale.copy()
    new = tr.moe.extend(pd.DataFrame({"batch": ["D"]}))
    tr.extend(tr.moe, new)
    after = tr.learned_stainlogscale
    assert np.allclose(after.loc[synthetic.BATCHES], before)  # old batches unchanged
    assert np.allclose(after.loc["D"], 0.)                     # "New batches start at factor 1"
