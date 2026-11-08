"""Tests of cytovanni.ref.model: OverlapFitModel (loss, fit, export / save / load, applying the model)."""
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.exceptions import OverlapStandardisationException, OverlapStandardisationWarning
from cytovanni.ref import (DummyTransformer, MultiOrdinalEncoder, OverlapFitModel, PanelEmbeddingModule, RareMarkerWeights,
                           SampleOverlapDataset, ScalingTransformer, SpectralFitTransformer, UnmxErrorExclusionMultiplier)
from cytovanni.ref.model import _sum_loss_separate_markers, _sum_loss_separate_markers_loop
from cytovanni.torch import GeomSampleLoss, MMDLoss
from synthetic import PANEL_COMPONENTS, make_panelconfig_csv, panel_spectra_of, set_transformer_params

CH = synthetic.CHANNELS_FLUORESCENCE
ST = synthetic.STAINS


def overlap_adatas(n=1000, batches=synthetic.BATCHES, abundances=None):
    """One reference sample measured in every batch, on the calibrated scale (stain scales of synthetic.STAIN_SCALES)."""
    ab = synthetic.cell_abundances(n, np.random.default_rng(7)) if abundances is None else abundances
    ads = []
    for i, b in enumerate(batches):
        rng = np.random.default_rng(i)
        sp = synthetic.SPECTRA.to_numpy()
        x = (ab * synthetic.STAIN_SCALES.loc[b].to_numpy()[None]) @ sp + 15 * rng.normal(size=ab.shape)
        ads.append(synthetic.make_adata(x, CH, batch=b, uid=f"ref_{b}", type="ref"))
    return ads


def quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


@pytest.fixture
def ads():
    return overlap_adatas()


@pytest.fixture
def moe(ads):
    return MultiOrdinalEncoder.from_adatas(ads, ["batch"])


def make_dataset(ads, moe, anchors=("A",), **kw):
    kw = dict(dict(sN_cells=400, min_sample_cells=100), **kw)
    isanchor = [ad.uns["batch"] in anchors for ad in ads] if anchors else None
    return quiet(SampleOverlapDataset, ads, moe, CH, data_isanchor=isanchor, **kw)


def scaling(moe, **kw):
    kw = dict(dict(scale_stains=True, anchor_bidx={"batch": 0}), **kw)
    return ScalingTransformer.from_data(moe, synthetic.SPECTRA, **kw)


# ------------------------------------------------------------------ helpers
def test_sum_loss_separate_markers_matches_loop():
    torch.manual_seed(0)
    x, y = torch.randn(40, 3), torch.randn(50, 3) + .5
    wx = torch.rand(40) + .5
    g = GeomSampleLoss("sinkhorn")
    batched = _sum_loss_separate_markers(g, x, y, wx, None)
    loop = sum(g(x[:, [i]], y[:, [i]], wx, None) for i in range(3))
    # the batched geomloss call shares one epsilon schedule over the markers, so it differs slightly from the loop
    assert torch.allclose(batched, loop, rtol=1e-2)
    m = MMDLoss(bandwidth=1.)
    assert torch.allclose(_sum_loss_separate_markers_loop(m, x, y, None, None),
                          sum(m(x[:, [i]], y[:, [i]]) for i in range(3)))


# ------------------------------------------------------------------ construction, losses
def test_default_ddist_mode_is_sinkhorn(moe):
    """R2-C9: direct construction without DDIST_mode works (default 'sinkhorn', was the typo 'sinnkhorn')."""
    m = OverlapFitModel(moe, scaling(moe))
    assert m.DDIST_mode == "sinkhorn" and isinstance(m.ddist_fct, GeomSampleLoss)
    assert isinstance(m.markerweight, type(OverlapFitModel(moe, scaling(moe)).markerweight))
    assert m.losshistory == {"train": [], "test": [], "train_DDIST_mat": [], "test_DDIST_mat": []}


@pytest.mark.parametrize("mode", ["sinkhorn", "hausdorff", "MMD"])
@pytest.mark.parametrize("separate", [False, True])
def test_setup_ddist_modes_and_separate(moe, mode, separate):
    m = OverlapFitModel(moe, scaling(moe), DDIST_mode=mode, DDIST_separate=separate, bandwidth_MMD=1.)
    torch.manual_seed(0)
    x, y = torch.randn(60, 3), torch.randn(50, 3) + 1
    loss = m.ddist(x, y, None, None)
    if separate:
        assert torch.allclose(loss, sum(m.ddist_fct(x[:, [i]], y[:, [i]], None, None) for i in range(3)), rtol=1e-2)
    else:
        assert torch.allclose(loss, m.ddist_fct(x, y, None, None))
    assert loss.item() > 0


def test_setup_ddist_unknown_mode_raises(moe):
    with pytest.raises(NotImplementedError):
        OverlapFitModel(moe, scaling(moe), DDIST_mode="wasserstein")
    m = OverlapFitModel(moe, scaling(moe), DDIST_mode="wasserstein", eval_only=True)
    assert not hasattr(m, "ddist")


def test_dataset_channel_mismatch_raises(ads, moe):
    """R2d-D3: training and test datasets must have the transformer's channels in the same order."""
    ds_rev = quiet(SampleOverlapDataset, ads, moe, CH[::-1], sN_cells=50, min_sample_cells=10)
    ds = make_dataset(ads, moe)
    tr = scaling(moe)
    with pytest.raises(ValueError, match="different channels"):
        OverlapFitModel(moe, tr, dataset=ds_rev)
    with pytest.raises(ValueError, match="different channels"):
        OverlapFitModel(moe, tr, dataset=ds, dataset_test=ds_rev)
    m = OverlapFitModel(moe, tr, dataset=ds)
    with pytest.raises(ValueError, match="different channels"):
        m.get_DDIST_from_dataset(ds_rev)


def test_get_ddist_from_dataset_loss_and_matrix(ads, moe):
    ds = make_dataset(ads, moe, anchors=None)
    m = OverlapFitModel(moe, scaling(moe), dataset=ds)
    loss, mat = m.get_DDIST_from_dataset(ds)
    assert loss["loss"] is loss["ddist"] and loss["ddist"].requires_grad
    assert mat.shape == (3, 3) and mat.index.tolist() == ["ref_A", "ref_B", "ref_C"]
    drawn = mat.to_numpy()[np.isfinite(mat.to_numpy())]
    assert len(drawn) == 3 and np.isclose(drawn.mean(), loss["ddist"].item(), rtol=1e-5)
    # no test dataset: NaN loss and an empty matrix
    tl, tm = m.get_DDIST(train=False)
    assert np.isnan(tl["loss"]) and tm.empty


def test_get_ddist_test_dataset_has_no_gradient(ads, moe):
    ds = make_dataset(ads, moe)
    m = OverlapFitModel(moe, scaling(moe), dataset=ds, dataset_test=make_dataset(ads, moe))
    tl, tm = m.get_DDIST(train=False)
    assert not tl["loss"].requires_grad and tm.shape == (3, 3)


def test_marker_weights_enter_the_loss(ads, moe):
    ds = make_dataset(ads, moe, anchors=None)
    tr = scaling(moe)
    rmw = RareMarkerWeights(tr, groups=[("FL1", 2e4, 5.)])
    m0 = OverlapFitModel(moe, tr, dataset=ds, checkpoint_DDIST=False)
    m1 = OverlapFitModel(moe, tr, dataset=ds, markerweight=rmw, checkpoint_DDIST=False)
    torch.manual_seed(1); np.random.seed(1)
    l0 = m0.get_DDIST()[0]["loss"].item()
    torch.manual_seed(1); np.random.seed(1)
    l1 = m1.get_DDIST()[0]["loss"].item()
    assert np.isfinite(l1) and l1 != pytest.approx(l0)


def test_checkpoint_gives_same_loss_and_gradient(moe):
    tr = scaling(moe)
    set_transformer_params(tr, stain=[[.1, .2, .3], [0., -.2, .1], [.3, 0., -.1]])
    x, y = torch.randn(80, 3), torch.randn(70, 3) + .5
    grads = []
    for cp in [True, False]:
        m = OverlapFitModel(moe, tr, checkpoint_DDIST=cp)
        tr.zero_grad()
        sx = tr(x * 1e4, {"batch": torch.tensor([1])})
        l = m.get_DDIST_single(sx, tr(y * 1e4, {"batch": torch.tensor([2])}), trainy=False)
        l.backward()
        grads.append((l.item(), tr.param_stainscale.parameter.param_trainable.grad.clone()))
    assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-6)
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-6)
    assert torch.isfinite(grads[0][1]).all()


# ------------------------------------------------------------------ MMD
def test_from_data_mmd_estimates_bandwidth_on_loss_stains(ads, moe):
    """R2b-G2-07 + R2-C12: MMD bandwidth from the stains of the loss, per stain with DDIST_separate."""
    ds = make_dataset(ads, moe)
    for scale in [[False, True, False], [True, True, True]]:
        tr = scaling(moe, scale_stains=pd.Series(scale, index=ST))
        for sep in [False, True]:
            m = quiet(OverlapFitModel.from_data, moe, tr, ds, DDIST_mode="MMD", DDIST_separate=sep)
            torch.manual_seed(5); np.random.seed(5)
            bw = m.estimate_MMD_bandwidth(ds)
            torch.manual_seed(5); np.random.seed(5)
            with torch.no_grad():
                xs = [tr(d["x"], d["idx_batch"])[:, torch.tensor(scale)] for d in ds.sample()["data"]]
            bws = []
            for x in xs:
                bws += [MMDLoss.estimate_bandwidth(x[:, [i]]) for i in range(x.shape[1])] if sep else [MMDLoss.estimate_bandwidth(x)]
            assert bw == pytest.approx(torch.stack(bws).median().item(), rel=1e-6)
            assert m.bandwidth_MMD > 0 and m.ddist_fct.kernel.bandwidth == m.bandwidth_MMD
    m = OverlapFitModel.from_data(moe, scaling(moe), ds)  # sinkhorn: no bandwidth
    assert m.bandwidth_MMD is None


def test_mmd_separate_survives_export():
    """R2-C12: DDIST_separate is exported; a reloaded MMD model keeps the per-marker loss."""
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"batch": ["a", "b"]}), ["batch"])
    st = ScalingTransformer.from_data(moe, synthetic.SPECTRA.iloc[:2, :2], scale_stains=True)
    m = OverlapFitModel(moe, st, DDIST_mode="MMD", bandwidth_MMD=1., DDIST_separate=True)
    X, Y = torch.randn(50, 2), torch.randn(60, 2) + 1
    ref = MMDLoss(bandwidth=1.)(X[:, [0]], Y[:, [0]]) + MMDLoss(bandwidth=1.)(X[:, [1]], Y[:, [1]])
    assert torch.allclose(m.ddist(X, Y, None, None), ref)
    m2 = OverlapFitModel.from_exported(m.export(), eval_only=False)
    assert m2.DDIST_separate and m2.bandwidth_MMD == 1.
    assert torch.allclose(m2.ddist(X, Y, None, None), ref)
    exported = m.export()
    del exported["DDIST_separate"]
    with pytest.raises(KeyError):  # strict: old saved models without the key must be re-saved (user decision)
        OverlapFitModel.from_exported(exported)


# ------------------------------------------------------------------ fit
def test_fit_without_dataset_raises(moe):
    with pytest.raises(OverlapStandardisationException):
        OverlapFitModel(moe, scaling(moe)).fit(Niter=1)


def test_fit_records_history_and_reduces_loss(ads, moe, device):
    ds = make_dataset(ads, moe, anchors=None, device=device, sN_cells=250)
    m = OverlapFitModel(moe, scaling(moe), dataset=ds, dataset_test=make_dataset(ads, moe, anchors=None, device=device, sN_cells=250),
                        DDIST_separate=True).to(device)
    m.fit(Niter=15, lr=5e-2)
    h = pd.DataFrame(m.losshistory["train"])
    assert len(h) == 15 and h["step"].tolist() == list(range(15))
    assert {"loss", "ddist", "l1_stainlogscale", "loss_transformer"} <= set(h.columns)
    assert len(m.losshistory["test"]) == 15 and len(m.losshistory["train_DDIST_mat"]) == 15
    assert len(m.transformer.paramhistory["stainlogscale"]) == 15
    assert h["loss"].iloc[-3:].mean() < h["loss"].iloc[:3].mean()
    m.plot_losshistory()
    # continued fit appends
    m.fit(Niter=2, lr=1e-3, optimizer="SGD")
    assert len(m.losshistory["train"]) == 17


def test_fit_keeps_spectralfit_anchor_parameters(tmp_path, device):
    ads = overlap_adatas(n=600, batches=["A", "B"])
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    pc = make_panelconfig_csv(tmp_path)
    pem = PanelEmbeddingModule.from_data(pc, moe, pd.Series("batch", index=PANEL_COMPONENTS.index), anchor_bidx={"batch": [0]})
    tr = SpectralFitTransformer.from_data(moe, pem, scale_stains=True, scale_channels=True, anchor_bidx={"batch": [0]})
    ds = make_dataset(ads, moe, device=device, sN_cells=300)
    m = OverlapFitModel(moe, tr, dataset=ds).to(device)
    m.fit(Niter=5, lr=5e-2)
    assert torch.all(tr.param_stainscale.factor[0] == 1.) and torch.all(tr.param_channelscale.factor[0] == 1.)
    assert torch.all(tr.pem(  {"batch": torch.tensor([0], device=device)})[:, 0] == 0.)
    assert not torch.all(tr.param_stainscale.factor[1] == 1.)


# ------------------------------------------------------------------ export, save, load
def test_save_load_roundtrip_scaling(ads, moe, tmp_path):
    ds = make_dataset(ads, moe)
    tr = scaling(moe)
    set_transformer_params(tr, stain=np.random.default_rng(0).normal(scale=.3, size=(3, 3)))
    m = OverlapFitModel(moe, tr, dataset=ds, DDIST_kwargs={"blur": .3}, checkpoint_DDIST=False)
    m.losshistory["train"].append({"loss": 1., "step": 0})
    path = str(tmp_path / "sub" / "model.pt")
    m.save(path)
    m2 = OverlapFitModel.from_saved(path)
    assert type(m2.transformer) is ScalingTransformer and m2.dataset is None
    assert m2.DDIST_kwargs == {"blur": .3} and m2.checkpoint_DDIST is False
    assert not hasattr(m2, "ddist")  # eval_only by default
    assert m2.losshistory["train"] == []  # history not saved by default
    assert np.allclose(m2.transformer.learned_stainlogscale, tr.learned_stainlogscale)
    m.save(path, include_history=True)
    assert OverlapFitModel.from_saved(path, eval_only=False).losshistory["train"] == [{"loss": 1., "step": 0}]


def test_export_roundtrip_spectralfit_with_weights_and_multiplier(tmp_path):
    ads = overlap_adatas(n=300, batches=["A", "B"])
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    pem = PanelEmbeddingModule.from_data(make_panelconfig_csv(tmp_path), moe, pd.Series("batch", index=PANEL_COMPONENTS.index),
                                         anchor_bidx={"batch": [0]})
    tr = SpectralFitTransformer.from_data(moe, pem, scale_stains=True, anchor_bidx={"batch": [0]})
    tr.set_smp(UnmxErrorExclusionMultiplier(tr, groups=[("FL1", 4e3, ["FL2"])]))
    set_transformer_params(tr, emb=np.full((3, 2), .5), stain=np.full((2, 3), .2))
    m = OverlapFitModel(moe, tr, markerweight=RareMarkerWeights(tr, groups=[("FL1", 4e3, 2.5)]))
    m2 = OverlapFitModel.from_exported(m.export())
    assert isinstance(m2.markerweight, RareMarkerWeights) and m2.markerweight.groups == [("FL1", 4e3, 2.5)]
    assert isinstance(m2.transformer.smp, UnmxErrorExclusionMultiplier)
    x = torch.tensor(ads[1].layers["calibrated"])
    bidx = moe.transform_ad(ads[1])
    assert torch.allclose(m2.transformer.standardise(x, bidx), tr.standardise(x, bidx))


def test_dummy_transformer_with_model(tmp_path):
    """R3b-B7: DummyTransformer works with OverlapFitModel for distances, applying, saving and extending."""
    x = np.column_stack([np.linspace(100, 200, 16), np.linspace(2000, 3000, 16)])
    channels = ["c1", "c2"]
    ads = [synthetic.make_adata(x, channels, batch="a", uid="u0", type="t"),
           synthetic.make_adata(x * 1.5, channels, batch="b", uid="u1", type="t")]
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    tr = DummyTransformer.from_data(channels)
    ds = quiet(SampleOverlapDataset, ads, moe, channels, min_sample_cells=1, sN_cells=16)
    m = OverlapFitModel(moe, tr, dataset=ds, DDIST_mode="MMD", bandwidth_MMD=1.)
    assert m.get_DDIST_from_dataset(ds)[0]["loss"].item() > 0
    m.add_ad_standardised(ads[1], addkey="std", name_by="stain")
    assert np.allclose(ads[1].obsm["std"].to_numpy(), ads[1].layers["calibrated"])
    path = str(tmp_path / "dummy.pt")
    m.save(path)
    assert OverlapFitModel.from_saved(path).transformer.stains == channels
    inds = moe.extend(pd.DataFrame({"batch": ["c"]}))
    tr.extend(moe, inds)


# ------------------------------------------------------------------ applying
@pytest.mark.parametrize("name_by, columns", [("both", ST), ("stain", ST), ("marker", ST)])
def test_add_ad_standardised_scaling(ads, moe, name_by, columns, device):
    tr = scaling(moe)
    set_transformer_params(tr, stain=[[.1, .2, .3], [0., -.2, .1], [.3, 0., -.1]])
    m = OverlapFitModel(moe, tr, eval_only=True).to(device)
    ad = ads[2]
    m.add_ad_standardised(ad, batch_size=300, name_by=name_by)
    out = ad.obsm["unmx_standardised"]
    assert out.columns.tolist() == columns and out.index.equals(ad.obs.index)
    expected = tr.standardise(torch.tensor(ad.layers["calibrated"], device=device), moe.transform_ad(ad), evl=True).detach().cpu().numpy()
    assert np.allclose(out.to_numpy(), expected, rtol=1e-5)
    assert np.allclose(ad.obsm["unmx_standardised_arcsinh"].to_numpy(), np.arcsinh(out.to_numpy() / 1500), rtol=1e-5)
    # fixed_batch overrides the labels in uns
    m.add_ad_standardised(ad, fixed_batch={"batch": "A"}, addkey="as_A", add_arcsinh=False)
    assert "as_A_arcsinh" not in ad.obsm
    expected_A = tr.standardise(torch.tensor(ad.layers["calibrated"], device=device), {"batch": torch.tensor([0])}, evl=True).detach().cpu().numpy()
    assert np.allclose(ad.obsm["as_A"].to_numpy(), expected_A, rtol=1e-5)


def test_add_ad_standardised_spectralfit_names_and_invalid(tmp_path):
    ads = overlap_adatas(n=200, batches=["A", "B"])
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    pem = PanelEmbeddingModule.from_data(make_panelconfig_csv(tmp_path), moe, pd.Series("batch", index=PANEL_COMPONENTS.index))
    tr = SpectralFitTransformer.from_data(moe, pem)
    m = OverlapFitModel(moe, tr, eval_only=True)
    for name_by, cols in [("both", [f"{s} | {mk}" for s, mk in zip(ST, synthetic.MARKERS)]), ("marker", synthetic.MARKERS),
                          ("stain", ST)]:
        m.add_ad_standardised(ads[1], name_by=name_by, addkey=name_by)
        assert ads[1].obsm[name_by].columns.tolist() == cols
    with pytest.raises(OverlapStandardisationException):
        m.add_ad_standardised(ads[1], name_by="dye")


def test_incorporate_stain_scaling_keeps_class_and_warns_on_clip(moe):
    """R2-K2: the copy keeps the model class, adds the other model's stain scaling and warns when it is clipped."""
    class MyModel(OverlapFitModel):
        pass
    tr = scaling(moe, max_abslogstain=.5)
    m = MyModel(moe, tr)
    tr_o = scaling(moe)
    set_transformer_params(tr_o, stain=[[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.]])
    with pytest.warns(OverlapStandardisationWarning, match="clipped"):
        new = m.incorporate_stain_scaling(MyModel(moe, tr_o))
    assert type(new) is MyModel and new is not m
    assert np.allclose(tr.learned_stainlogscale, 0.)  # the original is unchanged


def test_incorporate_stain_scaling_adds_differences(moe):
    tr = scaling(moe)
    m = OverlapFitModel(moe, tr)
    tr_o = scaling(moe)
    set_transformer_params(tr_o, stain=[[.2, 0., 0.], [-.1, .1, 0.], [-.1, -.1, 0.]])
    new = m.incorporate_stain_scaling(OverlapFitModel(moe, tr_o))
    assert np.allclose(new.transformer.learned_stainlogscale, tr_o.learned_stainlogscale, atol=1e-6)
    assert not hasattr(new, "ddist")


# ------------------------------------------------------------------ recovery
@pytest.mark.slow
def test_scaling_transformer_recovers_stain_scales(device):
    """C: the overlap fit with a ScalingTransformer recovers 1 / STAIN_SCALES relative to the anchor batch A (10 %)."""
    ads = overlap_adatas(n=1500)
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    ds = make_dataset(ads, moe, anchors=None, device=device, sN_cells=600)
    tr = scaling(moe)
    m = OverlapFitModel.from_data(moe, tr, ds, DDIST_separate=True).to(device)
    torch.manual_seed(0); np.random.seed(0)
    m.fit(Niter=100, lr=5e-2)
    got = tr.learned_stainscale
    expected = (1 / synthetic.STAIN_SCALES).div((1 / synthetic.STAIN_SCALES).loc["A"], axis=1)
    assert np.allclose(got.loc["A"], 1.)
    assert np.allclose(got.to_numpy(), expected.loc[got.index, got.columns].to_numpy(), rtol=.1), got


@pytest.mark.slow
def test_spectralfit_transformer_recovers_embedding_shift(tmp_path, device):
    """C: batch B measured with spectra at embedding (1.5, -1, 0) is fitted to that embedding (anchor A at 0), +-0.3."""
    true_emb = np.array([1.5, -1., 0.])
    ab = synthetic.cell_abundances(2000, np.random.default_rng(3))
    ads = []  # no stain scaling: only the spectra differ between the batches
    for i, b in enumerate(["A", "B"]):
        sp = panel_spectra_of(true_emb if b == "B" else np.zeros(3))
        x = ab @ sp + 15 * np.random.default_rng(i).normal(size=ab.shape)
        ads.append(synthetic.make_adata(x, CH, batch=b, uid=f"ref_{b}", type="ref"))
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    pem = PanelEmbeddingModule.from_data(make_panelconfig_csv(tmp_path), moe, pd.Series("batch", index=PANEL_COMPONENTS.index),
                                         anchor_bidx={"batch": [0]})
    tr = SpectralFitTransformer.from_data(moe, pem, anchor_bidx={"batch": [0]}, arcsinh_cofactor=1500)
    ds = make_dataset(ads, moe, device=device, sN_cells=1000)
    m = OverlapFitModel(moe, tr, dataset=ds).to(device)
    torch.manual_seed(0); np.random.seed(0)
    m.fit(Niter=200, lr=5e-2)
    emb = pem.get_learned_embedding()["batch"]
    assert np.allclose(emb["A"].to_numpy(), 0.)
    assert np.allclose(emb["B"].to_numpy(), true_emb[:2], atol=.3), emb
