"""cytovanni.ref.panel: PanelEmbeddingModule (trainable embedding per batch) and FixedPanelModule."""
import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from synthetic import USE_SQUARE, make_panelconfig_pca
from cytovanni.exceptions import OverlapStandardisationException
from cytovanni.ref import PanelModule, PanelEmbeddingModule, FixedPanelModule, MultiOrdinalEncoder
from cytovanni.torch import torch_invert_spectra

COMPS = ["FL1_PC0", "FL1_PC1", "FL2_PC0", "FL3_PC0"]


def make_moe():
    meta = pd.DataFrame({"day": ["d0", "d0", "d1", "d1"], "plate": ["p0", "p1", "p2", "p3"]})
    return MultiOrdinalEncoder.from_data(meta, ["day", "plate"])


def make_pem(batch_order=("day", "plate"), **kw):
    pc = make_panelconfig_pca()
    kcb = pd.Series(["day", "day", "day", "plate"], index=COMPS)
    return PanelEmbeddingModule.from_data(pc, make_moe(), kcb, batch_order=list(batch_order) if batch_order else None, **kw)


IDX = {"day": torch.tensor([0, 0, 1, 1]), "plate": torch.tensor([0, 1, 2, 3])}


# ------------------------------------------------------------------ construction
def test_from_data_attributes():
    pem = make_pem()
    assert pem.TRAINED_EMBEDDING is True
    assert pem.batch_keys == ["day", "plate"]
    assert list(pem.stains) == synthetic.STAINS and list(pem.channels) == synthetic.CHANNELS_FLUORESCENCE
    assert list(pem.panelmodule_components) == COMPS
    assert pem.param_embedding["day"].shape == (4, 2) and pem.param_embedding["plate"].shape == (4, 4)


def test_masks_without_batch_order():
    pem = make_pem(batch_order=None)
    assert pem.param_embedding_mask["day"][:, 0].tolist() == [1., 1., 1., 0.]
    assert pem.param_embedding_mask["plate"][:, 0].tolist() == [0., 0., 0., 1.]


def test_masks_with_batch_order_coarse_key_gets_later_components():
    pem = make_pem()
    assert pem.param_embedding_mask["day"][:, 0].tolist() == [1., 1., 1., 1.]
    assert pem.param_embedding_mask["plate"][:, 0].tolist() == [0., 0., 0., 1.]


def test_component_order_follows_panel_not_series():
    pc = make_panelconfig_pca()
    kcb = pd.Series(["plate", "day", "day", "day"], index=COMPS[::-1])  # FL3_PC0 -> plate, given reversed
    pem = PanelEmbeddingModule.from_data(pc, make_moe(), kcb)
    assert list(pem.key_comp_batch.index) == COMPS
    assert pem.param_embedding_mask["plate"][:, 0].tolist() == [0., 0., 0., 1.]


def test_missing_key_in_moe_raises():
    kcb = pd.Series(["day", "day", "day", "week"], index=COMPS)
    with pytest.raises(OverlapStandardisationException):
        PanelEmbeddingModule.from_data(make_panelconfig_pca(), make_moe(), kcb)


def test_missing_key_in_batch_order_raises():
    with pytest.raises(OverlapStandardisationException):
        make_pem(batch_order=("day",))


def test_missing_component_raises_keyerror():
    kcb = pd.Series(["day", "day", "day"], index=COMPS[:3])
    with pytest.raises(KeyError):
        PanelEmbeddingModule.from_data(make_panelconfig_pca(), make_moe(), kcb)


def test_init_emb_summed_embedding_starts_at_init():
    init = np.array([1., 2., -1., .5])
    for order in [("day", "plate"), None]:
        pem = make_pem(batch_order=order, init_emb=init)
        assert torch.allclose(pem(IDX), torch.tensor(init, dtype=torch.float32)[:, None].expand(4, 4))
        # each key part holds only its own components
        assert pem.param_embedding["plate"]()[:3].abs().sum() == 0 and pem.param_embedding["day"]()[3].abs().sum() == 0


def test_fixed_init_from_measured_embedding_nan_to_zero():
    pc = make_panelconfig_pca()
    pc.pca_embedding.loc["run1", "FL2_PC0"] = np.nan
    kcb = pd.Series(["day", "day", "day", "plate"], index=COMPS)
    pem = PanelEmbeddingModule.from_data(pc, make_moe(), kcb, fixed_init="run1")
    expected = pc.pca_embedding.loc["run1"].fillna(0).to_numpy()
    assert torch.allclose(pem(IDX)[:, 0], torch.tensor(expected, dtype=torch.float32).clamp(-6, 6))
    # init_emb has priority over fixed_init
    pem2 = PanelEmbeddingModule.from_data(pc, make_moe(), kcb, fixed_init="run1", init_emb=np.zeros(4))
    assert pem2(IDX).abs().sum() == 0


def test_default_init_is_zero():
    assert make_pem()(IDX).abs().sum() == 0


# ------------------------------------------------------------------ forward, clipping
def test_forward_sums_parts_of_the_event_batches():
    pem = make_pem()
    pem.param_embedding["day"].param_trainable.data = torch.tensor([[1., 2.]] * 4)
    pem.param_embedding["plate"].param_trainable.data = torch.tensor([[10., 20., 30., 40.]] * 4)
    out = pem(IDX)
    assert out.shape == (4, 4)  # (component, events)
    # components 0-2 only "day"; component 3 "day" (batch_order) + "plate"
    assert out[0].tolist() == [1., 1., 2., 2.]
    assert torch.allclose(out[3], torch.tensor([6., 6., 6., 6.]))  # 1+10, 1+20, ... clipped at 6


def test_max_abs_embedding_clips_value_not_parameter():
    pem = make_pem(init_emb=np.array([10., 0., -8., 3.]))
    out = pem(IDX)
    assert out[:, 0].tolist() == [6., 0., -6., 3.]
    assert pem.param_embedding["day"]().max().item() == 10.  # stored value exceeds the limit
    unl = make_pem(init_emb=np.array([10., 0., -8., 3.]), max_abs_embedding=None)
    assert unl(IDX)[:, 0].tolist() == [10., 0., -8., 3.]


def test_clip_is_straight_through():
    pem = make_pem(init_emb=np.array([10., 0., 0., 0.]))
    pem(IDX)[0].sum().backward()
    assert pem.param_embedding["day"].param_trainable.grad[0].tolist() == [2., 2.]


def test_forward_mask_unused():
    pem = make_pem(init_emb=np.array([1., 2., 3., 4.]))
    assert pem(IDX, mask_unused=True)[1].abs().sum() == 0
    assert pem(IDX)[1, 0] == 2.


def test_get_embedding_part_masked():
    pem = make_pem(batch_order=None)
    pem.param_embedding["plate"].param_trainable.data[:] = 1.
    assert pem.get_embedding_part("plate")[:3].abs().sum() == 0
    assert pem.get_embedding_part("plate", masked=False)[:3].sum() == 12


# ------------------------------------------------------------------ anchors, extend, freeze
def _train(pem, target, n=30):
    opt = torch.optim.Adam([p for p in pem.parameters() if p.requires_grad], lr=.1)
    for _ in range(n):
        opt.zero_grad()
        ((pem(IDX) - target) ** 2).sum().backward()
        opt.step()


def test_anchor_batches_frozen_during_training():
    pem = make_pem(anchor_bidx={"day": [0], "other": [1]}, init_emb=np.array([.5, 0., 0., 0.]))
    _train(pem, torch.full((4, 4), 2.), n=150)
    out = pem(IDX).detach()
    assert out[0, 0] == .5 and out[0, 1] == .5  # day 0 anchored
    assert abs(out[0, 2].item() - 2.) < .1


def test_anchor_batches_with_numpy_indices_from_moe():
    moe = make_moe()
    pem = make_pem(anchor_bidx=moe.transform_dict_simple({"day": ["d1"]}))
    assert pem.param_embedding["day"].param_mask_trainable[:, 1].sum() == 0
    assert pem.param_embedding["day"].param_mask_trainable[:, 0].sum() == 4


def test_freeze_all():
    pem = make_pem()
    pem.freeze()
    for key in pem.batch_keys:
        assert pem.param_embedding[key].param_mask_trainable.sum() == 0


def test_extend_adds_batches_at_init_and_freezes_old():
    pem = make_pem(init_emb=np.array([.5, 0., 0., .25]))
    _train(pem, torch.full((4, 4), 1.), n=10)
    before = pem(IDX).detach().clone()
    moe = pem.moe
    new = moe.extend(pd.DataFrame({"day": ["d2"], "plate": ["p4"]}))
    pem.extend(moe, new)
    assert pem.param_embedding["day"].shape == (4, 3) and pem.param_embedding["plate"].shape == (4, 5)
    assert torch.allclose(pem(IDX), before)
    idx_new = {"day": torch.tensor([2]), "plate": torch.tensor([4])}
    assert torch.allclose(pem(idx_new)[:, 0], torch.tensor([.5, 0., 0., .25]))
    assert pem.param_embedding["day"].param_mask_trainable[:, :2].sum() == 0


def test_requires_grad_toggle_returns_self():
    """R2-E1: requires_grad_ returns self; masks never trainable."""
    pem = make_pem()
    assert pem.requires_grad_(False) is pem
    assert not any(p.requires_grad for p in pem.parameters())
    pem.requires_grad_(True)
    trainable = [n for n, p in pem.named_parameters() if p.requires_grad]
    assert trainable and all("param_trainable" in n for n in trainable)


# ------------------------------------------------------------------ spectra, unmixing, L1
def test_get_spectra_and_inverse_shapes():
    pem = make_pem(init_emb=np.array([1., 0., -1., .5]))
    S = pem.get_spectra(IDX)
    assert S.shape == (4, 3, 3)
    assert torch.allclose(S[0], pem.panelmodule(torch.tensor([1., 0., -1., .5])))
    inv = pem.get_spectra_inv(IDX)
    assert inv.shape == (4, 3, 3) and torch.allclose(inv[0] , torch_invert_spectra(S[0]), atol=1e-5)


def test_get_unmixed_recovers_abundances():
    pem = make_pem(init_emb=np.array([1., 0., -1., .5]))
    ab = torch.rand(4, 3) * 1e4
    S = pem.get_spectra(IDX)
    x = torch.einsum("es, esc -> ec", ab, S)
    assert torch.allclose(pem.get_unmixed(x, IDX), ab, rtol=1e-3, atol=1e-1)


def test_get_freeparam_unmix_gradient_includes_unused():
    pem = make_pem(init_emb=np.array([1., 3., -1., .5]))
    params, f = pem.get_freeparam_unmix(IDX)
    emb = params["embedding"]
    assert emb.is_leaf and emb.requires_grad and emb.shape == (4, 4)
    assert emb[1].abs().sum() == 0  # unused component set to 0
    x = torch.rand(4, 3) * 1e4
    out = f(x)
    assert out.shape == (4, 3)
    out.sum().backward()
    assert (emb.grad[1].abs() > 0).all()


def test_get_L1_value():
    pem = make_pem(batch_order=None)
    pem.param_embedding["day"].param_trainable.data = torch.tensor([[1., -3.], [5., 5.], [2., 0.], [9., 9.]])
    pem.param_embedding["plate"].param_trainable.data = torch.tensor([[7.] * 4] * 3 + [[1., -1., 2., 0.]])
    # day: used + own comps 0, 2 -> mean |.| 2 and 1; FL1_PC1 unused, FL3_PC0 not own; plate: comp 3 -> 1
    assert pem.get_L1().item() == pytest.approx(2. + 1. + 1.)
    assert pem.get_L1().dim() == 0


def test_get_L1_uses_clipped_values():
    pem = make_pem(batch_order=None, init_emb=np.array([10., 0., 0., 0.]))
    assert pem.get_L1().item() == pytest.approx(6.)


def test_get_learned_embedding_layout():
    pem = make_pem(init_emb=np.array([1., 2., 3., 4.]))
    le = pem.get_learned_embedding()
    assert set(le) == {"day", "plate"}
    assert list(le["day"].index) == ["FL1_PC0", "FL2_PC0", "FL3_PC0"] and list(le["day"].columns) == ["d0", "d1"]
    assert list(le["plate"].index) == ["FL3_PC0"] and list(le["plate"].columns) == ["p0", "p1", "p2", "p3"]
    assert le["day"].loc["FL2_PC0", "d1"] == 3. and le["day"].loc["FL3_PC0", "d0"] == 0.


# ------------------------------------------------------------------ export
def test_export_roundtrip_after_training_and_extend(tmp_path):
    pem = make_pem(init_emb=np.array([.5, 0., 0., .25]), anchor_bidx={"day": [0]})
    _train(pem, torch.full((4, 4), 1.), n=5)
    new = pem.moe.extend(pd.DataFrame({"day": ["d2"], "plate": ["p4"]}))
    pem.extend(pem.moe, new)
    torch.save(pem.export(), tmp_path / "pem.pt")
    pem2 = PanelEmbeddingModule.from_exported(torch.load(tmp_path / "pem.pt", weights_only=False))
    idx = {"day": torch.tensor([0, 1, 2]), "plate": torch.tensor([0, 3, 4])}
    assert torch.equal(pem2(idx), pem(idx))
    assert torch.equal(pem2.get_spectra(idx), pem.get_spectra(idx))
    assert pem2.batch_order == pem.batch_order and pem2.max_abs_embedding == 6
    pd.testing.assert_series_equal(pem2.key_comp_batch, pem.key_comp_batch)
    assert pem2.param_embedding["day"].param_mask_trainable.tolist() == pem.param_embedding["day"].param_mask_trainable.tolist()


def test_device_move(device):
    pem = make_pem(init_emb=np.array([.5, 0., 0., .25])).to(device)
    idx = {k: v.to(device) for k, v in IDX.items()}
    assert pem.get_spectra(idx).device.type == device
    x = torch.rand(4, 3, device=device) * 1e3
    assert pem.get_unmixed(x, idx).device.type == device


def test_recovers_embedding_from_unmixing_loss():
    """C: fitting the per-day embedding so that known abundances reproduce the data recovers the true embedding."""
    torch.manual_seed(0)
    true = {"d0": np.array([1., 0., -1., .5]), "d1": np.array([-.5, 0., 2., -1.])}
    pc = make_panelconfig_pca()
    pmod = PanelModule.from_panelconfig(pc)
    kcb = pd.Series("day", index=COMPS)
    pem = PanelEmbeddingModule.from_data(pc, make_moe(), kcb)
    n = 400
    day = torch.randint(0, 2, (n,))
    ab = torch.rand(n, 3) * 1e4
    S_true = torch.stack([pmod(torch.tensor(true["d0"], dtype=torch.float32)),
                          pmod(torch.tensor(true["d1"], dtype=torch.float32))])[day]
    x = torch.einsum("es, esc -> ec", ab, S_true)
    idx = {"day": day, "plate": torch.zeros(n, dtype=torch.long)}
    opt = torch.optim.Adam([p for p in pem.parameters() if p.requires_grad], lr=.05)
    for _ in range(400):
        opt.zero_grad()
        loss = ((pem.get_unmixed(x, idx) - ab) / 1e3).pow(2).mean()
        loss.backward()
        opt.step()
    learned = pem.get_learned_embedding()["day"]
    for d in ["d0", "d1"]:
        used = [c for c in COMPS if USE_SQUARE[c]]
        assert np.allclose(learned.loc[used, d].to_numpy(), pd.Series(true[d], index=COMPS)[used].to_numpy(), atol=.05)


# ------------------------------------------------------------------ FixedPanelModule
@pytest.fixture
def fpm():
    return FixedPanelModule.from_data(synthetic.SPECTRA)


def test_fixed_attributes(fpm):
    assert fpm.TRAINED_EMBEDDING is False
    assert list(fpm.stains) == synthetic.STAINS and list(fpm.channels) == synthetic.CHANNELS_FLUORESCENCE
    assert list(fpm.stain_marker_name) == synthetic.STAINS and list(fpm.markers) == synthetic.STAINS
    assert fpm.batch_keys == [] and list(fpm.panelmodule_components) == []
    assert not fpm.param_spectra.requires_grad


def test_fixed_spectra_and_unmixing(fpm):
    S = fpm.get_spectra(IDX)
    assert S.shape == (3, 3) and torch.allclose(S, torch.tensor(synthetic.SPECTRA.to_numpy(), dtype=torch.float32))
    assert fpm.get_spectra_inv(IDX).shape == (3, 3)
    ab = torch.rand(5, 3) * 1e4
    assert torch.allclose(fpm.get_unmixed(ab @ S, IDX), ab, rtol=1e-4, atol=1e-1)


def test_fixed_spectral_panel():
    sp = pd.DataFrame([[1., .4, .1, .05], [.1, .3, 1., .5]], index=["A", "B"], columns=["c0", "c1", "c2", "c3"])
    fpm = FixedPanelModule.from_data(sp)
    ab = torch.rand(5, 2) * 1e4
    assert torch.allclose(fpm.get_unmixed(ab @ fpm.get_spectra(None), None), ab, rtol=1e-4, atol=1e-1)
    assert fpm.get_spectra_inv(None).shape == (4, 2)


def test_fixed_interface_noops(fpm):
    """R2-E1: requires_grad_ returns self (it used the undefined `pem`)."""
    assert fpm.requires_grad_(True) is fpm and not fpm.param_spectra.requires_grad
    assert fpm.anchor_batches({"day": [0]}) is None and fpm.freeze() is None and fpm.extend(None, {}) is None
    assert fpm(IDX).numel() == 0
    assert fpm.get_L1().item() == 0.
    assert fpm.get_learned_embedding() == {}
    with pytest.raises(NotImplementedError):
        fpm.get_freeparam_unmix(IDX)


def test_fixed_export_roundtrip(fpm, tmp_path):
    torch.save(fpm.export(), tmp_path / "f.pt")
    ex = torch.load(tmp_path / "f.pt", weights_only=False)
    assert ex["class"] == "fixed"
    f2 = FixedPanelModule.from_exported(ex)
    pd.testing.assert_frame_equal(f2.panel_spectra, synthetic.SPECTRA, check_names=False)
    assert torch.equal(f2.get_spectra(None), fpm.get_spectra(None))
