"""Tests of cytovanni.ref.model_weight (marker weightings) and cytovanni.ref.model_transformer_smp (scale multipliers)."""
import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.ref import (DefaultMultiplier, DummyWeights, MultiOrdinalEncoder, RareMarkerWeights, ScalingTransformer,
                           UnmxErrorExclusionMultiplier)
from cytovanni.ref.model_transformer_smp import ScaleMultiplier, pick_smp_cls
from cytovanni.ref.model_weight import pick_weight_cls


@pytest.fixture
def transformer():
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"batch": ["A", "B"]}), ["batch"])
    return ScalingTransformer.from_data(moe, synthetic.SPECTRA, scale_stains=True)


# ------------------------------------------------------------------ marker weights
def test_pick_weight_cls():
    assert pick_weight_cls("dummy") is DummyWeights
    assert pick_weight_cls("raremarker") is RareMarkerWeights
    with pytest.raises(NotImplementedError):
        pick_weight_cls("other")


def test_dummy_weights_return_none_and_roundtrip(transformer):
    w = DummyWeights("ignored", key=1)
    assert w(torch.ones(3, 3), {}, {}) is None
    assert w.export() == {}
    assert isinstance(DummyWeights.from_exported({}, transformer), DummyWeights)
    assert DummyWeights.NAME == "dummy"


def test_rare_marker_weights_mean_one_and_max_over_groups(transformer):
    rmw = RareMarkerWeights(transformer, groups=[("FL1", 1e3, 4.), ("FL2", 1e3, 2.)])
    x = torch.tensor([[1e5, 1e5, 0.],   # positive in both groups -> max weight 4
                      [0., 1e5, 0.],    # FL2 only -> 2
                      [0., 0., 1e5],    # none -> 1
                      [0., 0., 0.]])    # none -> 1
    w = rmw(transformer.apply_transformation(x), None, None)
    assert w.shape == (4,)
    assert torch.isclose(w.mean(), torch.tensor(1.))
    assert torch.allclose(w / w[3], torch.tensor([4., 2., 1., 1.]))


def test_rare_marker_weights_fractional_weight():
    """R2-A6: fractional group weights are kept (were truncated to int, 2.5 -> 2)."""
    moe = MultiOrdinalEncoder.from_data(pd.DataFrame({"batch": ["A", "B"]}), ["batch"])
    tr = ScalingTransformer.from_data(moe, synthetic.SPECTRA, scale_stains=True)
    rmw = RareMarkerWeights(tr, groups=[("FL1", 1e3, 2.5)])
    w = rmw(tr.apply_transformation(torch.tensor([[1e5, 0., 0.], [0., 0., 0.]])), None, None)
    assert torch.isclose(w[0] / w[1], torch.tensor(2.5))


def test_rare_marker_weights_below_one_act_as_one(transformer):
    rmw = RareMarkerWeights(transformer, groups=[("FL1", 1e3, .2)])
    w = rmw(transformer.apply_transformation(torch.tensor([[1e5, 0., 0.], [0., 0., 0.]])), None, None)
    assert torch.allclose(w, torch.ones(2))


def test_rare_marker_weights_empty_groups_is_dummy_and_export_roundtrip(transformer):
    assert RareMarkerWeights(transformer)(torch.ones(2, 3), None, None) is None
    rmw = RareMarkerWeights(transformer, groups=[("FL3", 5e3, 3.)])
    rmw2 = RareMarkerWeights.from_exported(rmw.export(), transformer)
    assert rmw2.groups == [("FL3", 5e3, 3.)]
    x = transformer.apply_transformation(torch.tensor([[0., 0., 1e4], [0., 0., 1e3]]))
    assert torch.equal(rmw(x, None, None), rmw2(x, None, None))


def test_requires_grad_overrides_return_self_and_stay_frozen(transformer):
    """R2-E1: the requires_grad_ overrides work for dummy and real groups and return self."""
    mods = [RareMarkerWeights(transformer, groups=[("FL1", 1e3, 2.)]), RareMarkerWeights(transformer),
            UnmxErrorExclusionMultiplier(transformer, groups=[("FL1", 1e4, ["FL2"])]), UnmxErrorExclusionMultiplier(transformer)]
    for m in mods:
        assert m.requires_grad_(True) is m
        assert not any(p.requires_grad for p in m.parameters())


# ------------------------------------------------------------------ scale multipliers
def test_pick_smp_cls():
    assert pick_smp_cls("default") is DefaultMultiplier
    assert pick_smp_cls("unmxerrexclusion") is UnmxErrorExclusionMultiplier
    with pytest.raises(NotImplementedError):
        pick_smp_cls("other")


def test_default_multiplier(transformer):
    m = DefaultMultiplier(1, a=2)
    assert isinstance(m, ScaleMultiplier)
    x, s = torch.rand(4, 3), torch.rand(4, 3)
    assert torch.equal(m(x, s), x * s)
    assert m.export() == {} and isinstance(DefaultMultiplier.from_exported({}, transformer), DefaultMultiplier)


def test_unmx_error_exclusion_blocks_scale_gradient_only_for_excluded_events(transformer):
    m = UnmxErrorExclusionMultiplier(transformer, groups=[("FL1", 1e4, ["FL2", "FL3"])])
    xc = torch.tensor([[2e4, 1., 1.], [1., 1., 1.]], requires_grad=True)
    scale = torch.ones(2, 3, requires_grad=True)
    out = m(xc, scale)
    assert torch.equal(out, xc * scale)
    out.sum().backward()
    # event 0 is above the FL1 cutoff: no gradient for its FL2/FL3 scale; event 1 and FL1 are unaffected
    assert scale.grad.tolist() == [[2e4, 0., 0.], [1., 1., 1.]]
    assert torch.equal(xc.grad, torch.ones(2, 3))


def test_unmx_error_exclusion_dummy_and_export_roundtrip(transformer):
    xc, s = torch.rand(3, 3), torch.rand(3, 3)
    assert torch.equal(UnmxErrorExclusionMultiplier(transformer)(xc, s), xc * s)
    m = UnmxErrorExclusionMultiplier(transformer, groups=[("FL2", 3e3, ["FL1"])])
    m2 = UnmxErrorExclusionMultiplier.from_exported(m.export(), transformer)
    assert m2.groups == m.groups and torch.equal(m2.block_mask, m.block_mask)


def test_unmx_error_exclusion_in_float64(transformer):
    """R2d-D7: the multiplier works after ``.double()`` (mask cast to the block mask's dtype)."""
    m = UnmxErrorExclusionMultiplier(transformer, groups=[("FL1", 1e4, ["FL2"])]).double()
    xc = torch.tensor([[2e4, 1., 1.]], dtype=torch.float64)
    scale = torch.full((1, 3), 2., dtype=torch.float64)
    out = m(xc, scale)
    assert out.dtype == torch.float64 and torch.equal(out, xc * scale)
