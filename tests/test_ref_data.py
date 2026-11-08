"""Tests of cytovanni.ref.data: graph helpers, ConnectionSampler, SampleOverlapDataset, adatas_process_anchors."""
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

import synthetic
from cytovanni.ref import MultiOrdinalEncoder, SampleOverlapDataset, adatas_process_anchors
from cytovanni.ref.data import (ConnectionSampler, generate_unique_connections, graph_shortest_path,
                                idx_to_connections, unique_connections)

CH = synthetic.CHANNELS_FLUORESCENCE


def batch_pairs(cs, idx_batch):
    """Set of (batch, batch) pairs (sorted) that have a connection."""
    idx_batch = np.asarray(idx_batch)
    return sorted({tuple(sorted((int(idx_batch[a]), int(idx_batch[b])))) for a, b in zip(*cs.connections.nonzero())})


def chain(nbatch=4):
    """Batches 0..nbatch-1 in a chain: batch k and k+1 share one sample; batch 0 is the anchor batch."""
    ids, bat = [], []
    for k in range(nbatch - 1):
        ids += [k, k]
        bat += [k, k + 1]
    anchor = [b == 0 for b in bat]
    return np.asarray(ids), np.asarray(bat), np.asarray(anchor)


# ------------------------------------------------------------------ graph helpers
def test_graph_shortest_path_counts_edges_inf_without_path():
    g = np.zeros((4, 4))
    g[0, 1] = g[1, 0] = g[1, 2] = g[2, 1] = 1
    d = graph_shortest_path(g)
    assert d[0, 2] == 2 and d[2, 0] == 2 and d[0, 1] == 1
    assert np.isinf(d[0, 3]) and np.isinf(d[3, 0])
    assert np.all(np.diag(d) == 1)
    assert np.all(np.diag(graph_shortest_path(g, filldiag=False))[:3] == 0)


def test_generate_unique_connections_skips_repeats_and_all_equal():
    out = list(generate_unique_connections([0, 1, 2], [0, 1, 2]))
    assert out == [(0, 1), (0, 2), (1, 2)]


def test_idx_to_connections_and_unique_connections_merge_directions():
    c = idx_to_connections([0, 2, 1, 0], [1, 1, 0, 1])
    assert c.shape == (2, 4)
    assert unique_connections(c).tolist() == [[0, 1], [1, 2]]


# ------------------------------------------------------------------ ConnectionSampler
def test_connection_sampler_without_anchors_uses_all_cross_batch_pairs_and_warns():
    with pytest.warns(UserWarning, match="No anchor"):
        cs = ConnectionSampler([0, 0, 0, 1, 1], [0, 1, 1, 0, 2])
    # same id in the same batch (samples 1, 2) is blocked
    pairs = sorted({tuple(sorted(p)) for p in zip(*cs.connections.nonzero())})
    assert pairs == [(0, 1), (0, 2), (3, 4)]
    assert np.isnan(cs.idx_anchor_distance).all()


def test_connection_sampler_block_within_batch_false_connects_same_batch():
    with pytest.warns(UserWarning):
        cs = ConnectionSampler([0, 0], [0, 0], block_within_batch=False)
    assert cs.connections[0, 1] and cs.connections[1, 0] and not cs.connections[0, 0]


def test_connection_sampler_anchor_distances_on_chain():
    ids, bat, anc = chain(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ConnectionSampler(ids, bat, is_anchor=anc, max_anchor_dist=3)
    assert cs.batch_anchor_distance.tolist() == [0, 1, 2, 3]
    assert cs.idx_anchor_distance.tolist() == cs.batch_anchor_distance[bat].tolist()
    assert cs.find_nontraining_batches().tolist() == []


@pytest.mark.parametrize("max_dist, trained_pairs, untrained", [(1, [(0, 1)], [2, 3]),
                                                               (2, [(0, 1), (1, 2)], [3]),
                                                               (3, [(0, 1), (1, 2), (2, 3)], [])])
def test_connection_sampler_max_anchor_dist_limits_training(max_dist, trained_pairs, untrained):
    """R2b-G2-06: batches up to `max_anchor_dist` from an anchor are trained; 1 = only anchor connections."""
    ids, bat, anc = chain(4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cs = ConnectionSampler(ids, bat, is_anchor=anc, max_anchor_dist=max_dist)
    assert batch_pairs(cs, bat) == trained_pairs
    assert cs.find_nontraining_batches().tolist() == untrained
    beyond = [x for x in w if "further than max_anchor_dist" in str(x.message)]
    assert len(beyond) == (1 if untrained else 0)


def test_connection_sampler_max_anchor_dist_below_one_raises():
    ids, bat, anc = chain(3)
    with pytest.raises(ValueError, match="at least 1"):
        ConnectionSampler(ids, bat, is_anchor=anc, max_anchor_dist=0)


def test_connection_sampler_unreachable_batches_are_not_anchors():
    """R2-C13: batches without a path to an anchor batch get distance inf, no connections and a warning."""
    with pytest.warns(UserWarning, match="no connection to any anchor"):
        cs = ConnectionSampler([0, 0, 1, 1], [0, 1, 2, 3], is_anchor=[True, False, False, False], train_higher=False)
    assert np.isinf(cs.batch_anchor_distance[[2, 3]]).all()
    assert cs.find_nontraining_batches().tolist() == [2, 3]
    assert batch_pairs(cs, [0, 1, 2, 3]) == [(0, 1)]


def test_connection_sampler_without_any_connection_raises():
    """R2-E8: nothing to train on raises a clear ValueError."""
    with pytest.raises(ValueError, match="No connections"):
        ConnectionSampler([0, 1], [0, 1], is_anchor=[True, False])


def test_connection_sampler_connect_train_within_anchors():
    kw = dict(is_anchor=[True, True, False], max_anchor_dist=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ConnectionSampler([0, 0, 0], [0, 0, 1], connect_train_within_anchors=True, **kw)
        cs0 = ConnectionSampler([0, 0, 0], [0, 0, 1], **kw)
    assert cs.connections[0, 1] and not cs0.connections[0, 1]


def test_connection_sampler_sample_is_unique_sorted_and_respects_nmax():
    # sample 0 has 5 partners, every other sample has 1
    ids = [0] * 6
    bat = [0, 1, 2, 3, 4, 5]
    with pytest.warns(UserWarning):
        cs = ConnectionSampler(ids, bat, Nmax_persample=2)
    assert len(cs.connections_sampledct) == 6  # everyone has 5 partners > 2
    np.random.seed(1)
    conn, ctrain = cs.sample()
    assert conn.shape[0] == 2 and ctrain.shape == conn.shape and ctrain.all()
    assert np.all(conn[0] < conn[1])
    assert unique_connections(conn).shape == conn.shape
    drawn = cs.sample_from_sampledct()
    assert drawn.shape == (2, 6 * 2)
    assert all((drawn[0] == k).sum() == 2 for k in range(6))


def test_connection_sampler_few_partners_are_fixed_and_sampled_dict_empty():
    ids, bat, anc = chain(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ConnectionSampler(ids, bat, is_anchor=anc, max_anchor_dist=2, Nmax_persample=2)
    assert cs.connections_sampledct == {}
    assert cs.sample_from_sampledct().shape == (2, 0)
    conn, _ = cs.sample()
    assert np.array_equal(conn, cs.connections_fixed)


def test_connection_sampler_train_higher_false_trains_only_the_far_side():
    ids, bat, anc = chain(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = ConnectionSampler(ids, bat, is_anchor=anc, max_anchor_dist=2, train_higher=False)
    df = cs.get_df_sample_connections()
    for _, r in df.iterrows():
        if r["adist_left"] == r["adist_right"] == 0:
            assert r["train_left"] and r["train_right"]
        else:
            assert r["train_left"] == (r["adist_left"] > r["adist_right"])
            assert r["train_right"] == (r["adist_right"] > r["adist_left"])
    assert list(df.columns) == ["train_left", "train_right", "adist_left", "adist_right", "idx_left", "idx_right"]


def test_get_df_sample_connections_without_anchors():
    """R2b-G2-16: without anchors the table has NaN anchor distances (was AttributeError)."""
    with pytest.warns(UserWarning):
        cs = ConnectionSampler([0, 0, 1, 1], [0, 1, 0, 1], is_anchor=None)
    df = cs.get_df_sample_connections()
    assert len(df) == 2
    assert df[["adist_left", "adist_right"]].isna().all().all()
    assert df[["train_left", "train_right"]].all().all()


# ------------------------------------------------------------------ SampleOverlapDataset
def overlap_adatas(n=1200, extra=None):
    """Samples s1, s2 measured in batches A and B (calibrated scale); uids u0..u3."""
    spec = [("A", "s1"), ("A", "s2"), ("B", "s1"), ("B", "s2")]
    ads = []
    for i, (b, s) in enumerate(spec):
        ab = synthetic.cell_abundances(n, np.random.default_rng(100 + int(s[1])))
        ad = synthetic.cell_adata(b, n=n, seed=i, abundances=ab, uid=f"u{i}")
        ad.uns["type"] = s
        ad.uns.update((extra or [{}] * 4)[i])
        ads.append(ad)
    return ads


@pytest.fixture
def ads4():
    return overlap_adatas()


@pytest.fixture
def moe4(ads4):
    return MultiOrdinalEncoder.from_adatas(ads4, ["batch"])


def make_ds(ads, moe, **kw):
    kw = dict(dict(sN_cells=300, min_sample_cells=100), **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SampleOverlapDataset(ads, moe, CH, **kw)


def test_dataset_loads_data_meta_and_split(ads4, moe4):
    ds = make_ds(ads4, moe4, data_isanchor=[True, True, False, False])
    assert len(ds) == 4
    assert ds.data.shape == (4 * 1200, 3) and ds.data.dtype == torch.float32
    assert np.allclose(ds.data[1200:2400].numpy(), ads4[1].layers["calibrated"])
    assert ds.meta["_N"].tolist() == [1200] * 4
    assert ds.data_idx_batch["batch"].tolist() == [0, 0, 1, 1]
    assert ds.data_idx_id.tolist() == [0, 1, 0, 1]
    assert [len(i) for i in ds.data_split_indices] == [1200] * 4


def test_dataset_channel_order_follows_channels_argument(ads4, moe4):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = SampleOverlapDataset(ads4, moe4, CH[::-1], sN_cells=50, min_sample_cells=10)
    assert np.allclose(ds.data[:1200].numpy(), ads4[0].layers["calibrated"][:, ::-1])


def test_dataset_load_obsm(ads4, moe4):
    for ad in ads4:
        ad.obsm["unmx"] = pd.DataFrame(ad.layers["calibrated"] * 2, index=ad.obs.index, columns=["m1", "m2", "m3"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = SampleOverlapDataset(ads4, moe4, ["m3", "m1"], key_layer="unmx", load_obsm=True, sN_cells=50, min_sample_cells=10)
    assert np.allclose(ds.data[:1200].numpy(), ads4[0].layers["calibrated"][:, [2, 0]] * 2)
    sp = ds.make_dummy_spectra()
    assert sp.index.tolist() == sp.columns.tolist() == ["m3", "m1"] and np.array_equal(sp.to_numpy(), np.identity(2))


def test_dataset_too_small_sample_raises(ads4, moe4):
    with pytest.raises(ValueError, match="less than"):
        SampleOverlapDataset(ads4, moe4, CH, min_sample_cells=5000)


def test_dataset_warns_for_samples_without_partner(moe4):
    ads = overlap_adatas()[:3]
    with pytest.warns(UserWarning, match="do not have any other matching"):
        SampleOverlapDataset(ads, moe4, CH, sN_cells=50, min_sample_cells=10, data_isanchor=[True, True, False])


def test_dataset_sample_single_shapes(ads4, moe4, device):
    ds = make_ds(ads4, moe4, device=device, data_isanchor=[True, True, False, False])
    s = ds.sample_single(2)
    assert s["x"].shape == (300, 3) and s["x"].device.type == device
    assert s["idx_batch"]["batch"].shape == (300,) and s["idx_batch"]["batch"].unique().tolist() == [1]
    assert s["idx_id"].shape == (300,) and s["idx_id"].unique().tolist() == [0]
    s1 = ds.sample_single(2, expand_idx=False)
    assert s1["idx_batch"]["batch"].shape == (1,) and s1["idx_id"].shape == (1,)
    assert ds[3]["x"].shape == (300, 3)


def test_dataset_sample_draws_all_samples_and_connections(ads4, moe4, device):
    ds = make_ds(ads4, moe4, device=device, data_isanchor=[True, True, False, False])
    out = ds.sample()
    assert len(out["data"]) == 4
    assert all(d["x"].shape == (300, 3) for d in out["data"])
    pairs = sorted(zip(out["cidx_left"].tolist(), out["cidx_right"].tolist()))
    assert pairs == [(0, 2), (1, 3)]
    assert out["dotr_left"].all() and out["dotr_right"].all()


def test_dataset_sample_takes_all_events_of_small_samples(ads4, moe4):
    ds = make_ds(ads4, moe4, sN_cells=5000)
    assert all(d["x"].shape[0] == 1200 for d in ds.sample()["data"])


def test_dataset_data_on_device_false_keeps_data_on_cpu(ads4, moe4, device):
    ds = make_ds(ads4, moe4, device=device, data_on_device=False)
    assert ds.data.device.type == "cpu"
    out = ds.sample()
    assert out["data"][0]["x"].device.type == device


def test_dataset_covariate_keys_are_loaded_and_sampled():
    """R2b-G2-03: `keys_sample_covariate` that are not batch/id/uid keys are loaded from ``adata.uns``."""
    ads = overlap_adatas(extra=[{"treatment": t} for t in ["x", "y", "x", "y"]])
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    ds = make_ds(ads, moe, keys_sample_covariate=["treatment"], data_isanchor=[True, True, False, False])
    assert ds.meta["treatment"].tolist() == ["x", "y", "x", "y"]
    assert sorted(d["samp_cov"]["treatment"] for d in ds.sample()["data"]) == ["x", "x", "y", "y"]


def test_dataset_key_batch_sample_none_connects_all_same_id_samples(ads4, moe4):
    """R2b-G2-19: without a blocking batch key all same-id measurements are connected, with and without anchors."""
    for anchors in [None, [True, True, False, False]]:
        ds = make_ds(ads4, moe4, key_batch_sample=None, data_isanchor=anchors)
        assert None not in ds.meta.columns
        pairs = sorted({tuple(sorted(p)) for p in zip(*ds.csampler.connections.nonzero())})
        assert pairs == [(0, 2), (1, 3)]


def test_dataset_default_max_anchor_dist_is_2_and_passed_on():
    """R2b-G2-06: SampleOverlapDataset keeps the old hard-coded behaviour by default (max_anchor_dist=2)."""
    assert inspect.signature(SampleOverlapDataset).parameters["max_anchor_dist"].default == 2
    spec = [("b0", "s1"), ("b1", "s1"), ("b1", "s2"), ("b2", "s2"), ("b2", "s3"), ("b3", "s3")]
    ads = []
    for i, (b, s) in enumerate(spec):
        ad = synthetic.make_adata(np.random.default_rng(i).normal(size=(150, 3)), CH, batch=b, type=s, uid=f"u{i}")
        ads.append(ad)
    moe = MultiOrdinalEncoder.from_adatas(ads, ["batch"])
    anchor = [True] + [False] * 5
    for m, untrained in [(None, [3]), (1, [2, 3]), (3, [])]:
        kw = {} if m is None else {"max_anchor_dist": m}
        ds = make_ds(ads, moe, data_isanchor=anchor, sN_cells=100, min_sample_cells=50, **kw)
        assert ds.csampler.find_nontraining_batches().tolist() == untrained


def test_adatas_process_anchors_matches_any_key():
    ads = [synthetic.make_adata(np.zeros((1, 3)), CH, batch=b, plate=p) for b, p in [("A", "p1"), ("B", "p2"), ("C", "p3")]]
    assert adatas_process_anchors(ads, {"batch": ["A"]}).tolist() == [True, False, False]
    assert adatas_process_anchors(ads, {"batch": ["A"], "plate": ["p3"]}).tolist() == [True, False, True]
