"""BD FACSDiscover S8 helpers (cytovanni.io.S8): spillover, gating strategy from metadata, image mask features."""
import json

import anndata as ad
import flowkit as fk
import flowutils
import numpy as np
import pandas as pd
import pytest

from cytovanni.io import load_S8_metadata_gatingstrategy, read_spill_S8, writefcs
from cytovanni.io import S8


def s8_file(tmp_path, name="s8.fcs", cyt="FACSDiscover S8", gates=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"FSC-A": rng.uniform(0, 100, n), "SSC-A": rng.uniform(0, 100, n), "FSC-H": rng.uniform(0, 100, n),
                       "FL1-A": rng.uniform(0, 100, n), "FL2-A": rng.uniform(0, 100, n), "FL3-A": rng.uniform(0, 100, n)})
    meta = {"spill": "3,FL1-A,FL2-A,FL3-A,1,0.1,0.05,0.2,1,0.3,0.01,0.02,1",
            "bdspectral probes": "FITC-A (SW-unmix),PE-A (SW-unmix)"}
    if cyt is not None:
        meta["cyt"] = cyt
    if gates is not None:
        meta["bdchorusdatarecord"] = json.dumps({"RecordingConfiguration": {"AnalysisModel": {"Gates": gates}}})
    p = str(tmp_path / name)
    writefcs(p, df, add_metadata=meta)
    return p, df


# ------------------------------------------------------------------------------------------------ spillover
def test_read_spill_S8_orientation(tmp_path):
    p, _ = s8_file(tmp_path)
    s = read_spill_S8(p)
    assert s.index.tolist() == ["FL1-A", "FL2-A", "FL3-A"] and s.columns.tolist() == ["FITC", "PE"]  # (channel, dye)
    # column j is row j of $SPILL; rows beyond the listed probes are dropped
    assert np.allclose(s["FITC"], [1, .1, .05]) and np.allclose(s["PE"], [.2, 1, .3])


def test_read_spill_S8_safe_check(tmp_path):
    p, _ = s8_file(tmp_path, cyt="FACSymphony A3")
    with pytest.raises(ValueError):
        read_spill_S8(p)
    assert read_spill_S8(p, safe=False).shape == (3, 2)
    p2, _ = s8_file(tmp_path, name="nocyt.fcs", cyt=None)
    with pytest.raises(ValueError):
        read_spill_S8(p2)


# ------------------------------------------------------------------------------------------------ gating strategy
def scatter_par(m):
    return {"ParameterKind": "Scatter", "MeasurementId": m, "Measurement": "A"}


SQ = lambda lo, hi: [{"X": lo, "Y": lo}, {"X": hi, "Y": lo}, {"X": hi, "Y": hi}, {"X": lo, "Y": hi}]
GATES = [  # children listed before their parents, and one entry without axes
    {"ParentPopulationId": "p2", "GateId": "g3", "Children": [{"PopulationId": "p3", "Name": "FITC+"}],
     "Parameters": [{"ParameterKind": "Color", "Fluorochrome": "FL1", "Measurement": "A"},
                    {"ParameterKind": "Color", "Fluorochrome": "FL2", "Measurement": "A"}],
     "Vertices": SQ(50, 100)},
    {"ParentPopulationId": "p1", "GateId": "g2", "Children": [{"PopulationId": "p2", "Name": "Singlets"}],
     "Parameters": [scatter_par("FSC"), {"ParameterKind": "Scatter", "MeasurementId": "FSC", "Measurement": "H"}],
     "Vertices": SQ(10, 90)},
    {"ParentPopulationId": "0-1", "GateId": "g1", "Children": [{"PopulationId": "p1", "Name": "Cells"}],
     "Parameters": [scatter_par("FSC"), scatter_par("SSC")], "Vertices": SQ(20, 80)},
    {"ParentPopulationId": "0-1", "GateId": "g0", "Children": [{"PopulationId": "p0", "Name": "All"}],
     "Parameters": [], "Vertices": []},
]


def test_gate_parses_entry():
    g = S8.Gate(GATES[0])
    assert g.id == "p3" and g.id_gate == "g3" and g.id_parent == "p2" and g.name == "FITC+"
    assert (g.key_x, g.key_y) == ("FL1-A", "FL2-A")
    assert g.vertices.shape == (4, 2) and np.allclose(g.vertices[1], [100, 50])
    root = S8.Gate(GATES[2])
    assert root.id_parent == "root" and (root.key_x, root.key_y) == ("FSC-A", "SSC-A")
    other = S8.Gate({**GATES[2], "Parameters": [{"ParameterKind": "Time", "MeasurementId": "Time"}, scatter_par("SSC")]})
    assert other.key_x == "Time"
    assert "child of 'root'" in repr(root)


def test_load_and_sort_gates():
    gates = S8.load_gates(GATES)
    assert len(gates) == 3  # the entry without parameters is skipped
    order = [g.name for g in S8.sort_gates(gates)]
    assert order.index("Cells") < order.index("Singlets") < order.index("FITC+")


def test_gate_parent_names_and_order():
    gates = S8.sort_gates(S8.load_gates(GATES))
    key = pd.Series([g.name for g in gates], index=[g.id for g in gates])
    for g in gates:
        g.add_parent_name(key)
    connect = pd.Series([g.name_parent for g in gates], index=[g.name for g in gates])
    for g in gates:
        g.add_order(connect)
    by = {g.name: g for g in gates}
    assert by["Cells"].gate_order == ("root",)
    assert by["Singlets"].gate_order == ("root", "Cells")
    assert by["FITC+"].gate_order == ("root", "Cells", "Singlets")
    assert "child of 'Singlets'" in repr(by["FITC+"])
    gate, path = by["FITC+"].to_fk()
    assert isinstance(gate, fk.gates.PolygonGate) and path == ("root", "Cells", "Singlets")


def test_load_gates_fromfile(tmp_path):
    p, _ = s8_file(tmp_path, gates=GATES)
    names = [g.name for g in S8.load_gates_fromfile(p)]
    assert names.index("Cells") < names.index("Singlets") < names.index("FITC+") and len(names) == 3


def test_load_S8_metadata_gatingstrategy_gates_like_polygons(tmp_path):
    p, df = s8_file(tmp_path, gates=GATES)
    strat = load_S8_metadata_gatingstrategy(p)
    assert isinstance(strat, fk.GatingStrategy)
    res = strat.gate_sample(fk.Sample(p))
    sq = lambda lo, hi: np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]], dtype=float)
    cells = flowutils.gating.points_in_polygon(sq(20, 80), df[["FSC-A", "SSC-A"]].to_numpy())
    singlets = cells & flowutils.gating.points_in_polygon(sq(10, 90), df[["FSC-A", "FSC-H"]].to_numpy())
    pos = singlets & flowutils.gating.points_in_polygon(sq(50, 100), df[["FL1-A", "FL2-A"]].to_numpy())
    assert np.array_equal(res.get_gate_membership("Cells"), cells)
    assert np.array_equal(res.get_gate_membership("Singlets"), singlets)
    assert np.array_equal(res.get_gate_membership("FITC+"), pos)
    assert 0 < pos.sum() < singlets.sum() < cells.sum() < len(df)


def test_load_S8_metadata_gatingstrategy_verbose(tmp_path, capsys):
    p, _ = s8_file(tmp_path, gates=GATES)
    load_S8_metadata_gatingstrategy(p, verbose=True)
    out = capsys.readouterr().out
    assert "Cells" in out and "FITC+" in out


# ------------------------------------------------------------------------------------------------ sizes
def test_size_conversions_of_a_sphere():
    r = 1.7
    a = np.pi * r ** 2
    assert np.isclose(S8.S8_size_to_diameter(a), 2 * r)
    assert np.isclose(S8.S8_size_to_volume(a), 4 / 3 * np.pi * r ** 3)
    assert np.isclose(S8.S8_size_to_surface(a), 4 * np.pi * r ** 2)
    assert np.isclose(S8.S8_PIXEL_AREA, S8.S8_PIXEL_WIDTH ** 2 * S8.S8_PIXEL_HEIGHT_RATIO)


# ------------------------------------------------------------------------------------------------ eccentricity
def padded(m, pad=2):
    return np.pad(m, pad)


def disk(r=6):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return padded(xx ** 2 + yy ** 2 <= r ** 2)


def test_eccentricity_circle_and_lines():
    e, _ = S8.calculate_single_eccentricity_angle(disk(), pixel_aspect_ratio=1.)
    assert e < .1
    horiz = padded(np.ones((2, 15), bool))  # along the columns
    vert = padded(np.ones((15, 2), bool))   # along the rows
    eh, ah = S8.calculate_single_eccentricity_angle(horiz, pixel_aspect_ratio=1.)
    ev, av = S8.calculate_single_eccentricity_angle(vert, pixel_aspect_ratio=1.)
    assert eh > .98 and ev > .98
    assert np.isclose(abs(ah), 90) and np.isclose(av, 0)


def test_eccentricity_angle_sign():
    """Positive when the major axis runs from low (row, column) to high (row, column)."""
    diag = padded(np.eye(12, dtype=bool) | np.eye(12, k=1, dtype=bool))
    anti = diag[:, ::-1]
    _, a = S8.calculate_single_eccentricity_angle(diag, pixel_aspect_ratio=1.)
    _, b = S8.calculate_single_eccentricity_angle(anti, pixel_aspect_ratio=1.)
    assert np.isclose(a, 45, atol=3) and np.isclose(b, -45, atol=3)


def test_eccentricity_pixel_aspect_correction():
    rect = padded(np.ones((13, 10), bool))  # 13 rows of pixels 1/1.3 as tall as wide: a square in physical units
    e1, _ = S8.calculate_single_eccentricity_angle(rect, pixel_aspect_ratio=1.)
    e2, _ = S8.calculate_single_eccentricity_angle(rect, pixel_aspect_ratio=1 / 1.3)
    assert e1 > .5 and e2 < .1
    e_default, _ = S8.calculate_single_eccentricity_angle(rect)
    assert np.isclose(e_default, e2)


def test_eccentricity_boundary_and_empty():
    m = np.zeros((10, 10), bool)
    assert S8.calculate_single_eccentricity_angle(m) == (0., 0.)
    for sl in [(0, slice(3, 5)), (-1, slice(3, 5)), (slice(3, 5), 0), (slice(3, 5), -1)]:
        mm = m.copy(); mm[sl] = True
        e, a = S8.calculate_single_eccentricity_angle(mm)
        assert np.isnan(e) and np.isnan(a)


def test_eccentricity_single_pixel_and_round_mask():
    """Documented: eccentricity 0 for a single pixel (no extent); the angle of a round mask is 90 (undefined)."""
    m = np.zeros((5, 5), bool); m[2, 2] = True
    assert S8.calculate_single_eccentricity_angle(m) == (0., 90.)
    e, a = S8.calculate_batched_eccentricity_angle([m])
    assert e[0] == 0. and a[0] == 90.


def test_batched_equals_single_for_oblique_masks():
    diag = padded(np.eye(12, dtype=bool) | np.eye(12, k=1, dtype=bool))
    masks = [diag, diag[:, ::-1], padded(np.ones((3, 9), bool)), disk()]
    masks = [np.pad(m, ((0, 0), (0, 20 - m.shape[1]))) for m in masks]  # one batch: same width
    single = np.array([S8.calculate_single_eccentricity_angle(m) for m in masks])
    e, a = S8.calculate_batched_eccentricity_angle(masks)
    assert np.allclose(single, np.vstack([e, a]).T)
    assert a[1] < 0 < a[0]


def random_masks(rng, n=150, w=20):
    masks = []
    for k in range(n):
        h = rng.integers(6, 30)
        m = np.zeros((h, w), bool)
        r0, c0 = rng.integers(0, h - 2), rng.integers(0, w - 2)
        m[r0:r0 + rng.integers(1, 8), c0:c0 + rng.integers(1, 8)] = True
        if k % 7 == 0:
            m[:] = False
        masks.append(m)
    return masks


def test_batched_equals_single(rng):
    """R2-D4: batched == single for masks of different heights, incl. cells cut off at their own bottom edge."""
    masks = random_masks(rng)
    single = np.array([S8.calculate_single_eccentricity_angle(m) for m in masks])
    e, a = S8.calculate_batched_eccentricity_angle(masks)
    assert e.shape == a.shape == (len(masks),)
    assert np.allclose(single, np.vstack([e, a]).T, equal_nan=True)
    cut_bottom = [i for i, m in enumerate(masks) if m[-1].any() and not (m[0].any() or m[:, 0].any() or m[:, -1].any())]
    assert cut_bottom and np.isnan(e[cut_bottom]).all()


def test_batched_bottom_edge_uses_own_height():
    """R2-D4: a short mask touching its own last row is NaN even if another mask in the batch is taller."""
    short = np.zeros((6, 10), bool); short[3:6, 3:6] = True
    tall = np.zeros((20, 10), bool); tall[5:10, 3:6] = True
    e, a = S8.calculate_batched_eccentricity_angle([short, tall])
    assert np.isnan(e[0]) and np.isnan(a[0]) and np.isfinite(e[1])


def test_pad_masks():
    out = S8._pad_masks([np.ones((2, 3), bool), np.ones((4, 3), bool)])
    assert out.shape == (2, 4, 3) and out[0, 2:].sum() == 0 and out[1].all()


# ------------------------------------------------------------------------------------------------ mask features
class FakeImageContainer:
    def __init__(self, masks):
        self.masks = masks
        self.requested = None

    def load_masks_batched(self, index):
        self.requested = list(index)
        return [self.masks[i] for i in index]


def test_add_S8_singlet_maskfeatures(rng):
    """R2-F3: works with any duck-typed image container that has `load_masks_batched(index)`; features as documented."""
    masks = {str(i): m for i, m in enumerate(random_masks(rng, n=12))}
    obs = pd.DataFrame({"seg_cell_totsize": rng.uniform(50, 200, 12), "seg_cell_count": [1] * 10 + [2, 0]},
                       index=[str(i) for i in range(12)])
    a = ad.AnnData(obs=obs)
    imgcont = FakeImageContainer(masks)
    S8.add_S8_singlet_maskfeatures(a, imgcont)
    assert imgcont.requested == list(a.obs.index)
    o = a.obs
    single = o["seg_cell_count"] == 1
    cs = o["seg_cell_totsize"] * S8.S8_PIXEL_AREA
    assert np.allclose(o.loc[single, "cell_crosssection"], cs[single])
    assert np.allclose(o.loc[single, "cell_diameter"], S8.S8_size_to_diameter(cs[single]))
    assert np.allclose(o.loc[single, "cell_volume"], S8.S8_size_to_volume(cs[single]))
    assert np.allclose(o.loc[single, "cell_surface"], S8.S8_size_to_surface(cs[single]))
    for c in ["cell_crosssection", "cell_volume", "cell_diameter", "cell_surface", "cell_eccentricity", "cell_deformangle"]:
        assert o.loc[~single, c].isna().all(), c
    e, ang = S8.calculate_batched_eccentricity_angle([masks[i] for i in o.index])
    assert np.allclose(o.loc[single, "cell_eccentricity"], e[single.to_numpy()], equal_nan=True)
    assert np.allclose(o.loc[single, "cell_deformangle"], ang[single.to_numpy()], equal_nan=True)
