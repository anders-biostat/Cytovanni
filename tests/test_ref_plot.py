"""cytovanni.ref.plot: overlays of all measurements of one sample (histograms, NxN)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import synthetic
import cytovanni.ref.plot as rplot
from cytovanni.ref import plot_eval_histogram, plot_eval_NxN
from cytovanni.utils import get_cmap


def _adatas():
    """Four measurements: sample s1 in batches A, B, A(second aliquot); sample s2 in batch C."""
    out = []
    for i, (uid, batch) in enumerate([("s1", "A"), ("s1", "B"), ("s2", "C"), ("s1", "A")]):
        ad = synthetic.cell_adata(batch, n=300, seed=i)
        ad.uns["sample"] = uid
        ad.obsm["unmx"] = pd.DataFrame(ad.layers["calibrated"], index=ad.obs.index, columns=["CD1", "CD2", "CD3"])
        out.append(ad)
    return out


@pytest.fixture
def adatas():
    return _adatas()


def _hist_colors(ax):
    return [p.get_edgecolor() for p in ax.patches]


def test_histogram_selects_sample_and_layout(adatas):
    plot_eval_histogram(adatas, "unmx", "s1", "sample")
    fig = plt.gcf()
    axes = [a for a in fig.axes if a.get_visible()]
    assert fig._suptitle.get_text() == "s1"
    assert len(fig.axes) >= 4
    panels = fig.axes[1:4]
    assert [a.get_xlabel() for a in panels] == ["CD1", "CD2", "CD3"]
    assert all(len(a.patches) == 3 for a in panels)  # three measurements of s1
    assert panels[0].get_yscale() == "log"
    assert len(axes) >= 1


def test_histogram_color_by_key_shares_colours(adatas):
    plot_eval_histogram(adatas, "unmx", "s1", "sample", key_color="batch", log=False)
    ax = plt.gcf().axes[1]
    cols = _hist_colors(ax)
    assert cols[0] == cols[2] != cols[1]  # both A aliquots share a colour
    assert ax.get_yscale() == "linear"
    labels = [t.get_text() for t in plt.gcf().axes[0].get_legend().get_texts()]
    assert labels == ["A", "B"]


def test_histogram_subset(adatas):
    plot_eval_histogram(adatas, "unmx", "s1", "sample", key_subset="batch", subset=["B"])
    assert all(len(a.patches) == 1 for a in plt.gcf().axes[1:4])


def test_histogram_no_match_raises(adatas):
    with pytest.raises(IndexError):
        plot_eval_histogram(adatas, "unmx", "nope", "sample")


def test_histogram_savepath_closes(adatas, tmp_path):
    p = tmp_path / "h.png"
    plot_eval_histogram(adatas, "unmx", "s1", "sample", savepath=str(p))
    assert p.exists() and p.stat().st_size > 0
    assert plt.get_fignums() == []


def test_histogram_bins_follow_xlim_and_cofactor(adatas, monkeypatch):
    seen = {}
    orig = plt.Axes.hist

    def spy(self, x, bins=None, **kw):
        seen.setdefault("bins", bins)
        return orig(self, x, bins=bins, **kw)

    monkeypatch.setattr(plt.Axes, "hist", spy)
    plot_eval_histogram(adatas, "unmx", "s1", "sample", xlim=[-1e3, 1e5], Nbin=11, cofactor=500)
    b = seen["bins"]
    assert len(b) == 11 and np.isclose(b[0], np.arcsinh(-2)) and np.isclose(b[-1], np.arcsinh(200))


def test_nxn_colours_per_dataset(adatas, monkeypatch):
    """R2-D6: one colour per dataset, datasets with the same key_color label share it (was one per unique label)."""
    got = {}
    monkeypatch.setattr(rplot, "base_plot_NxN_ds", lambda xs, colors, axlim, savepath: got.update(xs=xs, colors=colors, axlim=axlim))
    plot_eval_NxN(adatas, "unmx", "s1", "sample", key_color="batch", axlim=[-1e3, 1e5], cofactor=500)
    cmap = get_cmap(np.array(["A", "B", "A"]))
    assert got["colors"] == [cmap["A"], cmap["B"], cmap["A"]]
    assert len(got["xs"]) == 3
    assert np.allclose(got["axlim"], (np.arcsinh(-2), np.arcsinh(200)))
    assert np.allclose(got["xs"][1].to_numpy(), np.arcsinh(adatas[1].obsm["unmx"].to_numpy() / 500))


def test_nxn_matplotlib_version(adatas, monkeypatch):
    got = {}
    monkeypatch.setattr(rplot, "base_plot_NxN", lambda xs, colors, labels, savepath, suptitle, axlim:
                        got.update(colors=colors, labels=labels, suptitle=suptitle))
    out = plot_eval_NxN(adatas, "unmx", "s1", "sample", datashader=False, key_subset="batch", subset=["A"])
    assert out is None and got["suptitle"] == "s1"
    assert len(got["colors"]) == 2 and list(got["labels"]) == [0, 1]


def test_nxn_matplotlib_runs_and_saves(adatas, tmp_path):
    p = tmp_path / "n.png"
    plot_eval_NxN(adatas, "unmx", "s1", "sample", datashader=False, savepath=str(p))
    assert p.exists() and p.stat().st_size > 0


def test_nxn_datashader_passes_image_through(adatas, monkeypatch):
    """R3b-B4: plot_eval_NxN returns what base_plot_NxN_ds returns (the image; it returned None before)."""
    sentinel = object()
    monkeypatch.setattr(rplot, "base_plot_NxN_ds", lambda *a, **k: sentinel)
    assert plot_eval_NxN(adatas, "unmx", "s1", "sample") is sentinel


@pytest.mark.slow  # datashader / numba compilation, ~8 s
def test_nxn_datashader_returns_image_without_savepath(adatas, tmp_path):
    """R3b-B4: the datashader version returns the image (showed nothing before); None with a savepath."""
    pytest.importorskip("datashader")
    img = plot_eval_NxN(adatas, "unmx", "s1", "sample", key_color="batch")
    assert img is not None and hasattr(img, "size") and img.size[0] > 0
    p = tmp_path / "n.png"
    assert plot_eval_NxN(adatas, "unmx", "s1", "sample", savepath=str(p)) is None
    assert p.exists()
