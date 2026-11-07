"""Tests of cytovanni.utils.nxn: NxN scatter grids with matplotlib and datashader, and the image helpers."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cytovanni.utils import base_plot_NxN, base_plot_NxN_ds
from cytovanni.utils.nxn import (combine_images, draw_lines_origin, label_img_array, pad_border, rasterize_data,
                                 render_text_image, single_datashader_plot)

pytest.importorskip("datashader")


@pytest.fixture
def datas(rng):
    """Two datasets on an arcsinh-like scale; the second lacks marker 'c'."""
    a = pd.DataFrame(rng.normal(0, 2, (300, 3)), columns=["b", "a", "c"])
    b = pd.DataFrame(rng.normal(1, 1, (200, 2)), columns=["a", "b"])
    return [a, b]


def _panel(fig, x, y):
    return [ax for ax in fig.axes if ax.get_xlabel() == x and ax.get_ylabel() == y]


# ------------------------------------------------------------------ base_plot_NxN (matplotlib)
def test_base_plot_NxN_upper_triangle(datas):
    base_plot_NxN(datas, ["red", "blue"], labels=["one", "two"], suptitle="T", axlim=(-5, 5))
    fig = plt.gcf()
    assert len(fig.axes) == 6  # 3 markers: diagonal + upper triangle
    assert np.allclose(fig.get_size_inches(), [18, 18])
    (ab,) = _panel(fig, "a", "b")  # marker j (y) against marker i (x) for i <= j (sorted markers)
    assert len(ab.collections) == 2 and ab.get_xlim() == (-5, 5)
    (ac,) = _panel(fig, "a", "c")
    assert len(ac.collections) == 1  # the second dataset lacks 'c'
    assert not _panel(fig, "b", "a")
    (aa,) = _panel(fig, "a", "a")
    assert aa.get_legend() is not None and ab.get_legend() is None
    assert fig._suptitle.get_text() == "T"


def test_base_plot_NxN_arglist_and_no_zero_lines(datas):
    base_plot_NxN(datas, ["red", "blue"], arglist=[{"s": 3.}, {"color": "green", "marker": "x"}], mark_zero=False)
    fig = plt.gcf()
    (ab,) = _panel(fig, "a", "b")
    assert not ab.get_lines() and ab.get_legend() is None
    assert ab.collections[0].get_sizes()[0] == 3.
    assert np.allclose(ab.collections[1].get_facecolor()[0][:3], matplotlib.colors.to_rgb("green"))


def test_base_plot_NxN_colors_df_orientation(datas):
    """R2-E15: colors_df.loc[xkey, ykey] (was [ykey, xkey]), as in the _ds version and the docstring."""
    markers = ["a", "b", "c"]
    special = {("a", "b"): ["red", "red"], ("b", "a"): ["blue", "blue"]}
    colors_df = pd.DataFrame({c: {r: special.get((r, c), ["k", "k"]) for r in markers} for c in markers})
    base_plot_NxN(datas, ["k", "k"], colors_df=colors_df)
    (ab,) = _panel(plt.gcf(), "a", "b")  # x = a, y = b
    assert np.allclose(ab.collections[0].get_facecolor()[0][:3], matplotlib.colors.to_rgb("red"))


def test_base_plot_NxN_removed_arguments(datas):
    """R2-E15: `dask` and `label_arcsinh` were removed."""
    with pytest.raises(TypeError):
        base_plot_NxN(datas, ["r", "b"], label_arcsinh=True)
    with pytest.raises(TypeError):
        base_plot_NxN_ds(datas, dask=True)


def test_base_plot_NxN_savepath_closes(datas, tmp_path):
    n = len(plt.get_fignums())
    base_plot_NxN(datas, ["red", "blue"], savepath=str(tmp_path / "nxn.png"), dpi=20)
    assert (tmp_path / "nxn.png").stat().st_size > 0 and len(plt.get_fignums()) == n


# ------------------------------------------------------------------ image helpers
def test_rasterize_data_single_group_density():
    df = pd.DataFrame({"x": [0., 1., 1.], "y": [0., 1., 1.], "g": [0, 0, 0]})
    agg, img = rasterize_data(df, "x", "y", "g", drange=(0, 1), plot_pxl=50)
    assert img.shape == (50, 50, 4) and img.dtype == np.uint8
    assert agg.values.sum() == 3 and agg.ndim == 2


def test_rasterize_data_row_zero_is_top():
    df = pd.DataFrame({"x": [.5], "y": [.95], "g": [0]})
    _, img = rasterize_data(df, "x", "y", "g", drange=(0, 1), shift_out=0., plot_pxl=20)
    rows = np.nonzero(img[..., 3])[0]
    assert len(rows) and rows.max() < 3  # high y -> top rows


def test_rasterize_data_groups_and_norm():
    df = pd.DataFrame({"x": [.1, .1, .9] + [.5] * 100, "y": [.1, .1, .9] + [.5] * 100,
                       "g": pd.Categorical([0, 0, 0] + [1] * 100)})
    agg, img = rasterize_data(df, "x", "y", "g", drange=(0, 1), plot_pxl=30, palette=["#ff0000", "#0000ff"])
    assert agg.ndim == 3 and agg.shape[-1] == 2 and img.shape == (30, 30, 4)
    agg_n, _ = rasterize_data(df, "x", "y", "g", drange=(0, 1), plot_pxl=30, common_norm=False)
    m = agg_n.max(dim=("x", "y")).values
    assert np.allclose(m, 1.)  # every group normalised to its own maximum


def test_rasterize_data_widens_drange():
    df = pd.DataFrame({"x": [0., 1.], "y": [0., 1.], "g": [0, 0]})
    agg, _ = rasterize_data(df, "x", "y", "g", drange=(0, 1), shift_out=.1, plot_pxl=10)
    assert np.allclose(agg.attrs["x_range"], (-.1, 1.1)) and np.allclose(agg.attrs["y_range"], (-.1, 1.1))


def test_draw_lines_origin():
    df = pd.DataFrame({"x": [-1., 1.], "y": [-1., 3.], "g": [0, 0]})
    agg, img = rasterize_data(df, "x", "y", "g", drange=None, plot_pxl=40)
    img = np.zeros_like(img)
    out = draw_lines_origin(agg, img)
    assert out is img
    x0 = int((0 - agg.x_range[0]) / (agg.x_range[1] - agg.x_range[0]) * 40)
    y0 = int((agg.y_range[1] - 0) / (agg.y_range[1] - agg.y_range[0]) * 40)
    assert (img[:, x0, 3] == 255).all() and (img[y0, :, 3] == 255).all() and (img[..., :3] == 0).all()
    agg2, img2 = rasterize_data(pd.DataFrame({"x": [1., 2.], "y": [1., 2.], "g": [0, 0]}), "x", "y", "g", plot_pxl=10)
    before = img2.copy()
    assert np.array_equal(draw_lines_origin(agg2, img2), before)  # origin outside: unchanged


def test_pad_border():
    img = np.zeros((400, 200, 4), np.uint8)
    out = pad_border(img, share=200)
    assert out.shape == (404, 202, 4)
    assert (out[0, :, 3] == 255).all() and (out[:, 0, 3] == 255).all() and (out[0, :, :3] == 0).all()
    assert (out[2:-2, 1:-1, 3] == 0).all() and (img[..., 3] == 0).all()  # interior and input untouched


def test_render_text_image():
    im = render_text_image("CD3", size=(120, 30))
    assert isinstance(im, Image.Image) and im.mode == "RGBA" and im.size == (120, 30)
    a = np.array(im)
    assert a[..., 3].max() == 255 and a[0, 0, 3] == 0 and a[..., :3][a[..., 3] == 255].max() == 0


def test_label_img_array_shape():
    img = np.full((100, 100, 4), 255, np.uint8)
    out = label_img_array(img, "x", "y")
    assert out.shape == (110, 110, 4) and out.dtype == np.uint8
    assert (out[:100, 10:] == 255).all()  # image top right, labels below / left


def test_combine_images():
    tiles = [[np.full((2, 3, 4), 10 * (2 * i + j), np.uint8) for j in range(2)] for i in range(3)]
    out = combine_images(tiles)
    assert out.shape == (6, 6, 4) and out[0, 0, 0] == 0 and out[0, 3, 0] == 10 and out[5, 5, 0] == 50


# ------------------------------------------------------------------ base_plot_NxN_ds
@pytest.mark.parametrize("ndata", [1, 2])
def test_base_plot_NxN_ds_image(datas, ndata):
    img = base_plot_NxN_ds(datas[:ndata], pixel_per_channel=60, axlim=(-4, 4))
    side = 60 + 2 * 1  # frame of 60 // 200 -> 1 pixel, then label strips of side // 10
    side += side // 10
    n = 3  # markers a, b, c
    assert isinstance(img, Image.Image) and img.mode == "RGBA" and img.size == (n * side, n * side)
    a = np.array(img)
    assert (a[side:2 * side, :side, 3] == 0).all()  # lower triangle panel (1, 0) is empty / transparent
    assert a[:side, :side, 3].max() == 255


def test_base_plot_NxN_ds_savepath(datas, tmp_path):
    for name in ("nxn.png", "nxn.jpg"):
        assert base_plot_NxN_ds(datas, pixel_per_channel=40, savepath=str(tmp_path / name)) is None
        im = Image.open(tmp_path / name)
        assert im.mode == "RGB" and im.size[0] == im.size[1]
        assert np.array(im)[-1, 0].min() > 200  # lower triangle white on the saved image


def test_base_plot_NxN_ds_colors_df(datas):
    markers = ["a", "b", "c"]
    special = {("a", "b"): ["#ff0000", "#ff0000"]}
    colors_df = pd.DataFrame({c: {r: special.get((r, c), ["#000000"] * 2) for r in markers} for c in markers})
    a = np.array(base_plot_NxN_ds(datas, colors_df=colors_df, pixel_per_channel=60, mark_zero=False))
    side = (62 + 6)
    panel_ab = a[:62, side + 6:2 * side]  # row 0 (x = a), column 1 (y = b), without labels
    painted = panel_ab[panel_ab[..., 3] > 0][..., :3]
    assert len(painted) and (painted[:, 0] >= painted[:, 2]).all() and painted[:, 0].max() > 100


@pytest.mark.slow
def test_base_plot_NxN_ds_parallel_equals_serial(datas):
    serial = np.array(base_plot_NxN_ds(datas, pixel_per_channel=40))
    parallel = np.array(base_plot_NxN_ds(datas, pixel_per_channel=40, parallel=True))
    assert np.array_equal(serial, parallel)


def test_single_datashader_plot(rng):
    pytest.importorskip("colorcet")
    fig, ax = plt.subplots()
    single_datashader_plot(ax, rng.normal(size=500), rng.normal(size=500), Nh=30, Nw=40, range_x=(-3, 3), range_y=(-2, 2))
    (im,) = ax.images
    assert im.get_array().shape[:2] == (30, 40) and np.allclose(im.get_extent(), [-3, 3, -2, 2])
