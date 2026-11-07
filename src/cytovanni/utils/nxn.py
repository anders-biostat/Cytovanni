"""NxN grids of pairwise scatter plots of all markers, with matplotlib or rasterised with datashader."""
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from PIL import Image, ImageDraw, ImageFont

from .misc import labelscolors_to_legendhandles, palette_20

def _base_plot_NxN_single(datas, colors, labels, xkey, ykey, ax, axlim=None, s=.01, arglist=None, mark_zero=True, legend=False, fct_label_axis=None):
    """Scatter `ykey` against `xkey` for every DataFrame in `datas` that has both columns, into `ax`."""
    for di, data in enumerate(datas):
        if xkey in data and ykey in data:
            x, y = data[xkey], data[ykey]
            if arglist is None:
                ax.scatter(x, y, s=s, color=colors[di], zorder=1)
            else:
                kwargs = {"s":s, "color":colors[di]}
                kwargs.update(arglist[di])
                ax.scatter(x, y, zorder=1, **kwargs)
    ax.set_xlabel(f"{xkey}", size=25)
    ax.set_ylabel(f"{ykey}", size=25)
    if axlim is not None:
        ax.set_xlim(axlim)
        ax.set_ylim(axlim)
    if mark_zero:
        ax.axvline(0, color="black", linewidth=.5, zorder=3)
        ax.axhline(0, color="black", linewidth=.5, zorder=3)
    ax.grid()

    if legend:
        ax.legend(loc="upper left", handles=labelscolors_to_legendhandles(labels, colors))

    if fct_label_axis is not None:
        fct_label_axis(ax)

def base_plot_NxN(datas, colors, labels=None, colors_df=None, savepath="", suptitle="", axlim=None, s=.01, dpi=150, arglist=None, mark_zero=True):
    """Draw an NxN grid of pairwise scatter plots of all markers with matplotlib, overlaying several datasets.

    The markers are the sorted union of the columns of all `datas`. Panel (i, j) for i <= j (upper triangle and
    diagonal) shows marker j (y) against marker i (x); a dataset is skipped in panels whose markers it lacks. The
    diagonal panels carry the legend if `labels` is given. The figure is 6 x 6 inches per panel.

    Parameters
    ----------
    datas : list of pandas.DataFrame
        Datasets to overlay, each of shape (events, stain), usually on the arcsinh scale. Drawn in list order (later
        ones on top).
    colors : list
        Matplotlib colour of every dataset.
    labels : list of str, optional
        Legend label of every dataset. Default None: no legend.
    colors_df : pandas.DataFrame, optional
        Per-panel colours instead of `colors`: ``colors_df.loc[xkey, ykey]`` is the list of dataset colours for the
        panel of markers xkey (x) and ykey (y).
    savepath : str, default ""
        If given, save the figure there, then clear and close it to free memory. Otherwise the figure stays open
        (no ``plt.show()``).
    suptitle : str, default ""
        Figure title.
    axlim : tuple of float, optional
        Limits for both axes of every panel. Default None: matplotlib's automatic limits.
    s : float, default 0.01
        Marker size of the scatter points.
    dpi : int, default 150
        Resolution of the saved figure.
    arglist : list of dict, optional
        Extra keyword arguments for ``ax.scatter``, one dict per dataset (they override `s` and the colour).
    mark_zero : bool, default True
        Draw thin black lines at x = 0 and y = 0.

    See Also
    --------
    base_plot_NxN_ds : rasterised with datashader, much faster for many events.
    """
    channels = sorted(list(set().union(*[set(d.columns) for d in datas])))
    N = len(channels)

    fig = plt.figure(constrained_layout=True, figsize=(6*N, 6*N))
    axes = []
    spec = gridspec.GridSpec(ncols=N, nrows=N, figure=fig)
    for i in range(N):
        for j in range(N):
            if i<=j:
                ax = fig.add_subplot(spec[i, j])
                axes.append(ax)
                _base_plot_NxN_single(datas, colors if colors_df is None else colors_df.loc[channels[i], channels[j]],
                                      labels, channels[i], channels[j], ax, axlim=axlim, s=s, arglist=arglist, mark_zero=mark_zero, legend=((i==j) and (labels is not None)))
    
    if suptitle: fig.suptitle(suptitle, size=45)
    fig.set_layout_engine("constrained")
    
    if savepath:
        plt.savefig(savepath, dpi=dpi)
        fig.clear()
        plt.close()


def rasterize_data(df, xkey, ykey, gkey, drange=None, shift_out=.02, plot_pxl=300, palette=palette_20, common_norm=True):
    """Rasterise a scatter plot of `ykey` against `xkey` with datashader, coloured by the group column `gkey`.

    Needs datashader (optional dependency, extra ``plot``).

    Parameters
    ----------
    df : pandas.DataFrame
        Events with the columns `xkey`, `ykey` and `gkey` (categorical for several groups).
    xkey, ykey : str
        Columns for the x and y axis.
    gkey : str
        Group column. With more than one group, every group gets its colour from `palette` (``ds.count_cat``);
        with a single group, datashader's default colour map shows the event density and `palette` is not used.
    drange : tuple of float, optional
        Range (min, max) of both axes, widened by `shift_out` x its width on both sides. Default None: datashader
        takes the range from the data, separately for x and y.
    shift_out : float, default 0.02
        Relative margin added to `drange` on both sides.
    plot_pxl : int, default 300
        Width and height of the image in pixels.
    palette : list of str, default palette_20
        Colour of every group, in category order.
    common_norm : bool, default True
        If False, normalise the counts of every group to its own maximum, so that groups of different size get
        roughly equal visual weight.

    Returns
    -------
    agg : xarray.DataArray
        The datashader aggregate (counts per pixel, per group for several groups).
    img_array : numpy.ndarray
        RGBA image, shape (plot_pxl, plot_pxl, 4), uint8; row 0 is the top (highest y).
    """
    # load here as datashader throws some warnings if CUDA is not available
    import datashader as ds
    import datashader.transfer_functions as tf
    
    drange_plot = (drange + (drange[1]-drange[0]) * np.array([-1,1]) * shift_out) if drange is not None else None
    cvs = ds.Canvas(plot_width=plot_pxl, plot_height=plot_pxl, x_range=drange_plot, y_range=drange_plot)
    
    if len(df[gkey].unique())>1:
        agg = cvs.points(df, xkey, ykey, agg=ds.count_cat(gkey))
        if not common_norm:
            agg.data = (agg / agg.max(dim=(xkey, ykey))).data # norm
            agg.data = agg.where(agg!=0., np.nan).data # mask zero, I think tf.shade expects count input and is weird with floats
        img = tf.shade(agg, color_key=palette)
    else:
        agg = cvs.points(df, xkey, ykey, agg=ds.count())
        img = tf.shade(agg)
    img_array = np.array(img.to_pil())

    return agg, img_array

def draw_lines_origin(agg, img_array):
    """Draw black, opaque lines at x = 0 and y = 0 into the image of `rasterize_data`, if they lie inside it.

    Parameters
    ----------
    agg : xarray.DataArray
        Aggregate from `rasterize_data` (its ``x_range`` and ``y_range`` locate the origin).
    img_array : numpy.ndarray
        RGBA image of shape (height, width, 4); modified in place.

    Returns
    -------
    numpy.ndarray
        `img_array`.
    """
    x_range = agg.x_range
    y_range = agg.y_range
    height, width, _ = img_array.shape

    x0_pix = int((0 - x_range[0]) / (x_range[1] - x_range[0]) * width)
    y0_pix = int((y_range[1] - 0) / (y_range[1] - y_range[0]) * height)

    if 0 <= x0_pix < width:
        img_array[:, x0_pix, :3], img_array[:, x0_pix, 3] = 0, 255
    if 0 <= y0_pix < height:
        img_array[y0_pix, :, :3], img_array[y0_pix, :, 3] = 0, 255

    return img_array

def pad_border(img_array, share=200):
    """Add an opaque black frame around an RGBA image.

    Parameters
    ----------
    img_array : numpy.ndarray
        RGBA image of shape (height, width, 4).
    share : int, default 200
        The frame is ``size // share`` pixels thick (at least 1) along each axis.

    Returns
    -------
    numpy.ndarray
        Padded copy.
    """
    pad_x, pad_y = max(img_array.shape[0]//share, 1), max(img_array.shape[1]//share, 1)
    img_array_pad = img_array.copy()
    img_array_pad[...,-1] = 255-img_array_pad[...,-1]
    img_array_pad = np.pad(img_array_pad, ((pad_x, pad_x),(pad_y, pad_y), (0,0)), constant_values=0)
    img_array_pad[...,-1] = 255-img_array_pad[...,-1]
    return img_array_pad

def render_text_image(text, size=(300, 40), font_size=None, font_path=None):
    """Render `text` in black, centred, on a transparent image.

    Parameters
    ----------
    text : str
        Text to render.
    size : tuple of int, default (300, 40)
        Image size (width, height) in pixels.
    font_size : float, optional
        Font size in pixels. Default: height / 1.3 (floored).
    font_path : str, optional
        Path of a TrueType font. Default: matplotlib's DejaVu Sans.

    Returns
    -------
    PIL.Image.Image
        RGBA image.
    """
    img = Image.new('RGBA', size, (255, 255, 255, 0))  # Transparent background
    draw = ImageDraw.Draw(img)

    if font_path is None:
        font_path = mpl.font_manager.findfont("DejaVu Sans")
    if font_size is None:
        font_size = size[1]//1.3
    font = ImageFont.truetype(font_path, font_size)

    x, y = size[0] // 2, size[1] // 2
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255), anchor='mm')

    return img

def label_img_array(img_array, xlabel, ylabel, share=10):
    """Add an x label below and a rotated y label left of a square RGBA image.

    Parameters
    ----------
    img_array : numpy.ndarray
        RGBA image of shape (height, width, 4), uint8; the labels are sized for a square image.
    xlabel, ylabel : str
        Axis labels.
    share : int, default 10
        The label strips are ``size // share`` pixels high.

    Returns
    -------
    numpy.ndarray
        Labelled RGBA image, larger by the label strips (transparent background).
    """
    ximg_array = np.array(render_text_image(xlabel, size=(img_array.shape[0], img_array.shape[0]//share)))
    yimg_array = np.array(render_text_image(ylabel, size=(img_array.shape[1], img_array.shape[1]//share))).swapaxes(0,1)[::-1]
    
    cimg_array = np.zeros((img_array.shape[0] + ximg_array.shape[0], img_array.shape[1] + yimg_array.shape[1], 4), dtype=np.uint8)
    cimg_array[-ximg_array.shape[0]:, -ximg_array.shape[1]:, :] = ximg_array
    cimg_array[:yimg_array.shape[0], :yimg_array.shape[1], :] = yimg_array
    cimg_array[:img_array.shape[0], -img_array.shape[1]:, :] = img_array

    return cimg_array

def combine_images(image_grid):
    """Tile a grid of equally sized images into one image.

    Parameters
    ----------
    image_grid : list of list of numpy.ndarray
        Rows of images, all with the shape of ``image_grid[0][0]``, (height, width, channels).

    Returns
    -------
    numpy.ndarray
        Combined image, uint8, shape (rows * height, columns * width, channels).
    """
    img_height, img_width, channels = image_grid[0][0].shape
    total_height = img_height * len(image_grid)
    total_width = img_width * len(image_grid[0])
    
    combined_image = np.zeros((total_height, total_width, channels), dtype=np.uint8)
    for i, row in enumerate(image_grid):
        for j, img in enumerate(row):
            x_offset = j * img_width
            y_offset = i * img_height
            combined_image[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img
    
    return combined_image

def base_plot_NxN_ds(datas, colors=palette_20, colors_df=None, axlim=None, pixel_per_channel=300, mark_zero=True, savepath="", common_norm=False, parallel=False):
    """Draw an NxN grid of pairwise scatter plots of all markers as one image, rasterised with datashader.

    Like `base_plot_NxN`, but much faster for many events: every panel is a datashader image with a black frame and
    the two marker names as labels (no axis ticks). The markers are the sorted union of the columns of all `datas`;
    panel (i, j) for i <= j shows marker j (y) against marker i (x), the lower triangle is left empty (transparent,
    white in the saved file). Needs datashader (optional dependency, extra ``plot``).

    Parameters
    ----------
    datas : list of pandas.DataFrame
        Datasets to overlay, each of shape (events, stain), usually on the arcsinh scale. With a single dataset,
        the panels show the event density with datashader's default colour map instead of `colors`.
    colors : list of str, default palette_20
        Colour of every dataset (a longer palette is fine).
    colors_df : pandas.DataFrame, optional
        Per-panel colours instead of `colors`: ``colors_df.loc[xkey, ykey]`` is the list of dataset colours for the
        panel of markers xkey (x) and ykey (y).
    axlim : tuple of float, optional
        Range (min, max) of both axes in every panel. Values outside are clipped to it, so they pile up at the edge
        of the panel. Default None: the range of every panel is taken from its data, which can be confusing since
        there are no axis ticks.
    pixel_per_channel : int, default 300
        Width and height of every panel in pixels (before frame and labels).
    mark_zero : bool, default True
        Draw black lines at x = 0 and y = 0.
    savepath : str, default ""
        If given, save the image there (on a white background; JPEG with quality 85 for .jpg/.jpeg) and return
        None.
    common_norm : bool, default False
        If False, normalise every dataset to its own maximum count per panel, so that datasets of different size get
        roughly equal visual weight. If True, plot the counts as they are.
    parallel : bool, default False
        Render the rows in parallel with joblib (10 jobs, one row per task); usually not worth the overhead.

    Returns
    -------
    PIL.Image.Image or None
        The RGBA image (transparent background) if no `savepath` is given, else None.
    """
    channels = sorted(list(set().union(*[set(d.columns) for d in datas])))
    N = len(channels)
    pdatas = [d.copy() for d in datas]
    for i, d in enumerate(pdatas):
        d["__group__"] = i
    pdata = pd.concat(pdatas)
    pdata["__group__"] = pdata["__group__"].astype("category")
    
    for c in channels:
        if axlim is not None:
            pdata[c] = pdata[c].clip(*axlim)
        pdata["_"+c] = pdata[c]
    
    def generate_part(i,j):
        if i<=j:
            xkey, ykey = channels[i], channels[j]
            agg, img_array = rasterize_data(pdata, xkey, ykey if (xkey!=ykey) else "_"+ykey, "__group__", drange=axlim, shift_out=.01,
                                            plot_pxl=pixel_per_channel, palette=colors if colors_df is None else colors_df.loc[xkey, ykey], common_norm=common_norm)
            if mark_zero:
                img_array = draw_lines_origin(agg, img_array)
            img_array = label_img_array(pad_border(img_array), xkey, ykey)
            return img_array
        else:
            return None

    def generate_row(i):
        return [generate_part(i,j) for j in range(N)]

    if not parallel:
        imgs = []
        for i in range(N):
            row = []
            for j in range(N):
                row.append(generate_part(i,j))
            imgs.append(row)
    else:
        from joblib import Parallel, delayed
        imgs = Parallel(n_jobs=10)(delayed(generate_row)(i) for i in range(N))
    
    imgs = [[(j if (j is not None) else np.zeros_like(imgs[0][0])) for j in i] for i in imgs]
    img = combine_images(imgs)
    image = Image.fromarray(img)

    if savepath:
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, (0, 0), image)
        if savepath.endswith(".jpg") or savepath.endswith(".jpeg"):
            new_image.convert('RGB').save(savepath, quality=85, subsampling=0) 
        else:
            new_image.convert('RGB').save(savepath) 
    else:
        return image


def single_datashader_plot(ax, vx, vy, Nh=800, Nw=800, range_x=None, range_y=None, cmap=None):
    """Draw a rasterised scatter plot (event density) of `vy` against `vx` into a matplotlib axis.

    Needs datashader and colorcet (optional dependencies, extra ``plot``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (``ax.imshow``, with the data ranges as extent).
    vx, vy : array-like of float
        x and y values of the events, same length.
    Nh, Nw : int, default 800
        Height and width of the raster in pixels.
    range_x, range_y : tuple of float, optional
        Data range (min, max) of the raster. Default None: the range of the data.
    cmap : list or matplotlib colormap, optional
        Colour map for the density. Default None: colorcet's ``fire``.
    """
    import datashader as ds, colorcet as cc
    cvs = ds.Canvas(plot_width=Nw, plot_height=Nh, x_range=range_x, y_range=range_y)
    df = pd.DataFrame(np.vstack([vx, vy]).T, columns=["x","y"])
    agg = cvs.points(df, "x", "y")
    img = ds.tf.shade(agg, cmap=cc.fire if cmap is None else cmap)
    ax.imshow(np.asarray(img.to_pil()), extent=agg.x_range + agg.y_range, aspect="auto")


