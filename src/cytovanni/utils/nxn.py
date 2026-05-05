import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from PIL import Image, ImageDraw, ImageFont

from .misc import labelscolors_to_legendhandles, palette_20

def _base_plot_NxN_single(datas, colors, labels, xkey, ykey, ax, axlim=None, s=.01, arglist=None, mark_zero=True, legend=False, fct_label_axis=None):
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

def base_plot_NxN(datas, colors, labels=None, colors_df=None, savepath="", suptitle="", axlim=None, s=.01, dpi=150, arglist=None, mark_zero=True, label_arcsinh=False):
    """ Base for NxN plots of all stains.
        
        :param datas: list. List of dataframes with the data to be plotted, index event, column stain.
        
        :param colors: list. Colors for the different datas.

        :param colors_df: pd.Dataframe of lists. If given, gets color palette separately as colors_df.loc[xkey, ykey] for every individual plot.
        
        :param labels: list. Labels for the different datas.
        
        :param axlim: None, tuple. If not None, set both x and y axis limits to this.
        
        :param savepath: str. If given, saves the plot to file, closes it, and tries to clear as much of the used memory as possible. See https://stackoverflow.com/questions/28757348/how-to-clear-memory-completely-of-all-matplotlib-plots and https://github.com/matplotlib/matplotlib/issues/20300
        
        :param arglist: None, iterable. Either None, or list of kwarg dicts for ax.scatter.

        :param mark_zero: bool. If True, highlight zero along each axis.
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
                _base_plot_NxN_single(datas, colors if colors_df is None else colors_df.loc[channels[j], channels[i]],
                                      labels, channels[i], channels[j], ax, axlim=axlim, s=s, arglist=arglist, mark_zero=mark_zero, legend=((i==j) and (labels is not None)))
    
    if suptitle: fig.suptitle(suptitle, size=45)
    fig.set_layout_engine("constrained")
    
    if savepath:
        plt.savefig(savepath, dpi=dpi)
        fig.clear()
        plt.close()


def rasterize_data(df, xkey, ykey, gkey, drange=None, shift_out=.02, plot_pxl=300, palette=palette_20, common_norm=True):
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
    pad_x, pad_y = max(img_array.shape[0]//share, 1), max(img_array.shape[1]//share, 1)
    img_array_pad = img_array.copy()
    img_array_pad[...,-1] = 255-img_array_pad[...,-1]
    img_array_pad = np.pad(img_array_pad, ((pad_x, pad_x),(pad_y, pad_y), (0,0)), constant_values=0)
    img_array_pad[...,-1] = 255-img_array_pad[...,-1]
    return img_array_pad

def render_text_image(text, size=(300, 40), font_size=None, font_path=None):
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
    ximg_array = np.array(render_text_image(xlabel, size=(img_array.shape[0], img_array.shape[0]//share)))
    yimg_array = np.array(render_text_image(ylabel, size=(img_array.shape[1], img_array.shape[1]//share))).swapaxes(0,1)[::-1]
    
    cimg_array = np.zeros((img_array.shape[0] + ximg_array.shape[0], img_array.shape[1] + yimg_array.shape[1], 4), dtype=np.uint8)
    cimg_array[-ximg_array.shape[0]:, -ximg_array.shape[1]:, :] = ximg_array
    cimg_array[:yimg_array.shape[0], :yimg_array.shape[1], :] = yimg_array
    cimg_array[:img_array.shape[0], -img_array.shape[1]:, :] = img_array

    return cimg_array

def combine_images(image_grid):
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

def base_plot_NxN_ds(datas, colors=palette_20, colors_df=None, axlim=None, pixel_per_channel=300, mark_zero=True, savepath="", common_norm=False, parallel=False, dask=False):
    """ Base for NxN plots of all stains.

        - Maybe implement grid of fixed step to allow better automatic axlim?
        - Scaling of groups for equal visual impact!
        
        :param datas: list. List of dataframes with the data to be plotted, index event, column stain.
        
        :param colors: list. Colors for the different datas, can just be a longer palette.

        :param colors_df: pd.Dataframe of lists. If given, gets color palette separately as colors_df.loc[xkey, ykey] for every individual plot.
        
        :param axlim: None, tuple. Limit for both x and y axis, can be inferred automatically but not that this can be confusing as there are no axis ticks.
        
        :param savepath: str. If given, doesn't return the image but instead saves it to file.

        :param common_norm: bool. If True, simply plots the data, if False normalizes each group to (roughly) equal visual weight.

        :param mark_zero: bool. If True, highlight zero along each axis.

        :param pixel_per_channel: int. Number of pixels per axis for each individual scatter plot.

        :param parallel: bool. Can run in parallel, usually not actually worth the overhead.

        :param dask: bool. Whether to use dask for parallelization.
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
        dask = False
        if dask:
            from dask.distributed import Client
            client = Client()
            #tasks = [client.submit(generate_part, i, j) for i in range(N) for j in range(N)]
            tasks = [client.submit(generate_row, i) for i in range(N)]
            imgs = client.gather(tasks)
            #imgs = [imgs[i * N:(i + 1) * N] for i in range(N)]
            client.close()
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
    """ 
    """
    import datashader as ds, colorcet as cc
    cvs = ds.Canvas(plot_width=Nw, plot_height=Nh, x_range=range_x, y_range=range_y)
    df = pd.DataFrame(np.vstack([vx, vy]).T, columns=["x","y"])
    agg = cvs.points(df, "x", "y")
    img = ds.tf.shade(agg, cmap=cc.fire if cmap is None else cmap)
    ax.imshow(np.asarray(img.to_pil()), extent=agg.x_range + agg.y_range, aspect="auto")


