from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes._subplots import Axes
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure

DATA_PATH = Path().cwd() / 'data'


def plot_heatmap(heatmap: np.array, title: str):
    """Plot a heatmap with colorbar and title"""
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    im = axis.pcolormesh(np.flipud(heatmap), cmap='jet', )
    axis.set_title(title)
    axis.set_aspect('equal')

    # colorbar
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    return fig, axis, cax


def plot_mask(mask: np.array, title: str):
    """Plot mask with a different color per object and title."""
    labels, nobjs = measure.label(mask, return_num=True, connectivity=1)
    objs = measure.regionprops(labels)

    # create colormap: one color per object
    cmap = cm.get_cmap('tab20b', nobjs)

    # random shuffle to avoid similar colors to close-by objects
    np.random.shuffle(cmap.colors)

    # add black for background
    cmap = ListedColormap(np.insert(cmap.colors, 0, [0, 0, 0, 1], axis=0))

    # plot mask
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    axis.imshow(labels, cmap=cmap, vmin=0, vmax=nobjs)
    for obj in objs:
        plt.text(obj.bbox[1], obj.bbox[0], obj.label,
                 fontdict=dict(color='white', size=7),
                 bbox=dict(fill=False, linewidth=0)
                 )

    axis.set_title(f"{title} - N. objects: {nobjs}")
    plt.show()
    return fig, axis, cmap


def plot_masks_comparison(ax_raw: Axes, ax_processed: Axes, cmap: ListedColormap, title: str):
    """Plot masks before and after processing side by side for comparison."""
    raw_mask = ax_raw.get_images()[0].get_array()
    processed_mask = ax_processed.get_images()[0].get_array()

    raw_labels, raw_nobjs = measure.label(raw_mask, return_num=True, connectivity=1)
    raw_objs = measure.regionprops(raw_labels)
    processed_labels, processed_nobjs = measure.label(processed_mask, return_num=True, connectivity=1)
    processed_objs = measure.regionprops(processed_labels)

    fig_cmp, ax_cmp = plt.subplots(1, 2, figsize=(14, 6))
    ax_cmp[0].imshow(raw_mask, cmap=cmap)
    ax_cmp[0].set_title(f'before - N. objects: {raw_nobjs}')
    for obj in raw_objs:
        ax_cmp[0].text(obj.bbox[1], obj.bbox[0], obj.label,
                       fontdict=dict(color='white', size=7),
                       bbox=dict(fill=False, linewidth=0)
                       )
    ax_cmp[1].imshow(processed_mask, cmap=cmap)
    ax_cmp[1].set_title(f'after - N. objects: {processed_nobjs}')
    for obj in processed_objs:
        ax_cmp[1].text(obj.bbox[1], obj.bbox[0], obj.label,
                       fontdict=dict(color='white', size=7),
                       bbox=dict(fill=False, linewidth=0)
                       )

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig_cmp, ax_cmp


def print_stats(image: np.array):
    """Print useful image statistics."""
    shape, dtype, min_val, max_val, n_uniques = image.shape, image.dtype, image.min(), image.max(), len(
        np.unique(image))
    print(f"{shape=};\t{dtype=};\t{min_val=};\t{max_val=};\t{n_uniques=}")
