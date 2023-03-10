from argparse import ArgumentParser, Namespace

import numpy as np
from skimage import io, measure
from skimage.morphology import remove_small_objects

from src.utils import DATA_PATH
from src.utils import plot_heatmap, plot_mask, plot_masks_comparison, print_stats


def small_objects(heatmap: np.array, th: float, min_size: int, show: bool = True):
    _ = plot_heatmap(heatmap, "heatmap")
    print_stats(heatmap)

    # thresholding
    t_mask: np.array = heatmap > th
    print_stats(t_mask)

    # get objects, this is the first step
    labels_pred, nlabels_pred = measure.label(t_mask, return_num=True, connectivity=1)
    print_stats(labels_pred)
    print(f"{nlabels_pred=}")

    # remove small objects, under `min_size` area
    cleaned_mask: np.array = remove_small_objects(labels_pred, min_size=min_size, connectivity=1)

    if show:
        plot_heatmap(t_mask, "thresholded mask")
    fig_raw, ax_raw, cmap_raw = plot_mask(labels_pred, "raw mask")
    fig_clean, ax_clean, _ = plot_mask(cleaned_mask, "cleaned mask")
    plot_masks_comparison(ax_raw, ax_clean, cmap_raw, title="Remove small objects")

    return t_mask, cleaned_mask, labels_pred


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.usage = "Read heatmap and remove small objects that pass the thresholding."

    parser.add_argument('fn',
                        help="Filename of the heatmap to post-process. This assumes files are in `DATA_PATH` folder.",
                        type=str)
    parser.add_argument('threshold', help="Binarization threshold.", type=float, default=0.5, nargs='?')
    parser.add_argument('min_size', help="Minimum allowed object size.", type=int, default=100, nargs='?')
    args: Namespace = parser.parse_args()

    # get sample mask
    heatmap: np.array = io.imread(DATA_PATH / args.fn, as_gray=True) / 255
    t_mask, cleaned_mask, labels_pred = small_objects(heatmap, args.threshold, args.min_size)

    # save masks without small objects
    outpath = DATA_PATH / 'post-processed'
    outpath.mkdir(parents=True, exist_ok=True)
    io.imsave(outpath / f"{args.fn.split('.')[0]}-cleaned.png", cleaned_mask.astype('uint8') * 255,
              check_contrast=False)
