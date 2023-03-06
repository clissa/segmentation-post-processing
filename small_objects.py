import numpy as np
from skimage import io, measure
from skimage.morphology import remove_small_objects

from src.utils import DATA_PATH
from src.utils import plot_heatmap, plot_mask, plot_masks_comparison, print_stats

THRESHOLD: float = 0.45

MIN_SIZE: int = 100


def small_objects(th: float, min_size: int, show: bool=True):
    # get sample mask
    fn: str = '11.png'
    heatmap: np.array = io.imread(DATA_PATH / '11.png', as_gray=True) / 255

    _ = plot_heatmap(heatmap, "heatmap", show=True)
    print_stats(heatmap)

    # thresholding
    t_mask: np.array = heatmap > th
    plot_heatmap(t_mask, "thresholded mask", show=show)
    print_stats(t_mask)

    # get objects, this is the first step
    labels_pred, nlabels_pred = measure.label(t_mask, return_num=True, connectivity=1)
    print_stats(labels_pred)
    print(f"{nlabels_pred=}")

    # remove small objects, under `min_size` area
    cleaned_mask: np.array = remove_small_objects(labels_pred, min_size=min_size, connectivity=1)

    fig_raw, ax_raw, cmap_raw = plot_mask(labels_pred, "raw mask", show=False)
    fig_clean, ax_clean, _ = plot_mask(cleaned_mask, "cleaned mask", show=False)
    plot_masks_comparison(ax_raw, ax_clean, cmap_raw, show=show)

    # save masks without small objects
    io.imsave(DATA_PATH / f"{fn.split('.')[0]}-cleaned.png", cleaned_mask.astype('uint8') * 255, check_contrast=False)
    return fn, heatmap, t_mask, cleaned_mask, labels_pred

if __name__ == '__main__':
    small_objects(THRESHOLD, MIN_SIZE)
