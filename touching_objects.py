import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from small_objects import small_objects, THRESHOLD, MIN_SIZE
from src.utils import plot_heatmap, plot_mask, print_stats, plot_masks_comparison, DATA_PATH

FS_SIZE = (2, 2)

MAX_FILTER_SIZE = 20


def enhance_objects_basin(mask: np.array, max_filt_size: int, show: bool = True):
    # get boolean mask for successive steps
    bool_mask = mask.astype(bool)
    print('boolean mask')
    print_stats(bool_mask)

    # compute distance of each pixel from the closest background pixel
    distance = ndimage.distance_transform_edt(bool_mask)
    print('distance')
    print_stats(distance)

    # now apply maximum filtering to enhance the core part of objects; NOTE: play with `size` parameter
    maxi = ndimage.maximum_filter(distance, size=max_filt_size, mode='constant')
    print('maxi')
    print_stats(maxi)

    if show:
        plot_heatmap(distance, 'distance from borders')
        fig, ax, cax = plot_heatmap(maxi, f'max filter; size={max_filt_size}')

    return bool_mask, distance, maxi


def get_basin_markers(maxi: np.array, fs_size: (int, int), bool_mask: np.array, show: bool = True):
    # get indexes of local maxima
    local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones(fs_size), exclude_border=False,
                                labels=np.squeeze(bool_mask))

    print('local maximum')
    print_stats(local_maxi)

    # once we have the enhanced mask, we get markers for the bulk of the objects
    markers, nmarkers = ndimage.label(local_maxi)
    print('markers')
    print_stats(markers)

    if show:
        plot_mask(local_maxi, f'local maximum; footprint={fs_size}')

    return local_maxi, markers, nmarkers


def touching_objects(th: float, min_size: int, max_filt_size: int, fs_size: (int, int), show: bool = True):
    # first get cleaned mask
    fn, heatmap, t_mask, cleaned_mask, thresholded_objects = small_objects(th, min_size, show=False)

    # find local maxima; NOTE: play with `max_filter_size` and `footprint` parameters
    bool_mask, distance, maxi = enhance_objects_basin(cleaned_mask, max_filt_size, show=show)
    local_maxi, markers, nmarkers = get_basin_markers(maxi, fs_size, bool_mask, show=show)

    # finally apply watershed to separate close-by objects; NOTE: play with `compactness` and `watershed_line` parameters
    watershed_mask = watershed(-distance, markers, mask=np.squeeze(bool_mask), compactness=1, watershed_line=False)
    if show:
        plot_heatmap(watershed_mask, 'watershed')
        print_stats(watershed_mask)

    fig_clean, ax_clean, _ = plot_mask(cleaned_mask, "cleaned mask")
    fig_watershed, ax_watershed, cmap_watershed = plot_mask(watershed_mask, "watershed")
    plot_masks_comparison(ax_clean, ax_watershed, cmap_watershed, title="Separe close/overlapping objects")

    return labels_cleaned, nlabels_cleaned


if __name__ == '__main__':
    touching_objects(THRESHOLD, MIN_SIZE, MAX_FILTER_SIZE, FS_SIZE)
