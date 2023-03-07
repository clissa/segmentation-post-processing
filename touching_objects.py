import numpy as np
from scipy import ndimage
from skimage import io
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
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


def get_basin_markers(maxi: np.array, fs_size: (int, int), bool_mask: np.array, show: bool = True, min_size: int = 0):
    # get indexes of local maxima
    local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones(fs_size), exclude_border=False,
                                labels=np.squeeze(bool_mask))
    if min_size:
        local_maxi = remove_small_objects(local_maxi, min_size=min_size, connectivity=1)

    print('local maximum')
    print_stats(local_maxi)

    # once we have the enhanced mask, we get markers for the bulk of the objects
    markers, nmarkers = ndimage.label(local_maxi)
    print('markers')
    print_stats(markers)

    if show:
        plot_mask(local_maxi, f'local maximum; footprint={fs_size}')

    return local_maxi, markers, nmarkers


def touching_objects(heatmap: np.array, th: float, min_size: int, max_filt_size: int, fs_size: (int, int), show: bool = True):
    # first get cleaned mask
    t_mask, cleaned_mask, thresholded_objects = small_objects(heatmap, th, min_size, show=False)

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

    watershed_cleaned_mask = remove_small_objects(watershed_mask, min_size=min_size, connectivity=1)
    # labels_watershedcleaned, nlabels_cleaned = measure.label(watershed_mask, return_num=True, connectivity=1)
    fig_watershed_cleaned, ax_watershed_cleaned, cmap_watershed_cleaned = plot_mask(watershed_cleaned_mask,
                                                                                    f'watershed cleaned')
    plot_masks_comparison(ax_watershed, ax_watershed_cleaned, cmap_watershed,
                          title="Remove small objects after watershed")

    return watershed_mask, watershed_cleaned_mask


if __name__ == '__main__':
    # get sample mask
    fn: str = '11.png'
    heatmap: np.array = io.imread(DATA_PATH / '11.png', as_gray=True) / 255
    watershed_mask, watershed_cleaned_mask = touching_objects(heatmap, THRESHOLD, MIN_SIZE, MAX_FILTER_SIZE, FS_SIZE, show=False)

    # save masks without small objects
    io.imsave(DATA_PATH / f"{fn.split('.')[0]}-watershed.png", watershed_mask.astype('uint8') * 255,
              check_contrast=False)
    io.imsave(DATA_PATH / f"{fn.split('.')[0]}-watershed_cleaned.png", watershed_mask.astype('uint8') * 255,
              check_contrast=False)
