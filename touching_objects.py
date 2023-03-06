import numpy as np
from scipy import ndimage

from src.utils import DATA_PATH
from src.utils import plot_heatmap, plot_mask, plot_masks_comparison, print_stats
from small_objects import small_objects, THRESHOLD, MIN_SIZE

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

FS_SIZE = (2, 2)

MAX_FILTER_SIZE = 20


def enhance_objects_core(mask: np.array, max_filt_size: int, show: bool = True):
    # get boolean mask for successive steps
    bool_mask = mask.astype(bool)
    print('boolean mask')
    print_stats(bool_mask)

    # compute distance of each pixel from the closest background pixel
    distance = ndimage.distance_transform_edt(bool_mask)
    plot_heatmap(distance, 'distance from borders', show=show)
    print('distance')
    print_stats(distance)

    # now apply maximum filtering to enhance the core part of objects; NOTE: play with `size` parameter
    maxi = ndimage.maximum_filter(distance, size=max_filt_size, mode='constant')
    fig, ax, cax = plot_heatmap(maxi, f'max filter; size={max_filt_size}', show=show)
    print('maxi')
    print_stats(maxi)
    return bool_mask, distance, maxi


def get_basins_markers(maxi: np.array, fs_size: (int, int), bool_mask: np.array, show: bool = True):
    # get indexes of local maxima
    local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones(fs_size), exclude_border=False,
                                labels=np.squeeze(bool_mask))

    plot_mask(local_maxi, f'local maximum; footprint={fs_size}', show=show)
    print('local maximum')
    print_stats(local_maxi)

    # once we have the enhanced mask, we get markers for the bulk of the objects
    markers, nmarkers = ndimage.label(local_maxi)
    print('markers')
    print_stats(markers)
    return local_maxi, markers, nmarkers


def touching_objects(th: float, min_size: int, max_filt_size: int, fs_size: (int, int)):
    # first get cleaned mask
    fn, heatmap, t_mask, cleaned_mask, thresholded_objects = small_objects(th, min_size, show=False)

    bool_mask, distance, maxi = enhance_objects_core(cleaned_mask, max_filt_size, show=False)

    # find local maxima; NOTE: play with `footprint` and `exclude_border` parameters
    local_maxi, markers, nmarkers = get_basins_markers(maxi, fs_size, bool_mask, show=False)

    # finally apply watershed to separate close-by objects; NOTE: play with `compactness` and `watershed_line` parameters
    watershed_mask = watershed(-distance, markers, mask=np.squeeze(local_maxi), compactness=1, watershed_line=True)
    print_stats(watershed_mask)
    plot_heatmap(watershed_mask, 'watershed')


    labels_cleaned, nlabels_cleaned = ndimage.label(watershed_mask)
    plot_mask(watershed_mask, f'watershed: {nlabels_cleaned} objects')

    return labels_cleaned, nlabels_cleaned


if __name__ == '__main__':
    touching_objects(THRESHOLD, MIN_SIZE, MAX_FILTER_SIZE, FS_SIZE)