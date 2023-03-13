from argparse import ArgumentParser, Namespace

import numpy as np
from scipy import ndimage
from skimage import io
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from small_objects import small_objects  # , THRESHOLD, MIN_SIZE
from src.utils import plot_heatmap, plot_mask, print_stats, plot_masks_comparison, DATA_PATH


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


def get_basin_markers(maxi: np.array, fp_size: (int, int), bool_mask: np.array, show: bool = True, min_size: int = 0):
    # get indexes of local maxima
    local_maxi = peak_local_max(np.squeeze(maxi), indices=False, footprint=np.ones(fp_size), exclude_border=False,
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
        plot_mask(local_maxi, f'local maximum; footprint={fp_size}')

    return local_maxi, markers, nmarkers


def touching_objects(heatmap: np.array, th: float, min_size: int, max_filt_size: int, fp_size: (int, int),
                     show: bool = True):
    # first get cleaned mask
    t_mask, cleaned_mask, thresholded_objects = small_objects(heatmap, th, min_size, show=False)

    # find local maxima; NOTE: play with `max_filter_size` and `footprint` parameters
    bool_mask, distance, maxi = enhance_objects_basin(cleaned_mask, max_filt_size, show=show)
    local_maxi, markers, nmarkers = get_basin_markers(maxi, fp_size, bool_mask, show=show)

    # finally apply watershed to separate close-by objects; NOTE: play with `compactness` and `watershed_line` parameters
    watershed_mask = watershed(-distance, markers, mask=np.squeeze(bool_mask), compactness=1, watershed_line=False)
    if show:
        plot_heatmap(watershed_mask, 'watershed')
        print_stats(watershed_mask)

    fig_clean, ax_clean, _ = plot_mask(cleaned_mask, "cleaned mask")
    fig_watershed, ax_watershed, cmap_watershed = plot_mask(watershed_mask, "watershed")
    plot_masks_comparison(ax_clean, ax_watershed, cmap_watershed, title="Separe close/overlapping objects")

    local_maxi_cleaned, markers_cleaned, nmarkers_cleaned = get_basin_markers(maxi, fp_size, bool_mask, min_size=10)
    watershed_cleaned_mask = watershed(-distance, markers_cleaned, mask=np.squeeze(bool_mask), compactness=1,
                                       watershed_line=False)
    fig_watershed_cleaned, ax_watershed_cleaned, cmap_watershed_cleaned = plot_mask(watershed_cleaned_mask,
                                                                                    f'watershed cleaned')
    plot_masks_comparison(ax_watershed, ax_watershed_cleaned, cmap_watershed,
                          title="Remove small objects after watershed")

    return watershed_mask, watershed_cleaned_mask


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.usage = "Read heatmap and remove small objects that pass the thresholding."

    parser.add_argument('fn',
                        help="Filename of the heatmap to post-process. This assumes files are in `DATA_PATH` folder.",
                        type=str)
    parser.add_argument('threshold', help="Binarization threshold.", type=float, default=0.5, nargs='?')
    parser.add_argument('min_size', help="Minimum allowed object size.", type=int, default=100, nargs='?')
    parser.add_argument('max_filt_size', help="Filter size used in maximum filter.", type=int, default=4, nargs='?')
    parser.add_argument('fp_size', help="Footprint size used in to find local maxima (`peak_local_max`).", type=int,
                        default=6, nargs='?')
    args: Namespace = parser.parse_args()

    # get sample mask
    # fn: str = '11.png'
    heatmap: np.array = io.imread(DATA_PATH / args.fn, as_gray=True) / 255
    footprint_size = (args.fp_size, args.fp_size)
    watershed_mask, watershed_cleaned_mask = touching_objects(heatmap, args.threshold, args.min_size,
                                                              args.max_filt_size, footprint_size, show=False)

    # save masks without small objects
    outpath = DATA_PATH / 'post-processed'
    outpath.mkdir(parents=True, exist_ok=True)

    io.imsave(outpath / f"{args.fn.split('.')[0]}-watershed.png", watershed_mask.astype('uint8') * 255,
              check_contrast=False)
    io.imsave(outpath / f"{args.fn.split('.')[0]}-watershed_cleaned.png", watershed_mask.astype('uint8') * 255,
              check_contrast=False)
