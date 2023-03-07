import numpy as np
from skimage import io, draw

from src.utils import plot_mask, DATA_PATH


def get_shapes_data():
    import random
    random.seed(17)

    mask = np.zeros(shape=(100, 100))

    # circle
    row, col = draw.disk((31, 20), 17)
    mask[row, col] = 1

    # ellipses
    row, col = draw.ellipse(81, 29, 6, 11)
    mask[row, col] = 2

    # rectangle/ square
    row, col = draw.rectangle(start=(66, 8), end=(77, 24))
    mask[row, col] = 3

    # polygon
    row, col = draw.polygon((7, 19, 26), (44, 95, 51))
    mask[row, col] = 4
    row, col = draw.polygon((26, 19, 23, 37, 34), (32, 46, 54, 47, 36))
    mask[row, col] = 5

    # holes
    row, col = draw.disk((72, 18), 1)
    mask[row, col] = 0
    row, col = draw.disk((19, 61), 4)
    mask[row, col] = 0
    row, col = draw.disk((35, 17), 3)
    mask[row, col] = 0

    nobjs = 5
    # crowded area
    for _ in range(8):
        # x, y = random.sample(range(60, 100), k=2)
        ax_x, ax_y = random.sample(range(2, 9), k=2)
        x = random.sample(range(50, 100), k=1)[0]
        y = random.sample(range(60, 100), k=1)[0]
        alpha = random.sample(range(91), k=1)[0]
        row, col = draw.ellipse(y, x, ax_y, ax_x, rotation=np.deg2rad(alpha), shape=mask.shape)
        nobjs += 1
        mask[row, col] = nobjs

    # add some noise
    for _ in range(8):
        # x, y = random.sample(range(100), k=2)
        x = random.sample(range(40, 80), k=1)[0]
        y = random.sample(range(40, 60), k=1)[0]
        r = random.sample(range(1, 4), k=1)[0]

        print("Sampled:", f"{x=}, {y=}, {r=}")
        # casually add disk or rectangle
        if random.random() > 0.5:
            row, col = draw.disk((y, x), r)
        else:
            row, col = draw.rectangle(start=(y, x), extent=r)
        mask[row, col] = nobjs

    # save
    io.imsave(DATA_PATH / 'ground-truth-mask.png', mask.astype('uint8') * 255, check_contrast=False)

    return mask


def decode_data(mask: np.array):
    """ Decode ground-truth mask by rescaling pixel values from 0 (background) to the number of objects."""
    vals = np.unique(mask)
    for encoded_val, decoded_val in zip(vals, range(len(vals))):
        mask[mask == encoded_val] = decoded_val

    return mask


if __name__ == '__main__':
    gt_mask = get_shapes_data()
    plot_mask(gt_mask, 'ground-truth')
