import matplotlib.pyplot as plt
import numpy as np
from skimage import draw, measure
from src.utils import plot_mask


def shapes_data():
    gt_mask = np.zeros(shape=(100, 100))

    # circle
    row, col = draw.disk((31, 20), 17)
    gt_mask[row, col] = 1

    # ellipses
    row, col = draw.ellipse(81, 29, 6, 11)
    gt_mask[row, col] = 2

    # rectangle/ square
    row, col = draw.rectangle(start=(66, 8), end=(77, 24))
    gt_mask[row, col] = 3

    # polygon
    row, col = draw.polygon((7, 19, 26), (44, 95, 51))
    gt_mask[row, col] = 4
    row, col = draw.polygon((26, 19, 23, 37, 34), (32, 46, 53, 47, 36))
    gt_mask[row, col] = 5

    # holes
    row, col = draw.disk((72, 18), 1)
    gt_mask[row, col] = 0
    row, col = draw.disk((19, 61), 4)
    gt_mask[row, col] = 0
    row, col = draw.disk((35, 17), 3)
    gt_mask[row, col] = 0

    return gt_mask

if __name__ == '__main__':
    gt_mask = shapes_data()
    plot_mask(gt_mask, 'ground-truth')