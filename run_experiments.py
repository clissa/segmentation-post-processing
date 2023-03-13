import itertools
from argparse import ArgumentParser, Namespace
from itertools import product

import numpy as np
from skimage import io

from src.utils import DATA_PATH
from touching_objects import touching_objects

parser = ArgumentParser()

parser.usage = "Run experiments with segmentation post-processing. Supports single grid search:" \
               "" \
               "python run_experiments.py --filt-size 5 10 15 --fp-size 2 4" \
               "" \
               "or nested `fp-size` (by repeating the corresponding argument):" \
               "" \
               "python run_experiments.py --filt-size 5 10 15 --fp-size 2 4 --fp-size 2 5 8"

parser.add_argument('-fs', '--filt-size', help="Maximum filter size.", nargs='+', type=int, dest='max_filt_size')
parser.add_argument('-fp', '--fp-size',
                    help="Footprint size for `peak_local_max`. Supports both flat and nested lists (i.e. one list per each `filt-size`.",
                    nargs='+', type=int, action='append')
parser.add_argument('--notes', help="Experiment description to upload on Weights & Biases.", type=str, default='',
                    nargs='?')

# use the following for testing ...
# cli_call: str = "--filt-size 5 10 15 --fp-size 2 4 --fp-size 2 5 8"
# args: Namespace = parser.parse_args(args=cli_call.split(' ') + ["--notes", "Simple experiment"])

# ... or use this for real
args: Namespace = parser.parse_args()

config = {'file': ['11.png', '39.png', '1152.png'], 'threshold': [0.5], 'min_size': [20, 50, 100],
          'max_filt_size': args.max_filt_size, 'footprint_size': args.fp_size}

param_space: itertools.product = product(*config.values())
n_params = len(config['file']) * len(config['min_size']) * len(config['max_filt_size']) * len(config['footprint_size'])


def main(fn: str, threshold: float, min_size: int, max_filt_size: int, fp: int):
    heatmap: np.array = io.imread(DATA_PATH / fn, as_gray=True) / 255
    footprint_size = (fp, fp)
    watershed_mask, watershed_cleaned_mask = touching_objects(heatmap, threshold, min_size, max_filt_size,
                                                              footprint_size, show=False)
    from src.utils import print_stats
    print_stats(watershed_mask)
    row_data = [fn, threshold, min_size, max_filt_size, footprint_size,
                wandb.Image(heatmap * 255),
                wandb.Image(watershed_mask, masks={"predictions": {"mask_data": watershed_mask}}),
                wandb.Image(watershed_cleaned_mask, masks={"predictions": {"mask_data": watershed_cleaned_mask}})]

    return row_data


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('Agg')

    from tqdm.auto import tqdm
    import wandb

    print(f"{'Starting experiments':-^100}")
    print(f"{'N. configs: ' + str(n_params):.<100}")
    print(f"\n{'':-<100}\n")
    run = wandb.init(project="segmentation-post-processing", notes=args.notes, config=config)

    # initialize results table
    param_names = 'file threshold min_size filt_size fp_size'.split(' ')
    table = wandb.Table(columns=param_names + ['heatmap', 'watershed', 'watershed-cleaned'])

    for fn, threshold, min_size, max_filt_size, fps in tqdm(param_space, total=n_params):
        # min_size = 50 if fn == '11.png' else 100
        data = []
        if isinstance(fps, list):
            for fp in fps:
                data.append(main(fn, threshold, min_size, max_filt_size, fp))
        else:
            data.append(main(fn, threshold, min_size, max_filt_size, fps))

        for row in data:
            print(row)
            table.add_data(*row)

    run.log({"Results/post-processing": table})
