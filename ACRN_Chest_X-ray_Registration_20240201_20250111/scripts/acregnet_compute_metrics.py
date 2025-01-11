import argparse
import os
import glob
import time
import pandas as pd

from acregnet.utils.io import read_image
from acregnet.utils.metric import dc, hd, assd

if __name__ == '__main__':
    version = 'v4'
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='../results/DRLung/ACRegNet/test')
    parser.add_argument('--output_dir', type=str, default=f'../results/DRLung/ACRegNet/test/metrics_{version}')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    os.makedirs(args.output_dir, exist_ok=True)

    test_images = sorted(glob.glob(os.path.join(args.results_dir, f'output_{version}/*')))
    num_pairs = len(test_images)

    results_pre = {'dsc': [], 'hd': [], 'assd': []}
    results_pos = {'dsc': [], 'hd': [], 'assd': []}

    print('\nComputing metrics and saving results:\n')
    for i, image_dir in enumerate(test_images):
        print(f'\t{i + 1:>3}/{num_pairs}', end=' ', flush=True)
        t_start = time.time()

        moving_name = image_dir.split('/')[-1][:2]
        fix_name = image_dir.split('/')[-1][-2:]

        lb_mov = read_image(os.path.join(image_dir, f'lb_mov_{moving_name}.png'))
        lb_fix = read_image(os.path.join(image_dir, f'lb_fix_{fix_name}.png'))
        lb_out = read_image(os.path.join(image_dir, f'lb_out_{moving_name}to{fix_name}.png'))

        # Compute metrics before registration
        results_pre['dsc'].append(dc(lb_mov, lb_fix))
        results_pre['hd'].append(hd(lb_mov, lb_fix))
        results_pre['assd'].append(assd(lb_mov, lb_fix))

        # Compute metrics after registration
        results_pos['dsc'].append(dc(lb_out, lb_fix))
        results_pos['hd'].append(hd(lb_out, lb_fix))
        results_pos['assd'].append(assd(lb_out, lb_fix))

        print(f'({(time.time() - t_start):.2f} sec)')

    # Saving results to csv files
    data_pre = pd.DataFrame.from_dict(results_pre)
    data_pre.to_csv(os.path.join(args.output_dir, f'metrics_pre.csv'), index=False)
    data_pos = pd.DataFrame.from_dict(results_pos)
    data_pos.to_csv(os.path.join(args.output_dir, f'metrics_pos.csv'), index=False)

    print('\nDone.\n')
