import argparse
import os
import time
import numpy as np
import torch

from acregnet.models import ACRegNet
from acregnet.modules import SpatialTransformer
from acregnet.datasets import ImagePairsDataset_V1
from acregnet.utils.io import save_image
from acregnet.utils.tensor import rescale_intensity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_images_file', type=str, default='../data/DRLung/split/test_images.txt')
    parser.add_argument('--test_labels_file', type=str, default='../data/DRLung/split/test_labels.txt')
    parser.add_argument('--model_file', type=str, default='../results/DRLung/ACRegNet/train/model_v3.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default='../results/DRLung/ACRegNet/test')
    parser.add_argument('--save_images', action='store_true', default=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    os.makedirs(args.results_dir, exist_ok=True)

    if args.save_images:
        output_dir = os.path.join(args.results_dir, 'output_v3')

    test_dataset = ImagePairsDataset_V1(args.test_images_file, args.test_labels_file, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    model = ACRegNet(test_dataset.input_size, test_dataset.num_labels, args, mode='test')
    model.load(args.model_file)
    transformer = SpatialTransformer(test_dataset.input_size, mode='nearest').to(args.device)

    num_pairs = len(test_loader.dataset)

    print('\nTesting:\n')
    with torch.no_grad():

        t_start = time.time()
        for i, (images, labels, names) in enumerate(test_loader):
            print(f'\t{i+1:>3}/{num_pairs}', end=' ', flush=True)

            moving, fixed = images
            moving, fixed = moving.to(args.device), fixed.to(args.device)

            moving_label, fixed_label = labels
            moving_label, fixed_label = moving_label.to(args.device), fixed_label.to(args.device)

            moving_name, fix_name = names[0][0], names[1][0]

            # Run registration
            output, flow = model.register(moving, fixed)

            # Transform moving label
            output_label = transformer(moving_label, flow)

            if args.save_images:
                cur_out_dir = os.path.join(output_dir, f'{moving_name}to{fix_name}')
                os.makedirs(cur_out_dir, exist_ok=True)

                save_image(moving * 255., os.path.join(cur_out_dir, f'im_mov_{moving_name}.png'))
                save_image(fixed * 255., os.path.join(cur_out_dir, f'im_fix_{fix_name}.png'))
                save_image(output * 255., os.path.join(cur_out_dir, f'im_out_{moving_name}to{fix_name}.png'))

                save_image(rescale_intensity(moving_label), os.path.join(cur_out_dir, f'lb_mov_{moving_name}.png'))
                save_image(rescale_intensity(fixed_label), os.path.join(cur_out_dir, f'lb_fix_{fix_name}.png'))
                save_image(rescale_intensity(output_label), os.path.join(cur_out_dir, f'lb_out_{moving_name}to{fix_name}.png'))

                np.save(os.path.join(cur_out_dir, 'flow.npy'), flow.squeeze().permute(1, 2, 0).cpu().numpy())

            print(f'({(time.time() - t_start):.2f} sec)')
            t_start = time.time()

    with open(os.path.join(args.results_dir, 'run_args_v3.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
