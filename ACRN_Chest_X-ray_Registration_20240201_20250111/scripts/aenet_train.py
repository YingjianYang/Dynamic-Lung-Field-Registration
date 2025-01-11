import argparse
import os
import random
import time
import numpy as np
import torch
from tqdm import trange
import sys

sys.path.append('../')
from acregnet.models import AENet
from acregnet.datasets import LabelsDataset
from acregnet.dataloader import InfiniteDataLoader
from acregnet.utils.io import save_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_labels_file', type=str, default='../data/DRLung/split/train_labels.txt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default='../results/DRLung/AENet/train')
    parser.add_argument('--save_model', action='store_true', default=True)
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.results_dir, exist_ok=True)

    train_dataset = LabelsDataset(args.train_labels_file)
    # train_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               drop_last=True)

    model = AENet(train_dataset.input_size, train_dataset.num_labels, args)

    train_results = {
        'train_loss': [],
        'train_time': []
    }

    print('\nTraining:\n')
    for epoch in trange(args.epochs):

        train_loss = 0.0
        train_time = 0.0

        t_start = time.time()
        # for step in range(args.steps_per_epoch):
        #     inputs, target = next(train_loader)
        for inputs, target in train_loader:
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            target = torch.argmax(target, dim=1).long()  # CrossEntropyLoss不需要onehot编码

            # Training
            loss = model.train(inputs, target)

            train_loss += loss
            # train_time += (time.time() - t_start)
            # t_start = time.time()

        # train_time = train_time / args.steps_per_epoch
        train_time = time.time() - t_start
        train_results['train_time'].append(train_time)

        # train_loss = train_loss / args.steps_per_epoch
        train_loss = train_loss / len(train_loader)
        train_results['train_loss'].append(train_loss)

        if args.save_model:
            model.save(os.path.join(args.results_dir, 'model.pt'))

        print(f'\tEpoch {epoch + 1:>3}/{args.epochs} '
              f'train_loss: {train_loss:.4f} '
              f'epoch_time: {train_time:.2f} sec')

        # print(f'\tEpoch {epoch + 1:>3}/{args.epochs} '
        #       f'train_loss: {train_loss:.4f} '
        #      f'step_time: {train_time:.2f} sec')

    if args.save_model:
        model.save(os.path.join(args.results_dir, 'model.pt'))

    save_dict(train_results, os.path.join(args.results_dir, 'train_results.pkl'))

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
