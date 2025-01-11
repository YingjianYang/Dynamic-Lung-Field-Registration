#!/bin/bash

nohup python -u aenet_train.py \
    --train_labels_file=../data/DRLung/split/train_labels.txt \
    --epochs=100 \
    --steps_per_epoch=100 \
    --batch_size=16 \
    --lr=1e-3 \
    --results_dir=../results/DRLung/AENet/train \
    --save_model \
    >../log/01_run_aenet.log &

#python aenet_test.py \
#    --test_labels_file=../data/DRLung/split/valid_labels.txt \
#    --model_file=../results/DRLung/AENet/train/model.pt \
#    --results_dir=../results/DRLung/AENet/valid \
#    --save_images
#
#python aenet_compute_metrics.py \
#    --results_dir=../results/DRLung/AENet/valid \
#    --output_dir=../results/DRLung/AENet/valid/metrics