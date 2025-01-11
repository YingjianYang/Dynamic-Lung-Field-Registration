#!/bin/bash

#nohup python -u acregnet_train.py \
#    --train_images_file=../data/DRLung/split/train_images.txt \
#    --train_labels_file=../data/DRLung/split/train_labels.txt \
#    --autoencoder_file=../results/DRLung/AENet/train/model.pt \
#    --epochs=100 \
#    --steps_per_epoch=240 \
#    --batch_size=24 \
#    --lr=1e-3 \
#    --flow_weight=5e-5 \
#    --label_weight=1.0 \
#    --shape_weight=1e-1 \
#    --results_dir=../results/DRLung/ACRegNet/train \
#    --save_model \
#    >../log/02_run_acregnet.log &
#
#python acregnet_test.py \
#    --test_images_file=../data/DRLung/split/valid_images.txt \
#    --test_labels_file=../data/DRLung/split/valid_labels.txt \
#    --model_file=../results/DRLung/ACRegNet/train/model.pt \
#    --results_dir=../results/DRLung/ACRegNet/valid \
#    --save_images
#
#python acregnet_compute_metrics.py \
#    --results_dir=../results/DRLung/ACRegNet/valid \
#    --output_dir=../results/DRLung/ACRegNet/valid/metrics



nohup python -u acregnet_train_v4.py \
    --train_images_file=../data/DRLung/split/train_images.txt \
    --train_labels_file=../data/DRLung/split/train_labels.txt \
    --autoencoder_file=../results/DRLung/AENet/train/model.pt \
    --epochs=20 \
    --batch_size=24 \
    --lr=1e-3 \
    --flow_weight=5e-5 \
    --label_weight=1.0 \
    --shape_weight=1e-1 \
    --results_dir=../results/DRLung/ACRegNet/train \
    --save_model \
    >../log/02_run_acregnet_v4.log &