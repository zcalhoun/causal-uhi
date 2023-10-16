#!/bin/bash

python joint_train.py \
    --data_dir ./data/durham/ \
    --output_dir ./results/joint_train/ \
    --max_iter 100 \
    --gp_constant_1 0.014 \
    --gp_noise 0.1 \
    --gp_constant_2 1e-9 \
    --gp_length_scale 220 \
    --gp_sigma_0 0.04 \
    --l2_alpha 1 \
    --ndvi_ls 7 \
    --n_samples 1000 \
    --fit_intercept
