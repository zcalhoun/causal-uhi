#!/bin/bash

python bootstrap_coefficients.py \
    --data_dir ./data/durham/ \
    --output_dir ./results/bootstrap_final/ \
    --num_blocks 50 \
    --l2_alpha 0.1 \
    --ndvi_ls 16 \
    --albedo_ls 7 \
    --gp_noise 1e-1 \
    --gp_constant_1 0.5 \
    --gp_constant_2 3e-7 \
    --gp_length_scale 115 \
    --gp_sigma_0 0.1 \
    --iterate \
    --log_level INFO