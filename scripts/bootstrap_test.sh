#!/bin/bash

python bootstrap_coefficients.py \
    --data_dir ./data/durham/ \
    --output_dir ./results/bootstrap_test/ \
    --num_blocks 100 \
    --ndvi_ls 13 \
    --albedo_ls 2 \
    --use_coords \
    --log_level INFO