#!/bin/bash

#SBATCH --mail-user=zachary.calhoun@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=./f17.out
#SBATCH --error=./f17.err
#SBATCH --mem=64G

source ~/.bashrc
conda activate uhi

python calc_optimal_weights.py \
    --data_dir ./data/durham/ \
    --output_dir ./results/final/f17/ \
    --k_folds 5 \
    --k_folds_size 10 \
    --l2_alpha 5 \
    --gp_noise 1e-1 \
    --window_size 25 \
    --ndvi_ls_vals 1 30 \
    --albedo_ls_vals 1 20 \
    --gp_constant_1 0.5 \
    --gp_constant_2 3e-7 \
    --gp_length_scale 115 \
    --gp_sigma_0 0.1 \
    --max_iter 10 \
    --log_level INFO