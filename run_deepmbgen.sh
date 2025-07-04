#!/usr/bin/env bash

INPUT_CSV="./input/ibd.csv"
METHOD="diffusion"
OUTPUT_FOLDER="./output_sample/"
LATENT_DIM=16
HIDDEN_DIM=128
NUM_EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.001
K=10
TIME_STEPS=3000
RANDOM_SEED=42

# Ensure the CLI script is executable or call via python
python run_deepmbgen.py \
  --input $INPUT_CSV \
  --method $METHOD \
  --output_folder $OUTPUT_FOLDER \
    # --latent_dim $LATENT_DIM \
    # --hidden_dim $HIDDEN_DIM \
    # --num_epochs $NUM_EPOCHS \
    # --batch_size $BATCH_SIZE \
    # --learning_rate $LEARNING_RATE \
    # --k $K \
    # --time_steps $TIME_STEPS \
    # --random_seed $RANDOM_SEED

# Example:
# ./run_sim_single.sh data/ibd.csv vae output/ --latent_dim 32 --num_epochs 200
