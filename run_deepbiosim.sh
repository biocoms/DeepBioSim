#!/usr/bin/env bash

INPUT_CSV="./input/ibd.csv"
METHOD="iwae" # vae, iwae, diffusion, kde
OUTPUT_FOLDER="./output_sample/"
LATENT_DIM=16 # 8, 16, 32, 64
HIDDEN_DIM=128 # 64, 128, 256 as long as latent_dim <= hidden_dim
NUM_EPOCHS=100
BATCH_SIZE=64 # 32, 64, 128
LEARNING_RATE=0.001
K=10 # the larger the more accurate but slower
TIME_STEPS=3000 # the larger the more accurate but slower
RANDOM_SEED=42 # reproducibility

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
