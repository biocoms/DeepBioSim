#!/usr/bin/env bash

INPUT_CSV="./input/ibd.csv" # Make sure each row represents a sample and each column represents a taxa/gene. Usually there are more taxa than samples. Index column and headers should be included. See example input file in ./input.
METHOD="diffusion" # available simulation methods: vae, iwae, diffusion, kde. kde may take forever to run with high dimensional data with many samples.
OUTPUT_FOLDER="./output_sample/"

# **All parameters below are optional**

# ==== These parameters don't have to be the power of 2 ====
LATENT_DIM=16 # 8, 16, 32, 64 # as long as latent_dim <= number of samples
HIDDEN_DIM=128 # 64, 128, 256 as long as latent_dim <= hidden_dim
BATCH_SIZE=64 # 32, 64, 128
# ==========================================================

NUM_EPOCHS=100 # No need to adjust
LEARNING_RATE=0.001 # No need to adjust
K=10 # (for iwae only) the larger the more accurate but slower
TIME_STEPS=3000 # (for diffusion only) the larger the more accurate but slower
RANDOM_SEED=42 # reproducibility

# Run the simulation
python run_deepbiosim.py \
  --input $INPUT_CSV \
  --method $METHOD \
  --output_folder $OUTPUT_FOLDER \
  --latent_dim $LATENT_DIM \
  --hidden_dim $HIDDEN_DIM \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --K $K \
  --time_steps $TIME_STEPS \
  --random_seed $RANDOM_SEED
