#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example training script for RNA de novo structure design
# This fine-tunes from pretrained RNAPro using Flow Matching at C4' level

set -e

# =============================================================================
# Configuration
# =============================================================================

PROJECT="rnapro_design"
RUN_NAME="design_finetune_c4prime_001"
BASE_DIR="./experiments"

# Data paths (update these to your data locations)
TRAIN_DATA_DIR="./data/train"
VAL_DATA_DIR="./data/val"

# Pretrained RNAPro checkpoint (REQUIRED for fine-tuning)
# This should be a checkpoint from trained RNAPro structure prediction model
# Use the same --load_checkpoint_path convention as original RNAPro
LOAD_CHECKPOINT_PATH="./release_data/protenix_models/protenix_base_default_v0.5.0.pt"

# Training settings
MAX_STEPS=50000
BATCH_SIZE=4
LEARNING_RATE=1e-4

# Model settings
N_CYCLE=4
FLOW_N_STEPS=50

# Fine-tuning options:
# - Set to True to freeze pretrained weights (only train new embedder)
# - Set to False to fine-tune all weights (recommended)
FREEZE_PAIRFORMER=False
FREEZE_DIFFUSION=False

# Atom level representation
ATOM_LEVEL="c4prime"  # C4' atoms only (one per residue)

# =============================================================================
# Run Training
# =============================================================================

echo "=========================================="
echo "RNA De Novo Structure Design - Fine-tuning"
echo "=========================================="
echo "Pretrained checkpoint: ${LOAD_CHECKPOINT_PATH}"
echo "Atom level: ${ATOM_LEVEL}"
echo "Freeze pairformer: ${FREEZE_PAIRFORMER}"
echo "Freeze diffusion: ${FREEZE_DIFFUSION}"
echo "=========================================="

python runner/train_design.py \
    --project "${PROJECT}" \
    --run_name "${RUN_NAME}" \
    --base_dir "${BASE_DIR}" \
    --train_data_dir "${TRAIN_DATA_DIR}" \
    --val_data_dir "${VAL_DATA_DIR}" \
    --load_checkpoint_path "${LOAD_CHECKPOINT_PATH}" \
    --load_strict False \
    --load_params_only True \
    --freeze_pairformer ${FREEZE_PAIRFORMER} \
    --freeze_diffusion ${FREEZE_DIFFUSION} \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --model.N_cycle ${N_CYCLE} \
    --flow_n_steps ${FLOW_N_STEPS} \
    --design_mode "conditional" \
    --atom_level "${ATOM_LEVEL}" \
    --use_ss_constraints False \
    --use_distance_constraints False \
    --use_wandb True \
    --eval_interval 1000 \
    --checkpoint_interval 5000 \
    "$@"

echo "Training complete!"
