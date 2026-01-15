#!/bin/bash
#SBATCH --job-name=rnapro_design_train
#SBATCH --partition=gpu              # Change to your partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # Number of GPUs
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# RNAProDesign Training Script (SLURM)
# =============================================================================
# Usage: sbatch slurm_train_design.sh
# =============================================================================

set -e

# Create log directory
mkdir -p logs

# =============================================================================
# Environment Setup
# =============================================================================

# Activate conda environment (modify as needed)
# source ~/.bashrc
# conda activate rnapro

# Or use module system
# module load cuda/12.0
# module load python/3.12

# =============================================================================
# Configuration
# =============================================================================

PROJECT="rnapro_design"
RUN_NAME="design_finetune_${SLURM_JOB_ID}"
BASE_DIR="./experiments"

# Data paths
TRAIN_DATA_DIR="./data/train"
VAL_DATA_DIR="./data/val"

# Pretrained checkpoint
LOAD_CHECKPOINT_PATH="./release_data/protenix_models/protenix_base_default_v0.5.0.pt"

# Training settings
MAX_STEPS=50000
BATCH_SIZE=4
LEARNING_RATE=1e-4

# Model settings
N_CYCLE=4
FLOW_N_STEPS=50

# Fine-tuning options
FREEZE_PAIRFORMER=False
FREEZE_DIFFUSION=False

# =============================================================================
# Multi-GPU Setup
# =============================================================================

# Get master address for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_ON_NODE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================================="

# =============================================================================
# Run Training
# =============================================================================

srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=${SLURM_GPUS_ON_NODE} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    runner/train_design.py \
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
    --atom_level "c4prime" \
    --use_wandb True \
    --eval_interval 1000 \
    --checkpoint_interval 5000 \
    --num_workers 8

echo "Training complete!"
