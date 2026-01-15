#!/bin/bash
#SBATCH --job-name=rnapro_design_sample
#SBATCH --partition=gpu              # Change to your partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sample_%j.out
#SBATCH --error=logs/sample_%j.err

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# RNAProDesign Sampling Script (SLURM)
# =============================================================================
# Usage: sbatch slurm_sample_design.sh
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

# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint - UPDATE THIS
CHECKPOINT="./experiments/rnapro_design/checkpoints/checkpoint_latest.pt"

# Output directory
OUTPUT_DIR="./generated_structures/${SLURM_JOB_ID}"

# Generation settings
N_SAMPLES=10
N_STEPS=100
LENGTH=50
SEED=42

# Optional: Secondary structure constraint
# DOT_BRACKET="((((....))))"

# =============================================================================
# Run Sampling
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

python runner/sample_design.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}/unconditional" \
    --length ${LENGTH} \
    --n_samples ${N_SAMPLES} \
    --n_steps ${N_STEPS} \
    --seed ${SEED} \
    --save_pdb

# Optional: SS-constrained generation
if [ -n "${DOT_BRACKET}" ]; then
    echo "Generating SS-constrained structures..."
    python runner/sample_design.py \
        --checkpoint "${CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}/ss_constrained" \
        --dot_bracket "${DOT_BRACKET}" \
        --n_samples ${N_SAMPLES} \
        --n_steps ${N_STEPS} \
        --seed ${SEED} \
        --save_pdb
fi

echo "Sampling complete! Output: ${OUTPUT_DIR}"
