#!/bin/bash
#SBATCH --job-name=rnapro_preprocess
#SBATCH --partition=cpu              # CPU partition for preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Data Preprocessing Script (SLURM)
# =============================================================================
# Usage: sbatch slurm_preprocess.sh
# =============================================================================

set -e

mkdir -p logs

# =============================================================================
# Configuration
# =============================================================================

INPUT_DIR="./raw_structures"      # Directory with PDB/CIF files
OUTPUT_DIR="./data/train"         # Output directory for .pt files
MIN_LENGTH=10
MAX_LENGTH=512
FILE_FORMAT="both"                # "pdb", "cif", or "both"

# =============================================================================
# Run Preprocessing
# =============================================================================

echo "=========================================="
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python preprocess/prepare_design_data.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --file_format "${FILE_FORMAT}" \
    --min_length ${MIN_LENGTH} \
    --max_length ${MAX_LENGTH}

echo "Preprocessing complete!"
