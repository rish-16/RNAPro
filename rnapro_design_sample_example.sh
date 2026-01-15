#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Example sampling script for RNA de novo structure design
# Generates 3D structures using a fine-tuned RNAProDesign model

set -e

# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint (REQUIRED)
# This should be a checkpoint from fine-tuned RNAProDesign model
CHECKPOINT="./experiments/rnapro_design/design_finetune_c4prime_001/checkpoints/checkpoint_latest.pt"
OUTPUT_DIR="./generated_structures"

# Generation settings
N_SAMPLES=10           # Number of structures per input
N_STEPS=50             # Integration steps (more = higher quality, slower)
LENGTH=50              # Sequence length for unconditional generation
SEED=42

# Uncomment to use secondary structure constraint
# DOT_BRACKET="((((....))))"

# =============================================================================
# Run Sampling
# =============================================================================

echo "=========================================="
echo "RNA De Novo Structure Design - Sampling"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples: ${N_SAMPLES}"
echo "Integration steps: ${N_STEPS}"
echo "=========================================="

# Run unconditional generation
echo "Generating unconditional RNA structures..."
python runner/sample_design.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}/unconditional" \
    --length ${LENGTH} \
    --n_samples ${N_SAMPLES} \
    --n_steps ${N_STEPS} \
    --seed ${SEED} \
    --save_pdb

# Run constraint-conditioned generation (if DOT_BRACKET is set)
if [ -n "${DOT_BRACKET}" ]; then
    echo "Generating SS-constrained RNA structures..."
    python runner/sample_design.py \
        --checkpoint "${CHECKPOINT}" \
        --output_dir "${OUTPUT_DIR}/ss_constrained" \
        --dot_bracket "${DOT_BRACKET}" \
        --n_samples ${N_SAMPLES} \
        --n_steps ${N_STEPS} \
        --seed ${SEED} \
        --save_pdb
fi

echo "Generation complete! Output saved to ${OUTPUT_DIR}"
