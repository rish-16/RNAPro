# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration for RNA de novo structure design model.

This configuration removes sequence-dependent components and adds
design-specific settings for flow matching based generation.
"""

from rnapro.config.extend_types import (
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

# =============================================================================
# Basic Configuration
# =============================================================================

basic_configs = {
    "project": RequiredValue(str),
    "run_name": RequiredValue(str),
    "base_dir": RequiredValue(str),
    
    # Training settings
    "eval_interval": 1000,
    "log_interval": 100,
    "checkpoint_interval": 5000,
    "eval_first": False,
    "iters_to_accumulate": 1,
    "eval_only": False,
    
    # Checkpoint loading (follows RNAPro convention from README)
    # Use --load_checkpoint_path with --load_params_only True for fine-tuning
    # Use --load_strict False to allow missing keys (design model has different structure)
    "load_checkpoint_path": "",
    "load_ema_checkpoint_path": "",
    "load_strict": False,  # Allow missing keys for design model
    "load_params_only": True,  # Only load model params, not optimizer state
    "skip_load_step": False,
    "skip_load_optimizer": False,
    "skip_load_scheduler": False,
    
    # Fine-tuning options (for controlling which modules to train)
    "freeze_pairformer": False,    # Freeze pairformer stack during training
    "freeze_diffusion": False,     # Freeze diffusion module during training
    "find_unused_parameters": False,  # For DDP with frozen parameters
    
    # Logging
    "use_wandb": True,
    "wandb_id": "",
    
    # Reproducibility
    "seed": 42,
    "deterministic": False,
    "deterministic_seed": False,
    
    # EMA
    "ema_decay": 0.9999,
    "eval_ema_only": True,
    "ema_mutable_param_keywords": [""],
    
    # Model name
    "model_name": "rnapro_design_v0.1.0",
    
    # Design mode: "unconditional" or "conditional"
    "design_mode": "conditional",
    
    # Representation level
    "atom_level": "c4prime",  # "c4prime" (1 atom/residue) or "all_atom" (future)
}

# =============================================================================
# Data Configuration
# =============================================================================

data_configs = {
    # Data paths
    "train_data_dir": RequiredValue(str),
    "val_data_dir": "",
    
    # Data format options
    "use_pdb_directly": False,        # If True, load PDB/CIF files directly (slower)
    "pdb_file_pattern": "*.pdb",      # Glob pattern for PDB files
    "atom_selection": "c4prime",      # "c4prime", "backbone", or "all"
    
    # Data settings
    "max_length": 512,
    "min_length": 10,
    
    # Constraints
    "use_ss_constraints": False,
    "use_distance_constraints": False,
    
    # Augmentation
    "augment_coords": True,
    
    # DataLoader
    "batch_size": 4,
    "num_workers": 4,
}

# =============================================================================
# Optimizer Configuration
# =============================================================================

optim_configs = {
    # Learning rate
    "lr": 1e-4,
    "lr_scheduler": "cosine_annealing",
    "warmup_steps": 1000,
    "max_steps": RequiredValue(int),
    "min_lr_ratio": 0.01,
    "decay_every_n_steps": 50000,
    "grad_clip_norm": 1.0,
    
    # Adam optimizer
    "adam": {
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 1e-6,
        "lr": GlobalConfigValue("lr"),
        "use_adamw": True,
    },
    
    # Learning rate scheduler
    "cosine_lr_scheduler": {
        "warmup_steps": GlobalConfigValue("warmup_steps"),
        "max_steps": GlobalConfigValue("max_steps"),
        "min_lr_ratio": GlobalConfigValue("min_lr_ratio"),
    },
}

# =============================================================================
# Model Configuration
# =============================================================================

model_configs = {
    # Core dimensions (matching RNAPro for weight loading)
    "c_s": 384,
    "c_z": 128,
    "c_s_inputs": 449,
    "c_atom": 128,
    "c_atompair": 16,
    "c_token": 384,
    
    # Architecture
    "n_blocks": 48,
    "max_atoms_per_token": 24,
    "sigma_data": 16.0,
    
    # Flow matching settings
    "flow_sigma_min": 0.001,
    "flow_n_steps": 50,
    
    # Diffusion batch size (number of noisy samples per structure)
    "diffusion_batch_size": 16,
    "diffusion_chunk_size": ValueMaybeNone(4),
    
    # Checkpointing
    "blocks_per_ckpt": ValueMaybeNone(1),
    
    # Kernels
    "triangle_multiplicative": "cuequivariance",
    "triangle_attention": "cuequivariance",
    
    # Precision
    "dtype": "bf16",
    
    # Design-specific
    "max_length": 1024,
    "n_ss_classes": 4,
    "n_distance_bins": 64,
    
    # Inference settings
    "n_samples": 5,
    "infer_setting": {
        "chunk_size": ValueMaybeNone(64),
        "sample_diffusion_chunk_size": ValueMaybeNone(1),
    },
    
    # Sub-module configurations
    "model": {
        "N_model_seed": 1,
        "N_cycle": 4,
        
        # Relative position encoding (KEPT)
        "relative_position_encoding": {
            "r_max": 32,
            "s_max": 2,
            "c_z": GlobalConfigValue("c_z"),
        },
        
        # Pairformer (KEPT)
        "pairformer": {
            "n_heads": 16,
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_hidden_mul": 128,
            "c_hidden_pair_att": 32,
            "no_heads_pair": 4,
            "dropout": 0.0,  # Less dropout for design
            "n_blocks": GlobalConfigValue("n_blocks"),
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        
        # Diffusion module (KEPT)
        "diffusion_module": {
            "sigma_data": GlobalConfigValue("sigma_data"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s": GlobalConfigValue("c_s"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "c_atom": GlobalConfigValue("c_atom"),
            "c_atompair": GlobalConfigValue("c_atompair"),
            "c_token": GlobalConfigValue("c_token"),
            "n_blocks": 3,
            "n_heads": 16,
            "c_noise_embedding": 256,
            "blocks_per_ckpt": GlobalConfigValue("blocks_per_ckpt"),
        },
        
        # Condition embedder (NEW for design)
        "condition_embedder": {
            "c_s": GlobalConfigValue("c_s"),
            "c_z": GlobalConfigValue("c_z"),
            "c_s_inputs": GlobalConfigValue("c_s_inputs"),
            "max_length": GlobalConfigValue("max_length"),
            "n_ss_classes": GlobalConfigValue("n_ss_classes"),
            "n_distance_bins": GlobalConfigValue("n_distance_bins"),
            "use_learnable_base": True,
        },
    },
}

# =============================================================================
# Loss Configuration
# =============================================================================

loss_configs = {
    # Main flow matching loss
    "flow_loss_weight": 1.0,
    "loss_type": "mse",
    "timestep_weighting": False,
    
    # Structural regularization
    "structural_loss_weight": 0.1,
    "bond_loss_weight": 1.0,
    "clash_loss_weight": 0.1,
}

# =============================================================================
# Combined Configuration
# =============================================================================

configs = {
    **basic_configs,
    **data_configs,
    **optim_configs,
    **model_configs,
    "loss": loss_configs,
}


def get_design_configs():
    """Return a copy of the design configuration."""
    return configs.copy()


def create_design_config_from_rnapro(rnapro_configs):
    """
    Create design configuration from existing RNAPro config.
    
    This allows loading pretrained RNAPro weights into the design model.
    
    Args:
        rnapro_configs: Configuration from trained RNAPro model.
        
    Returns:
        Design configuration with compatible settings.
    """
    design_config = configs.copy()
    
    # Copy compatible settings
    compatible_keys = [
        "c_s", "c_z", "c_s_inputs", "c_atom", "c_atompair", "c_token",
        "n_blocks", "sigma_data", "blocks_per_ckpt",
    ]
    
    for key in compatible_keys:
        if key in rnapro_configs:
            design_config[key] = rnapro_configs[key]
    
    # Copy model sub-configs
    if "model" in rnapro_configs:
        if "relative_position_encoding" in rnapro_configs["model"]:
            design_config["model"]["relative_position_encoding"] = \
                rnapro_configs["model"]["relative_position_encoding"]
        
        if "pairformer" in rnapro_configs["model"]:
            design_config["model"]["pairformer"] = \
                rnapro_configs["model"]["pairformer"]
        
        if "diffusion_module" in rnapro_configs["model"]:
            design_config["model"]["diffusion_module"] = \
                rnapro_configs["model"]["diffusion_module"]
    
    return design_config
