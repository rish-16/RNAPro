# RNAProDesign: Fine-tuning from Pretrained RNAPro

This document explains how to fine-tune RNAProDesign from pretrained RNAPro checkpoints for de novo RNA 3D structure design.

## Overview

RNAProDesign is a generative model that learns to design RNA 3D structures using **Flow Matching**. It fine-tunes from pretrained RNAPro structure prediction weights, specifically:

- **PairformerStack**: Core representation learning module
- **DiffusionModule**: Structure denoising/generation module
- **RelativePositionEncoding**: Positional information embedding
- **Recycling layers**: Inter-cycle communication layers

The sequence-dependent components (MSA, templates, sequence embedder) are **removed** and replaced with learnable structure condition embedders.

## Architecture

```
Design Constraints → ConditionEmbedder → Pairformer (pretrained) → FlowMatching → Structure
                      (NEW)              (FINE-TUNED)               (NEW loss)
```

### What's Loaded from Pretrained
- `pairformer_stack.*` - All pairformer layers
- `diffusion_module.*` - All diffusion module layers  
- `relative_position_encoding.*` - Position encoding
- `layernorm_z_cycle.*`, `linear_no_bias_z_cycle.*` - Z recycling
- `layernorm_s_cycle.*`, `linear_no_bias_s_cycle.*` - S recycling
- `linear_no_bias_token_bond.*` - Token bond embedding

### What's Randomly Initialized (NEW)
- `condition_embedder.*` - Structure condition embedding

## C4' Level Representation

The current implementation operates at **C4' level** (one atom per residue):
- `n_atoms == n_tokens == n_residues`
- Coordinates shape: `[B, N_residues, 3]`
- Simplifies training and inference
- Can be extended to all-atom backbone later

## Quick Start

### 1. Prepare Your Data

Your training data should be preprocessed `.pt` files containing:
```python
{
    "coordinates": torch.Tensor,  # [N_residues, 3] C4' coordinates
    "coordinate_mask": torch.Tensor,  # [N_residues] validity mask
    "n_tokens": int,  # Number of tokens (= N_residues for C4')
}
```

### 2. Configure Fine-tuning

Key configuration options follow the same convention as the main RNAPro README:

```python
# Path to pretrained RNAPro/Protenix checkpoint (same as --load_checkpoint_path in RNAPro)
"load_checkpoint_path": "/path/to/protenix_base_default_v0.5.0.pt",

# Required flags for fine-tuning (same as RNAPro)
"load_strict": False,      # Allow missing keys (design model has different structure)
"load_params_only": True,  # Only load model params, not optimizer state

# Freeze options (set True to freeze pretrained weights)
"freeze_pairformer": False,  # Recommended: False (fine-tune all)
"freeze_diffusion": False,   # Recommended: False (fine-tune all)

# Required for DDP if freezing modules
"find_unused_parameters": True,  # Set True if freezing any module

# Representation level
"atom_level": "c4prime",
```

### 3. Run Fine-tuning

```bash
# Edit the example script with your paths
./rnapro_design_train_example.sh
```

Or run directly (following RNAPro README convention):
```bash
python runner/train_design.py \
    --load_checkpoint_path ./release_data/protenix_models/protenix_base_default_v0.5.0.pt \
    --load_strict False \
    --load_params_only True \
    --train_data_dir /path/to/train \
    --max_steps 50000 \
    --freeze_pairformer False \
    --freeze_diffusion False
```

## Training Modes

### Full Fine-tuning (Recommended)
```bash
--freeze_pairformer False --freeze_diffusion False
```
- All weights are trainable
- Best performance but requires more GPU memory
- ~100M+ parameters to optimize

### Partial Fine-tuning
```bash
--freeze_pairformer True --freeze_diffusion False
```
- Only diffusion module and embedder are trained
- Faster training, less memory
- Use if you have limited data

### Feature Extraction
```bash
--freeze_pairformer True --freeze_diffusion True
```
- Only condition embedder is trained
- Fastest but may have limited performance
- Good for quick experiments

## Flow Matching Training

Flow matching learns a velocity field that transports noise to data:
- Sample timestep `t ~ U(0, 1)`
- Interpolate: `x_t = (1-t) * noise + t * data`
- Target velocity: `v = data - noise`
- Loss: `MSE(v_pred, v_target)`

Key hyperparameters:
```python
"flow_sigma_min": 0.001,  # Minimum noise level
"flow_n_steps": 50,       # Integration steps at inference
"sigma_data": 16.0,       # Data normalization
```

## Sampling

After training, generate structures:

```bash
./rnapro_design_sample_example.sh
```

Or:
```bash
python runner/sample_design.py \
    --checkpoint /path/to/design_checkpoint.pt \
    --length 50 \
    --n_samples 10 \
    --n_steps 100
```

## Weight Mapping

Some weights have different names between RNAPro and RNAProDesign:

| RNAPro | RNAProDesign |
|--------|--------------|
| `linear_no_bias_s.*` | `linear_no_bias_s_cycle.*` |
| `layernorm_s.*` | `layernorm_s_cycle.*` |

The `load_pretrained_rnapro()` method handles this automatically.

## Debugging

Check parameter loading:
```python
from rnapro.model.RNAProDesign import RNAProDesign

model = RNAProDesign(configs)
info = model.get_trainable_params_info()
print(info)
# Output: {'condition_embedder': {'total': X, 'trainable': X, 'frozen': 0}, ...}
```

## Extending to All-Atom

To extend from C4' to all-atom backbone:

1. Update `atom_level` config: `"atom_level": "all_atom"`
2. Modify dataset to return all backbone atoms
3. Update `atom_to_token_idx` mapping
4. Adjust `n_atoms != n_tokens` handling in model

## Common Issues

### Shape Mismatch During Loading
If you see shape mismatch warnings, check that your pretrained checkpoint was trained with the same architecture dimensions (`c_s`, `c_z`, etc.).

### OOM During Training
- Reduce `batch_size`
- Reduce `diffusion_batch_size` 
- Enable gradient checkpointing via `blocks_per_ckpt`
- Use partial fine-tuning

### DDP Errors with Frozen Parameters
Set `find_unused_parameters=True` in config when freezing any module.
