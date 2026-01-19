# RNAProducer: De Novo RNA 3D Structure Design

Fine-tune the pretrained RNAPro diffusion model for unconditional RNA structure generation.

---

## Overview

RNAPro already contains a complete diffusion-based generative model:

```
RNAProDesign Model:
├── Position Embeddings (learnable, replaces sequence features)
├── PairformerStack (48 blocks, pretrained)
└── DiffusionModule (EDM-style, pretrained)
    ├── DiffusionConditioning (Fourier + RelPosEnc)
    ├── AtomAttentionEncoder (3 blocks)
    ├── DiffusionTransformer (24 blocks, 16 heads)
    └── AtomAttentionDecoder (3 blocks)
```

**Training**: Add noise to C4' coordinates → Denoise → Weighted MSE loss

**Inference**: Sample from noise using predictor-corrector sampling

---

## Quick Start

### Step 1: Prepare Data

```bash
# Create data directory
mkdir -p ./data/rnasolo

# Download RNAsolo equivalence classes (PDB format)
# Visit: https://rnasolo.cs.put.poznan.pl/
# Download: "Representative set" → "PDB format" → "All equivalence classes"

# Or use wget:
wget -O rnasolo_representatives.zip "https://rnasolo.cs.put.poznan.pl/download/..."
unzip rnasolo_representatives.zip -d ./data/rnasolo/
```

### Step 2: Verify Data

```bash
# Check structure count
ls ./data/rnasolo/*.pdb | wc -l

# Verify C4' atoms exist
grep " C4'" ./data/rnasolo/*.pdb | head -20

# Check length distribution
for f in ./data/rnasolo/*.pdb; do 
    grep "^ATOM" "$f" | grep " C4'" | wc -l
done | sort -n | uniq -c | head -20
```

### Step 3: Train

**Single GPU:**
```bash
python train_design_simple.py \
    --data_dir ./data/rnasolo \
    --checkpoint_path ./checkpoints/rnapro.pt \
    --output_dir ./outputs/design \
    --batch_size 4 \
    --max_steps 50000 \
    --cfg_drop_prob 0.1
```

**Multi-GPU:**
```bash
torchrun --nproc_per_node=4 train_design_simple.py \
    --data_dir ./data/rnasolo \
    --checkpoint_path ./checkpoints/rnapro.pt \
    --output_dir ./outputs/design \
    --batch_size 4 \
    --max_steps 50000 \
    --cfg_drop_prob 0.1
```

### Step 4: Sample Structures

```python
import torch
from rnapro.model.generator import InferenceNoiseScheduler, sample_diffusion

# Load trained model
checkpoint = torch.load('./outputs/design/checkpoint_50000.pt')
model = ...  # Initialize and load model

# Setup sampling
scheduler = InferenceNoiseScheduler(sigma_data=16.0)
noise_schedule = scheduler(N_step=200, device='cuda')

# Sample (see train_design_simple.py for full example)
```

---

## Configuration

### Training Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | required | Directory with PDB files |
| `--checkpoint_path` | None | Pretrained RNAPro checkpoint |
| `--output_dir` | `outputs/design` | Output directory |
| `--batch_size` | 4 | Batch size per GPU |
| `--n_sample` | 4 | Diffusion samples per structure |
| `--lr` | 1e-4 | Learning rate |
| `--weight_decay` | 0.01 | AdamW weight decay |
| `--grad_clip` | 1.0 | Gradient clipping |
| `--max_steps` | 50000 | Training steps |
| `--cfg_drop_prob` | 0.1 | CFG dropout probability |
| `--max_length` | 512 | Maximum sequence length |
| `--min_length` | 10 | Minimum sequence length |
| `--seed` | 42 | Random seed |
| `--num_workers` | 4 | DataLoader workers |
| `--log_every` | 100 | Logging frequency |
| `--save_every` | 5000 | Checkpoint frequency |

### Classifier-Free Guidance (CFG)

CFG enables the model to do both conditional and unconditional generation:

- **Training**: Randomly drop conditioning with probability `cfg_drop_prob`
- **Inference**: Interpolate between conditional and unconditional predictions

```bash
# Training with CFG
--cfg_drop_prob 0.1  # 10% of batches have conditioning dropped
```

---

## How It Works

### Training Pipeline

```
PDB files → Extract C4' coordinates → RNADesignDataset
                                            ↓
                               Position Embeddings (learnable)
                                            ↓
                               PairformerStack (pretrained)
                                            ↓
                    sample_diffusion_training() with EDM noise
                                            ↓
                               DiffusionModule (pretrained)
                                            ↓
                               Weighted MSE Loss (EDM weighting)
```

### Key Components

| Component | Source | Status |
|-----------|--------|--------|
| `PairformerStack` | Pretrained RNAPro | ✅ Loaded & fine-tuned |
| `DiffusionModule` | Pretrained RNAPro | ✅ Loaded & fine-tuned |
| `Position Embedding` | New | Randomly initialized |
| `Linear layers` | New | Randomly initialized |

### EDM Diffusion Training

The training uses Elucidated Diffusion Models (EDM) approach:

1. **Sample noise level**: `σ ~ exp(N(-1.2, 1.5)) × σ_data`
2. **Add noise**: `x_noisy = x_gt + σ × ε`
3. **Denoise**: `x_denoised = DiffusionModule(x_noisy, σ, s_trunk, z_trunk)`
4. **Loss**: Weighted MSE with EDM weighting `w(σ) = (σ² + σ_data²) / (σ × σ_data)²`

---

## Hardware Requirements

| Task | GPUs | VRAM |
|------|------|------|
| Training | 1-4× GPU | ~20-40GB each |
| Sampling | 1× GPU | ~16GB |

---

## Common Issues

**Out of Memory:**
```bash
--batch_size 2 --n_sample 2
```

**Slow data loading:**
```bash
--num_workers 8
```

**No C4' atoms found:**
- Ensure PDB files contain RNA (not protein)
- Check for standard atom naming (`C4'` not `C4*`)

**Distributed training issues:**
```bash
# Set environment variables
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

---

## Example SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=rnapro_design
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j.out

module load cuda/12.1

torchrun --nproc_per_node=4 train_design_simple.py \
    --data_dir ./data/rnasolo \
    --checkpoint_path ./checkpoints/rnapro.pt \
    --output_dir ./outputs/design_${SLURM_JOB_ID} \
    --batch_size 4 \
    --max_steps 50000
```

---

## Files

| File | Description |
|------|-------------|
| `train_design_simple.py` | Main training script |
| `rnapro/model/generator.py` | Diffusion sampling functions |
| `rnapro/model/modules/diffusion.py` | DiffusionModule |
| `rnapro/model/modules/pairformer.py` | PairformerStack |
