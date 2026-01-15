# RNAProducer: De Novo RNA 3D Structure Design

A flow matching-based generative model for RNA 3D structure design, fine-tuned from pretrained RNAPro.

---

## Quick Start

### 1. Prepare Data

**From PDB/CIF files:**
```bash
python preprocess/prepare_design_data.py \
    --input_dir ./raw_structures \
    --output_dir ./data/train \
    --min_length 10 \
    --max_length 512
```

**From existing .pt files:** Place them directly in `./data/train/`

Each `.pt` file should contain:
```python
{
    "coordinates": Tensor[N, 3],      # C4' atom positions
    "coordinate_mask": Tensor[N],     # Validity mask (all 1s)
    "n_tokens": int,                  # Number of residues
}
```

---

### 2. Train (Multi-GPU)

**Local (torchrun):**
```bash
torchrun --nproc_per_node=4 runner/train_design.py \
    --project rnapro_design \
    --run_name my_experiment \
    --base_dir ./experiments \
    --train_data_dir ./data/train \
    --load_checkpoint_path ./release_data/protenix_models/protenix_base_default_v0.5.0.pt \
    --load_strict False \
    --load_params_only True \
    --max_steps 50000 \
    --batch_size 4 \
    --lr 1e-4
```

**SLURM cluster:**
```bash
sbatch slurm_train_design.sh
```

---

### 3. Sample (Single GPU)

**Local:**
```bash
python runner/sample_design.py \
    --checkpoint ./experiments/my_experiment/checkpoints/checkpoint_latest.pt \
    --output_dir ./outputs \
    --length 50 \
    --n_samples 10 \
    --n_steps 100 \
    --save_pdb
```

**SLURM cluster:**
```bash
sbatch slurm_sample_design.sh
```

---

## SLURM Workflow

```bash
# 1. Preprocess PDB/CIF → .pt (CPU job)
sbatch slurm_preprocess.sh

# 2. Train on multi-GPU (after step 1)
sbatch slurm_train_design.sh

# 3. Sample structures (after step 2)
sbatch slurm_sample_design.sh
```

**Customize SLURM scripts:**
```bash
#SBATCH --partition=your_partition   # Your GPU partition
#SBATCH --account=your_account       # Your account (if required)
#SBATCH --gpus-per-node=4            # Number of GPUs
#SBATCH --time=48:00:00              # Wall time
```

---

## Key Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 4 | Per-GPU batch size |
| `--lr` | 1e-4 | Learning rate |
| `--max_steps` | 50000 | Training steps |
| `--n_steps` | 50 | Sampling integration steps |
| `--freeze_pairformer` | False | Freeze pretrained pairformer |
| `--freeze_diffusion` | False | Freeze pretrained diffusion module |
| `--use_wandb` | True | Enable Weights & Biases logging |

---

## Training Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Full fine-tune | `--freeze_pairformer False --freeze_diffusion False` | Best quality |
| Partial fine-tune | `--freeze_pairformer True` | Limited GPU memory |
| Feature extraction | `--freeze_pairformer True --freeze_diffusion True` | Quick experiments |

---

## Wandb Metrics

Training automatically logs:
- **Loss**: `total_loss`, `flow_loss`, `bond_loss`, `clash_loss`
- **Flow Matching**: Error by timestep (early/mid/late)
- **Optimization**: Learning rate, gradient norm
- **Validation**: RMSD, coordinate error, GDT-like metrics

---

## Directory Structure

```
./experiments/my_experiment/
├── config.yaml           # Saved configuration
├── checkpoints/
│   ├── checkpoint_latest.pt
│   └── checkpoint_50000_final.pt
└── samples/              # Generated structures
```

---

## Hardware Requirements

| Task | GPUs | VRAM |
|------|------|------|
| Training | 4× A100 | ~40GB each |
| Sampling | 1× GPU | ~16GB |

---

## Common Issues

**OOM during training:**
```bash
--batch_size 2 --diffusion_batch_size 8
```

**DDP errors with frozen params:**
```bash
--find_unused_parameters True
```

**Slow data loading:**
```bash
--num_workers 8
```
