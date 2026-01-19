# RNAProducer: De Novo RNA 3D Structure Design

A flow matching-based generative model for RNA 3D structure design, fine-tuned from pretrained RNAPro.

---

## Quick Start with RNAsolo Data

### Step 1: Download RNAsolo Equivalence Classes

```bash
# Create data directory
mkdir -p ./data/rnasolo

# Download RNAsolo equivalence classes (PDB format)
# Visit: https://rnasolo.cs.put.poznan.pl/
# Download: "Representative set" → "PDB format" → "All equivalence classes"

# Or use wget (update URL as needed):
wget -O rnasolo_representatives.zip "https://rnasolo.cs.put.poznan.pl/download/..."
unzip rnasolo_representatives.zip -d ./data/rnasolo/
```

### Step 2: Verify Your Data

```bash
# Check structure count
ls ./data/rnasolo/*.pdb | wc -l

# Verify C4' atoms exist (should see ~1 per residue)
grep " C4'" ./data/rnasolo/*.pdb | head -20

# Check length distribution
for f in ./data/rnasolo/*.pdb; do 
    grep "^ATOM" "$f" | grep " C4'" | wc -l
done | sort -n | uniq -c | head -20
```

### Step 3: Train the Model

**Option A: Direct PDB Loading (Simplest)**

```bash
torchrun --nproc_per_node=4 runner/train_design.py \
    --project rnapro_design \
    --run_name rnasolo_cfg \
    --base_dir ./experiments \
    --train_data_dir ./data/rnasolo \
    --use_pdb_directly True \
    --pdb_file_pattern "*.pdb" \
    --atom_selection c4prime \
    --design_mode cfg \
    --cfg_drop_prob 0.1 \
    --use_ss_constraints True \
    --load_checkpoint_path ./release_data/protenix_models/protenix_base_default_v0.5.0.pt \
    --load_strict False \
    --load_params_only True \
    --max_steps 50000 \
    --batch_size 4 \
    --lr 1e-4
```

**Option B: Preprocess First (Faster Training)**

```bash
# Step 3a: Preprocess to .pt files (extracts C4' coords + sequence)
python preprocess/prepare_design_data.py \
    --input_dir ./data/rnasolo \
    --output_dir ./data/rnasolo_pt \
    --min_length 10 \
    --max_length 512

# Step 3b: Train from preprocessed files
torchrun --nproc_per_node=4 runner/train_design.py \
    --project rnapro_design \
    --run_name rnasolo_cfg \
    --base_dir ./experiments \
    --train_data_dir ./data/rnasolo_pt \
    --use_pdb_directly False \
    --design_mode cfg \
    --cfg_drop_prob 0.1 \
    --use_ss_constraints True \
    --load_checkpoint_path ./release_data/protenix_models/protenix_base_default_v0.5.0.pt \
    --load_strict False \
    --load_params_only True \
    --max_steps 50000 \
    --batch_size 4 \
    --lr 1e-4
```

### Step 4: Sample Structures

```bash
# Unconditional generation
python runner/sample_design.py \
    --checkpoint ./experiments/rnasolo_cfg/checkpoints/checkpoint_latest.pt \
    --output_dir ./outputs \
    --length 50 \
    --n_samples 10 \
    --n_steps 100 \
    --save_pdb

# SS-conditioned with Classifier-Free Guidance
python runner/sample_design.py \
    --checkpoint ./experiments/rnasolo_cfg/checkpoints/checkpoint_latest.pt \
    --output_dir ./outputs \
    --dot_bracket "((((....))))" \
    --cfg_scale 1.5 \
    --n_samples 10 \
    --n_steps 100 \
    --save_pdb

# Sequence + SS conditioned
python runner/sample_design.py \
    --checkpoint ./experiments/rnasolo_cfg/checkpoints/checkpoint_latest.pt \
    --output_dir ./outputs \
    --sequence "GGCAUUAGCC" \
    --dot_bracket "(((....)))" \
    --cfg_scale 2.0 \
    --n_samples 10 \
    --save_pdb
```

---

## SLURM Cluster Workflow

```bash
# 1. (Optional) Preprocess on CPU node
sbatch slurm_preprocess.sh

# 2. Train on multi-GPU node
sbatch slurm_train_design.sh

# 3. Sample structures
sbatch slurm_sample_design.sh
```

Edit SLURM scripts to match your cluster:
```bash
#SBATCH --partition=your_gpu_partition
#SBATCH --account=your_account
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
```

---

## Design Modes

| Mode | Training | Inference | Use Case |
|------|----------|-----------|----------|
| `cfg` | Randomly drops conditioning (10%) | CFG interpolation | **Recommended** - supports both conditional and unconditional |
| `conditional` | Always uses sequence/SS | Requires constraints | When you always want conditioning |
| `unconditional` | Never uses sequence/SS | Length only | Pure unconditional generation |

### Classifier-Free Guidance (CFG)

CFG trains a single model that can do both conditional and unconditional generation:

```bash
# Training: 10% of batches have conditioning dropped
--design_mode cfg --cfg_drop_prob 0.1

# Inference: interpolate between conditional and unconditional
--cfg_scale 1.5  # 1.0 = conditional only, >1.0 = stronger guidance
```

---

## Key Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--design_mode` | cfg | Training mode: cfg, conditional, unconditional |
| `--cfg_drop_prob` | 0.1 | Probability of dropping conditioning (CFG) |
| `--cfg_scale` | 1.5 | Guidance scale at inference |
| `--use_ss_constraints` | True | Detect base pairs from coordinates |
| `--batch_size` | 4 | Per-GPU batch size |
| `--lr` | 1e-4 | Learning rate |
| `--max_steps` | 50000 | Training steps |
| `--n_steps` | 50 | Sampling integration steps |
| `--freeze_pairformer` | False | Freeze pretrained pairformer |
| `--freeze_diffusion` | False | Freeze pretrained diffusion module |

---

## What Gets Loaded from Pretrained RNAPro

| Component | Status | Notes |
|-----------|--------|-------|
| `pairformer_stack` | ✅ Loaded | Core representation learning |
| `diffusion_module` | ✅ Loaded | Structure generation |
| `relative_position_encoding` | ✅ Loaded | Positional info |
| `condition_embedder` | ❌ Random init | New module for design |

---

## Data Pipeline

```
RNAsolo PDB → parse_pdb() → (C4' coords, sequence) → detect_base_pairs()
                                                              ↓
                                                       ss_constraints
                                                              ↓
                              StructureConditionEmbedder(sequence, pair_compat, ss_constraints)
                                                              ↓
                                                    Pairformer (pretrained)
                                                              ↓
                                                    Flow Matching Training
```

---

## Wandb Metrics

Training automatically logs:
- **Loss**: `total_loss`, `flow_loss`, `bond_loss`, `clash_loss`
- **Flow Matching**: Error by timestep (early/mid/late)
- **Optimization**: Learning rate, gradient norm
- **Validation**: RMSD, coordinate error, GDT-like metrics

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

**No C4' atoms found:**
Check that your PDB files contain RNA (not protein) with standard atom names.
