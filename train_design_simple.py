#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simplified RNA Structure Design Training

Fine-tunes the existing RNAPro diffusion model for unconditional structure generation.
Uses the native EDM-style diffusion training - no additional Flow Matching layer needed.

Key insight: RNAPro already has a full diffusion generative model:
  - DiffusionModule: AtomAttentionEncoder → DiffusionTransformer → AtomAttentionDecoder
  - Training: Add noise, denoise, compute Smooth LDDT loss
  - Inference: Predictor-corrector sampling from noise

For design, we just need to:
  1. Remove sequence/MSA conditioning (or make it optional)
  2. Train on C4' coordinates from RNAsolo
  3. Use CFG by randomly dropping conditioning

Usage:
    python train_design_simple.py \
        --data_dir /path/to/rnasolo/pdb_files \
        --checkpoint_path /path/to/rnapro_checkpoint.pt \
        --output_dir /path/to/output \
        --batch_size 4 \
        --max_steps 50000
"""

import argparse
import os
import random
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from rnapro.model.RNAPro import RNAPro
from rnapro.model.generator import TrainingNoiseSampler, sample_diffusion_training
from rnapro.model.loss import SmoothLDDTLoss
from rnapro.model.utils import centre_random_augmentation
from rnapro.utils.seed import seed_everything
from rnapro.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Simple PDB Dataset for C4' Coordinates
# =============================================================================

class RNADesignDataset(torch.utils.data.Dataset):
    """
    Dataset for RNA structure design training.
    Loads PDB files and extracts C4' coordinates.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_length: int = 512,
        min_length: int = 10,
    ):
        """
        Args:
            data_dir: Directory containing PDB files
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.min_length = min_length
        
        # Find all PDB files
        self.pdb_files = list(self.data_dir.glob("**/*.pdb"))
        if not self.pdb_files:
            self.pdb_files = list(self.data_dir.glob("**/*.cif"))
        
        logger.info(f"Found {len(self.pdb_files)} structure files in {data_dir}")
        
    def __len__(self) -> int:
        return len(self.pdb_files)
    
    def _extract_c4_coords(self, pdb_path: Path) -> Optional[torch.Tensor]:
        """Extract C4' coordinates from PDB file."""
        coords = []
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        atom_name = line[12:16].strip()
                        if atom_name == "C4'":
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
        except Exception as e:
            logger.warning(f"Error reading {pdb_path}: {e}")
            return None
        
        if len(coords) < self.min_length or len(coords) > self.max_length:
            return None
            
        return torch.tensor(coords, dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        pdb_path = self.pdb_files[idx]
        coords = self._extract_c4_coords(pdb_path)
        
        if coords is None:
            # Return a random valid item instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        n_residues = coords.shape[0]
        
        return {
            "coordinate": coords,  # [N, 3]
            "coordinate_mask": torch.ones(n_residues, dtype=torch.float32),  # [N]
            "n_residues": n_residues,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate batch with padding."""
    # Filter None values
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    max_len = max(b["n_residues"] for b in batch)
    batch_size = len(batch)
    
    coords = torch.zeros(batch_size, max_len, 3)
    masks = torch.zeros(batch_size, max_len)
    
    for i, b in enumerate(batch):
        n = b["n_residues"]
        coords[i, :n] = b["coordinate"]
        masks[i, :n] = b["coordinate_mask"]
    
    return {
        "coordinate": coords,
        "coordinate_mask": masks,
    }


# =============================================================================
# Design Model Wrapper
# =============================================================================

class RNAProDesignSimple(nn.Module):
    """
    Simplified design model that wraps RNAPro for unconditional generation.
    
    Key changes from structure prediction:
    - s_inputs: learned embeddings instead of sequence features
    - Optional CFG: randomly drop conditioning during training
    """
    
    def __init__(
        self,
        configs,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.configs = configs
        self.cfg_drop_prob = cfg_drop_prob
        
        # Core dimensions
        self.c_s = configs.c_s  # 384
        self.c_z = configs.c_z  # 128
        self.c_s_inputs = configs.c_s_inputs  # 449
        
        # Use existing RNAPro components
        self.pairformer_stack = None  # Will load from checkpoint
        self.diffusion_module = None  # Will load from checkpoint
        
        # Learnable position embeddings (replace sequence embeddings)
        self.max_len = 1024
        self.position_embedding = nn.Embedding(self.max_len, self.c_s_inputs)
        
        # Learnable single/pair initialization
        self.linear_sinit = nn.Linear(self.c_s_inputs, self.c_s, bias=False)
        self.linear_zinit1 = nn.Linear(self.c_s, self.c_z, bias=False)
        self.linear_zinit2 = nn.Linear(self.c_s, self.c_z, bias=False)
        
        # Noise sampler
        self.noise_sampler = TrainingNoiseSampler(
            p_mean=-1.2,
            p_std=1.5,
            sigma_data=configs.get("sigma_data", 16.0),
        )
        
    def load_pretrained(self, checkpoint_path: str):
        """Load pretrained RNAPro weights."""
        logger.info(f"Loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract model state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Load Pairformer
        pairformer_state = {
            k.replace("pairformer_stack.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("pairformer_stack.")
        }
        if pairformer_state:
            self.pairformer_stack.load_state_dict(pairformer_state)
            logger.info(f"Loaded PairformerStack ({len(pairformer_state)} params)")
        
        # Load DiffusionModule
        diffusion_state = {
            k.replace("diffusion_module.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("diffusion_module.")
        }
        if diffusion_state:
            self.diffusion_module.load_state_dict(diffusion_state)
            logger.info(f"Loaded DiffusionModule ({len(diffusion_state)} params)")
    
    def create_input_features(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict, dict]:
        """
        Create input features for diffusion module.
        
        Args:
            batch: Dict with "coordinate" [B, N, 3] and "coordinate_mask" [B, N]
            
        Returns:
            input_feature_dict, label_dict
        """
        coords = batch["coordinate"]  # [B, N, 3]
        mask = batch["coordinate_mask"]  # [B, N]
        B, N, _ = coords.shape
        device = coords.device
        dtype = coords.dtype
        
        # Position indices
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        
        # s_inputs: position embeddings [B, N, c_s_inputs]
        s_inputs = self.position_embedding(positions)
        
        # Initialize s and z
        s_init = self.linear_sinit(s_inputs)  # [B, N, c_s]
        z_init = (
            self.linear_zinit1(s_init).unsqueeze(-2) +
            self.linear_zinit2(s_init).unsqueeze(-3)
        )  # [B, N, N, c_z]
        
        # Token-level features for DiffusionModule
        input_feature_dict = {
            # Token indices
            "residue_index": positions,
            "token_index": positions,
            "asym_id": torch.zeros(B, N, device=device, dtype=torch.long),
            "entity_id": torch.zeros(B, N, device=device, dtype=torch.long),
            "sym_id": torch.zeros(B, N, device=device, dtype=torch.long),
            # Atom features (for C4' rep: n_atoms = n_tokens)
            "atom_to_token_idx": positions,
            "ref_pos": torch.zeros(B, N, 3, device=device, dtype=dtype),
            "ref_charge": torch.zeros(B, N, device=device, dtype=dtype),
            "ref_mask": mask,
            "ref_space_uid": torch.zeros(B, N, device=device, dtype=torch.long),
            "ref_element": torch.ones(B, N, device=device, dtype=torch.long),
            "ref_atom_name_chars": torch.zeros(B, N, 4, device=device, dtype=torch.long),
        }
        
        label_dict = {
            "coordinate": coords,
            "coordinate_mask": mask,
        }
        
        return input_feature_dict, label_dict, s_inputs, s_init, z_init
    
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        n_sample: int = 4,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Returns:
            Dict with "x_denoised", "x_gt", "noise_level", "loss"
        """
        # Create features
        input_feature_dict, label_dict, s_inputs, s, z = self.create_input_features(batch)
        
        # Run Pairformer
        s, z = self.pairformer_stack(
            s, z,
            pair_mask=None,
            triangle_multiplicative=True,
            triangle_attention=True,
        )
        
        # CFG: randomly drop conditioning
        use_conditioning = not (self.training and random.random() < self.cfg_drop_prob)
        
        # Diffusion training step
        x_gt_aug, x_denoised, noise_level = sample_diffusion_training(
            noise_sampler=self.noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=n_sample,
            use_conditioning=use_conditioning,
        )
        
        return {
            "x_denoised": x_denoised,
            "x_gt": x_gt_aug,
            "noise_level": noise_level,
            "mask": label_dict["coordinate_mask"],
        }


# =============================================================================
# Loss Function
# =============================================================================

def compute_diffusion_loss(
    x_denoised: torch.Tensor,
    x_gt: torch.Tensor,
    noise_level: torch.Tensor,
    mask: torch.Tensor,
    sigma_data: float = 16.0,
) -> torch.Tensor:
    """
    Compute weighted diffusion loss (Smooth LDDT + MSE).
    
    Uses EDM weighting: w(σ) = (σ² + σ_data²) / (σ * σ_data)²
    
    Args:
        x_denoised: Denoised coordinates [B, N_sample, N_atom, 3]
        x_gt: Ground truth coordinates [B, N_sample, N_atom, 3]
        noise_level: Noise levels [B, N_sample]
        mask: Atom mask [B, N_atom]
        sigma_data: Data standard deviation
        
    Returns:
        Scalar loss
    """
    # EDM weighting
    weight = (noise_level**2 + sigma_data**2) / (noise_level * sigma_data)**2
    weight = weight.unsqueeze(-1).unsqueeze(-1)  # [B, N_sample, 1, 1]
    
    # Expand mask
    mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N_atom, 1]
    
    # MSE loss
    mse = ((x_denoised - x_gt) ** 2 * mask_expanded).sum(dim=(-1, -2))  # [B, N_sample]
    mse = mse / (mask.sum(dim=-1, keepdim=True) * 3 + 1e-8)  # Normalize
    
    # Weighted loss
    loss = (weight.squeeze(-1).squeeze(-1) * mse).mean()
    
    return loss


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Main training function."""
    
    # Setup distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Seed
    seed_everything(args.seed + local_rank, deterministic=False)
    
    # Dataset
    dataset = RNADesignDataset(
        data_dir=args.data_dir,
        max_length=args.max_length,
        min_length=args.min_length,
    )
    
    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    # Create minimal config for model
    from ml_collections import ConfigDict
    configs = ConfigDict({
        "c_s": 384,
        "c_z": 128,
        "c_s_inputs": 449,
        "sigma_data": 16.0,
        "model": {
            "pairformer": {
                "n_blocks": 48,
                "n_heads": 16,
                "c_s": 384,
                "c_z": 128,
            },
            "diffusion_module": {
                "sigma_data": 16.0,
                "c_atom": 128,
                "c_atompair": 16,
                "c_token": 768,
                "c_s": 384,
                "c_z": 128,
                "c_s_inputs": 449,
                "atom_encoder": {"n_blocks": 3, "n_heads": 4},
                "transformer": {"n_blocks": 24, "n_heads": 16},
                "atom_decoder": {"n_blocks": 3, "n_heads": 4},
            },
        },
    })
    
    # Model
    model = RNAProDesignSimple(configs, cfg_drop_prob=args.cfg_drop_prob)
    
    # Initialize modules from configs
    from rnapro.model.modules.pairformer import PairformerStack
    from rnapro.model.modules.diffusion import DiffusionModule
    
    model.pairformer_stack = PairformerStack(**configs.model.pairformer)
    model.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
    
    # Load pretrained weights
    if args.checkpoint_path:
        model.load_pretrained(args.checkpoint_path)
    
    model = model.to(device)
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    global_step = 0
    model.train()
    
    logger.info(f"Starting training for {args.max_steps} steps")
    
    while global_step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(global_step // len(dataloader))
        
        for batch in dataloader:
            if batch is None:
                continue
            
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            outputs = model(batch, n_sample=args.n_sample)
            
            # Loss
            loss = compute_diffusion_loss(
                x_denoised=outputs["x_denoised"],
                x_gt=outputs["x_gt"],
                noise_level=outputs["noise_level"],
                mask=outputs["mask"],
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Logging
            if global_step % args.log_every == 0 and local_rank == 0:
                logger.info(f"Step {global_step}: loss = {loss.item():.4f}")
            
            # Save checkpoint
            if global_step % args.save_every == 0 and local_rank == 0:
                save_path = Path(args.output_dir) / f"checkpoint_{global_step}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "step": global_step,
                    "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")
            
            global_step += 1
            if global_step >= args.max_steps:
                break
    
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="RNA Structure Design Training (Simplified)")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with PDB files")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_length", type=int, default=10)
    
    # Model
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Pretrained RNAPro checkpoint")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1, help="CFG dropout probability")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=4, help="Diffusion samples per structure")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging/Saving
    parser.add_argument("--output_dir", type=str, default="outputs/design_simple")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=5000)
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
