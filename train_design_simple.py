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
    Loads PDB files or preprocessed .pt files and extracts C4' coordinates.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_length: int = 512,
        min_length: int = 10,
    ):
        """
        Args:
            data_dir: Directory containing PDB or .pt files
            max_length: Maximum sequence length
            min_length: Minimum sequence length
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.min_length = min_length
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # First try to find .pt files (preprocessed)
        self.pt_files = list(self.data_dir.glob("**/*.pt"))
        if not self.pt_files:
            self.pt_files = list(self.data_dir.glob("*.pt"))
        
        if self.pt_files:
            # Use preprocessed .pt files
            self.use_pt = True
            self.data_files = self.pt_files
            logger.info(f"Found {len(self.pt_files)} preprocessed .pt files in {data_dir}")
            # Pre-filter valid .pt files
            self._prefilter_pt_files()
        else:
            # Fall back to PDB files
            self.use_pt = False
            self._find_pdb_files()
        
    def _find_pdb_files(self):
        """Find and validate PDB files."""
        # Find all PDB files (try multiple patterns)
        self.data_files = list(self.data_dir.glob("**/*.pdb"))
        if not self.data_files:
            self.data_files = list(self.data_dir.glob("**/*.PDB"))
        if not self.data_files:
            self.data_files = list(self.data_dir.glob("**/*.ent"))
        if not self.data_files:
            self.data_files = list(self.data_dir.glob("**/*.cif"))
        if not self.data_files:
            self.data_files = list(self.data_dir.glob("*.pdb"))
        
        logger.info(f"Found {len(self.data_files)} PDB files in {self.data_dir}")
        
        if len(self.data_files) == 0:
            contents = list(self.data_dir.iterdir())[:20]
            logger.error(f"No files found! Directory contents (first 20): {contents}")
            raise ValueError(f"No PDB/PT files found in {self.data_dir}")
        
        # Pre-filter valid files
        self._prefilter_pdb_files()
        
    def _prefilter_pt_files(self):
        """Pre-filter .pt files to ensure valid samples."""
        valid_files = []
        invalid_count = 0
        
        for pt_path in self.data_files:
            try:
                data = torch.load(pt_path, map_location='cpu')
                coords = self._get_coords_from_pt(data)
                if coords is not None:
                    n = coords.shape[0]
                    if self.min_length <= n <= self.max_length:
                        valid_files.append(pt_path)
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
            except Exception as e:
                logger.warning(f"Error loading {pt_path}: {e}")
                invalid_count += 1
        
        logger.info(f"Pre-filtering .pt files: {len(valid_files)} valid, {invalid_count} invalid")
        
        if len(valid_files) == 0:
            if self.data_files:
                sample = torch.load(self.data_files[0], map_location='cpu')
                logger.error(f"Sample .pt file keys: {sample.keys() if isinstance(sample, dict) else type(sample)}")
            raise ValueError(f"No valid .pt files found in {self.data_dir}")
        
        self.data_files = valid_files
    
    def _prefilter_pdb_files(self):
        """Pre-filter PDB files to ensure we have valid samples."""
        valid_files = []
        invalid_count = 0
        
        for pdb_path in self.data_files:
            coords = self._extract_c4_coords(pdb_path, log_errors=False)
            if coords is not None:
                valid_files.append(pdb_path)
            else:
                invalid_count += 1
        
        logger.info(f"Pre-filtering PDB files: {len(valid_files)} valid, {invalid_count} invalid")
        
        if len(valid_files) == 0:
            if self.data_files:
                sample_file = self.data_files[0]
                logger.error(f"Sample file: {sample_file}")
                try:
                    with open(sample_file, 'r') as f:
                        lines = f.readlines()[:20]
                        logger.error(f"First 20 lines: {''.join(lines)}")
                except Exception as e:
                    logger.error(f"Could not read sample file: {e}")
            raise ValueError(f"No valid RNA structures with C4' atoms found in {self.data_dir}")
        
        self.data_files = valid_files
        
    def __len__(self) -> int:
        return len(self.data_files)
    
    def _get_coords_from_pt(self, data: dict) -> Optional[torch.Tensor]:
        """Extract coordinates from .pt file data."""
        # Try common key names
        for key in ['coordinate', 'coordinates', 'coords', 'c4_coords', 'positions']:
            if key in data:
                coords = data[key]
                if isinstance(coords, torch.Tensor):
                    coords = coords.float()
                else:
                    coords = torch.tensor(coords, dtype=torch.float32)
                
                # Ensure shape is [N, 3]
                if coords.ndim == 1:
                    # Single point [3] -> [1, 3]
                    if coords.shape[0] == 3:
                        coords = coords.unsqueeze(0)
                    else:
                        continue
                elif coords.ndim == 2:
                    # [N, 3] - correct shape
                    if coords.shape[-1] != 3:
                        continue
                elif coords.ndim == 3:
                    # [1, N, 3] or [N, A, 3] - take first slice or reshape
                    if coords.shape[0] == 1:
                        coords = coords.squeeze(0)  # [N, 3]
                    elif coords.shape[-1] == 3:
                        # Flatten to [N*A, 3] if all-atom, or just take first
                        coords = coords.reshape(-1, 3)
                    else:
                        continue
                else:
                    continue
                
                return coords
        return None
    
    def _extract_c4_coords(self, pdb_path: Path, log_errors: bool = True) -> Optional[torch.Tensor]:
        """Extract C4' coordinates from PDB file."""
        coords = []
        
        try:
            with open(pdb_path, 'r') as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        atom_name = line[12:16].strip()
                        if atom_name in ("C4'", "C4*"):
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
        except Exception as e:
            if log_errors:
                logger.warning(f"Error reading {pdb_path}: {e}")
            return None
        
        if len(coords) < self.min_length or len(coords) > self.max_length:
            return None
            
        return torch.tensor(coords, dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        file_path = self.data_files[idx]
        
        if self.use_pt:
            # Load from .pt file
            try:
                data = torch.load(file_path, map_location='cpu')
                coords = self._get_coords_from_pt(data)
                if coords is None:
                    return self.__getitem__(random.randint(0, len(self) - 1))
            except Exception:
                return self.__getitem__(random.randint(0, len(self) - 1))
        else:
            # Load from PDB file
            coords = self._extract_c4_coords(file_path)
            if coords is None:
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
        """Load pretrained RNAPro/Protenix weights."""
        logger.info(f"Loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Extract model state dict - handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            # Assume it's the state dict directly
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DDP)
        if all(k.startswith("module.") for k in list(state_dict.keys())[:10]):
            logger.info("Removing 'module.' prefix from checkpoint keys (DDP checkpoint)")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        # Debug: show available top-level keys
        top_keys = set(k.split('.')[0] for k in state_dict.keys())
        logger.info(f"Checkpoint top-level keys: {top_keys}")
        
        # Load PairformerStack weights (prefix: pairformer_stack.)
        pairformer_state = {
            k.replace("pairformer_stack.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("pairformer_stack.")
        }
        
        if pairformer_state:
            logger.info(f"Found {len(pairformer_state)} pairformer_stack weights")
            try:
                missing, unexpected = self.pairformer_stack.load_state_dict(pairformer_state, strict=False)
                logger.info(f"Loaded PairformerStack: {len(missing)} missing, {len(unexpected)} unexpected")
                if missing:
                    logger.warning(f"Missing keys (first 5): {missing[:5]}")
                if unexpected:
                    logger.warning(f"Unexpected keys (first 5): {unexpected[:5]}")
            except Exception as e:
                logger.error(f"Error loading PairformerStack: {e}")
        else:
            logger.warning("No PairformerStack weights found in checkpoint")
        
        # Load DiffusionModule weights (prefix: diffusion_module.)
        diffusion_state = {
            k.replace("diffusion_module.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("diffusion_module.")
        }
        
        if diffusion_state:
            logger.info(f"Found {len(diffusion_state)} diffusion_module weights")
            try:
                missing, unexpected = self.diffusion_module.load_state_dict(diffusion_state, strict=False)
                logger.info(f"Loaded DiffusionModule: {len(missing)} missing, {len(unexpected)} unexpected")
                if missing:
                    logger.warning(f"Missing keys (first 5): {missing[:5]}")
                if unexpected:
                    logger.warning(f"Unexpected keys (first 5): {unexpected[:5]}")
            except Exception as e:
                logger.error(f"Error loading DiffusionModule: {e}")
        else:
            logger.warning("No DiffusionModule weights found in checkpoint")
        
        # Load linear init layers (at top level in checkpoint)
        # Checkpoint has: linear_no_bias_sinit, linear_no_bias_zinit1, linear_no_bias_zinit2
        # Our model has: linear_sinit, linear_zinit1, linear_zinit2
        init_mapping = {
            "linear_no_bias_sinit.weight": "linear_sinit",
            "linear_no_bias_zinit1.weight": "linear_zinit1",
            "linear_no_bias_zinit2.weight": "linear_zinit2",
        }
        
        for ckpt_key, model_attr in init_mapping.items():
            if ckpt_key in state_dict:
                if hasattr(self, model_attr):
                    getattr(self, model_attr).weight.data = state_dict[ckpt_key]
                    logger.info(f"Loaded {model_attr} from {ckpt_key}")
            else:
                logger.warning(f"Init layer {ckpt_key} not found in checkpoint")
    
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
        # Note: AtomAttentionEncoder expects:
        #   - ref_element: [B, N, 128] one-hot (or will be embedded)
        #   - ref_atom_name_chars: [B, N, 4, 64] one-hot (4 chars x 64 classes)
        #   - ref_mask: [B, N, 1] or [B, N]
        
        # For C4' atoms: element is Carbon (atomic number 6)
        # ref_element should be one-hot with 128 classes
        ref_element = torch.zeros(B, N, 128, device=device, dtype=dtype)
        ref_element[..., 6] = 1.0  # Carbon = element 6
        
        # ref_atom_name_chars: "C4'" = ['C', '4', "'", ' '] encoded as one-hot
        # Character encoding: assume ASCII-based, ' '=32, '\'=39, '4'=52, 'C'=67
        # But simpler: use indices 0-63 for common chars
        ref_atom_name_chars = torch.zeros(B, N, 4, 64, device=device, dtype=dtype)
        # Just set a default encoding (the model will learn from this)
        ref_atom_name_chars[..., 0, 3] = 1.0  # 'C'
        ref_atom_name_chars[..., 1, 4] = 1.0  # '4'
        ref_atom_name_chars[..., 2, 7] = 1.0  # "'"
        ref_atom_name_chars[..., 3, 0] = 1.0  # padding/space
        
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
            "ref_element": ref_element,  # [B, N, 128] one-hot
            "ref_atom_name_chars": ref_atom_name_chars.reshape(B, N, 4 * 64),  # [B, N, 256]
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
        
        # Debug: validate shapes
        coords = label_dict["coordinate"]
        mask = label_dict["coordinate_mask"]
        logger.info(f"DEBUG: coords shape={coords.shape}, mask shape={mask.shape}, coords dtype={coords.dtype}, mask dtype={mask.dtype}")
        assert coords.ndim == 3 and coords.shape[-1] == 3, f"Expected coords [B, N, 3], got {coords.shape}"
        assert mask.ndim == 2, f"Expected mask [B, N], got {mask.shape}"
        assert coords.shape[:-1] == mask.shape, f"Shape mismatch: coords {coords.shape} vs mask {mask.shape}"
        
        # Run Pairformer
        s, z = self.pairformer_stack(
            s, z,
            pair_mask=None,
            triangle_multiplicative="torch",
            triangle_attention="torch",
        )
        
        # CFG: randomly drop conditioning
        use_conditioning = not (self.training and random.random() < self.cfg_drop_prob)
        
        # Diffusion training step
        # Note: sample_diffusion_training expands coords to [B, N_sample, N, 3]
        # The DiffusionModule.f_forward passes token-level features (asym_id, residue_index, etc.)
        # to DiffusionConditioning which expects [B, N] shape.
        # But atom-level features (ref_pos, ref_mask, atom_to_token_idx, etc.) go to
        # AtomAttentionEncoder which expects [B, N_sample, N_atom, ...] shape.
        # 
        # The model handles this internally by expanding s_trunk/z_pair after DiffusionConditioning.
        # For atom features, it uses the batch_shape from ref_pos, so we need those to match.
        # 
        # Solution: Keep token-level features at [B, N] but expand atom-level features to [B, N_sample, N].
        
        N_sample = n_sample
        
        # Token-level features stay at [B, N] - used by DiffusionConditioning
        token_level_keys = {"asym_id", "residue_index", "entity_id", "token_index", "sym_id"}
        
        # Atom-level features need [B, N_sample, N_atom, ...] - used by AtomAttentionEncoder
        atom_level_keys = {"ref_pos", "ref_charge", "ref_mask", "ref_space_uid", 
                          "ref_element", "ref_atom_name_chars", "atom_to_token_idx"}
        
        expanded_input_feature_dict = {}
        for k, v in input_feature_dict.items():
            if isinstance(v, torch.Tensor):
                if k in atom_level_keys:
                    # Expand from [B, N, ...] to [B, N_sample, N, ...]
                    expanded_input_feature_dict[k] = v.unsqueeze(1).expand(
                        v.shape[0], N_sample, *v.shape[1:]
                    ).contiguous()
                else:
                    # Keep token-level features unchanged
                    expanded_input_feature_dict[k] = v
            else:
                expanded_input_feature_dict[k] = v
        
        x_gt_aug, x_denoised, noise_level = sample_diffusion_training(
            noise_sampler=self.noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=expanded_input_feature_dict,
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
    })
    
    # Model
    model = RNAProDesignSimple(configs, cfg_drop_prob=args.cfg_drop_prob)
    
    # Initialize modules from configs
    from rnapro.model.modules.pairformer import PairformerStack
    from rnapro.model.modules.diffusion import DiffusionModule
    
    # PairformerStack: 48 blocks, 16 heads
    model.pairformer_stack = PairformerStack(
        n_blocks=48,
        n_heads=16,
        c_s=384,
        c_z=128,
        dropout=0.25,
    )
    
    # DiffusionModule: matches Protenix checkpoint structure
    model.diffusion_module = DiffusionModule(
        sigma_data=16.0,
        c_atom=128,
        c_atompair=16,
        c_token=768,
        c_s=384,
        c_z=128,
        c_s_inputs=449,
        atom_encoder={"n_blocks": 3, "n_heads": 4},
        transformer={"n_blocks": 24, "n_heads": 16},
        atom_decoder={"n_blocks": 3, "n_heads": 4},
    )
    
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


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint structure for debugging."""
    print(f"\n=== Inspecting checkpoint: {checkpoint_path} ===\n")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict):
        print(f"Top-level keys: {list(checkpoint.keys())}")
        
        # Find the state dict
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("\nUsing 'model' key")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("\nUsing 'state_dict' key")
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("\nUsing 'model_state_dict' key")
        else:
            state_dict = checkpoint
            print("\nUsing checkpoint directly as state_dict")
        
        # Remove module. prefix for analysis
        if all(k.startswith("module.") for k in list(state_dict.keys())[:10]):
            print("\nStripping 'module.' prefix for analysis...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        # Count keys by first-level prefix
        prefix_counts = {}
        for k in state_dict.keys():
            prefix = k.split('.')[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        print(f"\nParameter counts by first-level prefix:")
        for prefix, count in sorted(prefix_counts.items(), key=lambda x: -x[1]):
            print(f"  {prefix}: {count}")
        
        # Count keys by second-level prefix
        print(f"\nParameter counts by two-level prefix:")
        prefix2_counts = {}
        for k in state_dict.keys():
            parts = k.split('.')
            if len(parts) >= 2:
                prefix = '.'.join(parts[:2])
            else:
                prefix = parts[0]
            prefix2_counts[prefix] = prefix2_counts.get(prefix, 0) + 1
        
        for prefix, count in sorted(prefix2_counts.items(), key=lambda x: -x[1])[:20]:
            print(f"  {prefix}: {count}")
        
        # Search for specific patterns
        print(f"\n--- Keys containing 'pairformer' ---")
        pairformer_keys = [k for k in state_dict.keys() if 'pairformer' in k.lower()]
        print(f"Found {len(pairformer_keys)} keys")
        for k in pairformer_keys[:10]:
            print(f"  {k}")
        
        print(f"\n--- Keys containing 'diffusion' ---")
        diffusion_keys = [k for k in state_dict.keys() if 'diffusion' in k.lower()]
        print(f"Found {len(diffusion_keys)} keys")
        for k in diffusion_keys[:10]:
            print(f"  {k}")
        
        print(f"\n--- Keys containing 'trunk' ---")
        trunk_keys = [k for k in state_dict.keys() if 'trunk' in k.lower()]
        print(f"Found {len(trunk_keys)} keys")
        for k in trunk_keys[:10]:
            print(f"  {k}")
        
        print(f"\n--- Keys containing 'structure' ---")
        struct_keys = [k for k in state_dict.keys() if 'structure' in k.lower()]
        print(f"Found {len(struct_keys)} keys")
        for k in struct_keys[:10]:
            print(f"  {k}")
        
        # Show all unique first two levels
        print(f"\n--- All unique two-level prefixes ---")
        for prefix in sorted(prefix2_counts.keys()):
            print(f"  {prefix}")
            
    else:
        print(f"Checkpoint is type: {type(checkpoint)}")


def main():
    parser = argparse.ArgumentParser(description="RNA Structure Design Training (Simplified)")
    
    # Data
    parser.add_argument("--data_dir", type=str, required=False, help="Directory with PDB files")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_length", type=int, default=10)
    
    # Model
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Pretrained RNAPro checkpoint")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1, help="CFG dropout probability")
    parser.add_argument("--inspect_checkpoint", action="store_true", help="Just inspect checkpoint and exit")
    
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
    
    # Inspect checkpoint mode
    if args.inspect_checkpoint:
        if not args.checkpoint_path:
            print("Error: --checkpoint_path required with --inspect_checkpoint")
            return
        inspect_checkpoint(args.checkpoint_path)
        return
    
    # Normal training mode
    if not args.data_dir:
        print("Error: --data_dir required for training")
        return
    
    train(args)


if __name__ == "__main__":
    main()
