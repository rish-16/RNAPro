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
Training script for RNA de novo structure design model.

This trainer uses Flow Matching for unconditional or constraint-conditioned
RNA 3D structure generation.
"""

import datetime
import logging
import os
import time
from argparse import Namespace
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import wandb
from ml_collections.config_dict import ConfigDict
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from configs.configs_design import configs as design_configs
from rnapro.config import parse_configs, parse_sys_args
from rnapro.config.config import save_config
from rnapro.data.design_dataset import (
    RNADesignDataset,
    get_design_dataloaders,
)
from rnapro.model.RNAProDesign import RNAProDesign
from rnapro.model.loss.design_loss import RNAProDesignLoss, compute_design_metrics
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.lr_scheduler import get_lr_scheduler
from rnapro.utils.metrics import SimpleMetricAggregator
from rnapro.utils.seed import seed_everything
from rnapro.utils.torch_utils import to_device
from rnapro.utils.training import get_optimizer
from runner.ema import EMAWrapper

# Disable WANDB's console output capture
os.environ["WANDB_CONSOLE"] = "off"

torch.serialization.add_safe_globals([Namespace])


def create_structure_visualization(
    pred_coords: torch.Tensor,
    target_coords: Optional[torch.Tensor] = None,
    title: str = "Structure",
) -> Optional[wandb.Object3D]:
    """
    Create a 3D visualization for wandb logging.
    
    Args:
        pred_coords: Predicted coordinates [N_atom, 3].
        target_coords: Optional target coordinates [N_atom, 3].
        title: Title for the visualization.
        
    Returns:
        wandb.Object3D or None if visualization fails.
    """
    try:
        import numpy as np
        
        # Convert to numpy
        pred_np = pred_coords.cpu().numpy()
        
        # Create point cloud data
        # Format: [[x, y, z, r, g, b], ...]
        points = []
        
        # Add predicted points (blue)
        for i, coord in enumerate(pred_np):
            points.append([coord[0], coord[1], coord[2], 0, 0, 255])
        
        # Add target points (green) if provided
        if target_coords is not None:
            target_np = target_coords.cpu().numpy()
            for coord in target_np:
                points.append([coord[0], coord[1], coord[2], 0, 255, 0])
        
        return wandb.Object3D(np.array(points))
    except Exception as e:
        logging.warning(f"Failed to create structure visualization: {e}")
        return None


def create_distance_histogram(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Optional[wandb.Histogram]:
    """
    Create histogram of per-atom distance errors for wandb.
    
    Args:
        pred_coords: Predicted coordinates [N_atom, 3].
        target_coords: Target coordinates [N_atom, 3].
        mask: Optional atom mask.
        
    Returns:
        wandb.Histogram or None.
    """
    try:
        distances = torch.sqrt(((pred_coords - target_coords) ** 2).sum(dim=-1))
        if mask is not None:
            distances = distances[mask.bool()]
        return wandb.Histogram(distances.cpu().numpy())
    except Exception as e:
        logging.warning(f"Failed to create distance histogram: {e}")
        return None


class DesignTrainer:
    """
    Trainer for RNA de novo structure design model.
    """
    
    def __init__(self, configs):
        """
        Initialize trainer.
        
        Args:
            configs: Configuration object.
        """
        self.configs = configs
        
        self.init_env()
        self.init_basics()
        self.init_model()
        self.init_log()
        self.init_loss()
        self.init_data()
        self.try_load_checkpoint()
    
    def init_basics(self):
        """Initialize basic training state."""
        self.step = 0
        self.global_step = 0
        self.start_step = 0
        self.iters_to_accumulate = self.configs.iters_to_accumulate
        
        self.run_name = self.configs.run_name
        run_names = DIST_WRAPPER.all_gather_object(
            self.run_name if DIST_WRAPPER.rank == 0 else None
        )
        self.run_name = [name for name in run_names if name is not None][0]
        
        self.run_dir = f"{self.configs.base_dir}/{self.run_name}"
        self.checkpoint_dir = f"{self.run_dir}/checkpoints"
        self.sample_dir = f"{self.run_dir}/samples"
        
        if DIST_WRAPPER.rank == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.sample_dir, exist_ok=True)
            save_config(
                self.configs,
                os.path.join(self.run_dir, "config.yaml"),
            )
        
        self.print(
            f"Run name: {self.run_name}, "
            f"Run dir: {self.run_dir}, "
            f"Checkpoint dir: {self.checkpoint_dir}"
        )
    
    def init_log(self):
        """Initialize logging and wandb."""
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            # Initialize wandb with comprehensive config
            wandb_config = vars(self.configs) if hasattr(self.configs, '__dict__') else dict(self.configs)
            
            # Add computed info
            wandb_config["n_params_total"] = sum(p.numel() for p in self.model.parameters())
            wandb_config["n_params_trainable"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            
            wandb.init(
                project=self.configs.project,
                name=self.run_name,
                config=wandb_config,
                id=self.configs.wandb_id or None,
                tags=["flow_matching", "rna_design", self.configs.get("atom_level", "c4prime")],
            )
            
            # Define metrics to track
            wandb.define_metric("train/loss/*", summary="min")
            wandb.define_metric("train/flow/*", summary="last")
            wandb.define_metric("train/timestep/*", summary="last")
            wandb.define_metric("train/optim/*", summary="last")
            wandb.define_metric("val/loss/*", summary="min")
            wandb.define_metric("val/quality/*", summary="min")
            
            self.print(f"Wandb initialized: {wandb.run.url}")
        
        self.train_metric_wrapper = SimpleMetricAggregator(["avg"])
        self.val_metric_wrapper = SimpleMetricAggregator(["avg"])
    
    def init_env(self):
        """Initialize environment."""
        logging.info(
            f"Distributed: world_size={DIST_WRAPPER.world_size}, "
            f"rank={DIST_WRAPPER.rank}, local_rank={DIST_WRAPPER.local_rank}"
        )
        
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device(f"cuda:{DIST_WRAPPER.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(
                backend="nccl" if self.use_cuda else "gloo",
                timeout=datetime.timedelta(hours=2),
            )
        
        # Set precision
        self.dtype = torch.bfloat16 if self.configs.dtype == "bf16" else torch.float32
        
        # Seed
        seed_everything(
            seed=self.configs.seed + DIST_WRAPPER.rank,
            deterministic=self.configs.deterministic,
        )
    
    def init_model(self):
        """Initialize model and optionally load pretrained weights."""
        self.print("Initializing RNAProDesign model...")
        
        self.model = RNAProDesign(self.configs)
        self.model = self.model.to(self.device)
        
        # Load pretrained RNAPro weights if specified
        # Uses --load_checkpoint_path with --load_params_only True (same as RNAPro)
        load_checkpoint_path = self.configs.get("load_checkpoint_path", "")
        if load_checkpoint_path and self.configs.get("load_params_only", True):
            self.print(f"Loading pretrained RNAPro weights from {load_checkpoint_path}")
            model_without_ddp = self.model
            
            freeze_pairformer = self.configs.get("freeze_pairformer", False)
            freeze_diffusion = self.configs.get("freeze_diffusion", False)
            
            load_info = model_without_ddp.load_pretrained_rnapro(
                checkpoint_path=load_checkpoint_path,
                device=self.device,
                freeze_pairformer=freeze_pairformer,
                freeze_diffusion=freeze_diffusion,
            )
            self.print(f"Pretrained weights loaded: {load_info}")
            
            # Print trainable params info
            params_info = model_without_ddp.get_trainable_params_info()
            self.print("Parameter counts per module:")
            for module_name, counts in params_info.items():
                if isinstance(counts, dict):
                    self.print(f"  {module_name}: {counts['trainable']:,} trainable / {counts['total']:,} total")
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
        
        # DDP wrapper
        if DIST_WRAPPER.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[DIST_WRAPPER.local_rank],
                find_unused_parameters=self.configs.get("find_unused_parameters", False),
            )
        
        # Optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.print(f"Optimizing {len(trainable_params)} parameter groups")
        
        self.optimizer = get_optimizer(
            self.model,
            self.configs,
            finetune_params=[],
        )
        
        # Learning rate scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            self.configs,
        )
        
        # EMA
        self.ema = None
        if self.configs.ema_decay > 0:
            model_without_ddp = self.model.module if hasattr(self.model, "module") else self.model
            self.ema = EMAWrapper(
                model_without_ddp,
                decay=self.configs.ema_decay,
            )
    
    def init_loss(self):
        """Initialize loss function."""
        self.loss_fn = RNAProDesignLoss(
            flow_loss_weight=self.configs.loss["flow_loss_weight"],
            structural_loss_weight=self.configs.loss["structural_loss_weight"],
            bond_loss_weight=self.configs.loss["bond_loss_weight"],
            clash_loss_weight=self.configs.loss["clash_loss_weight"],
            loss_type=self.configs.loss["loss_type"],
            timestep_weighting=self.configs.loss["timestep_weighting"],
            sigma_data=self.configs.sigma_data,
        )
    
    def init_data(self):
        """Initialize data loaders."""
        self.print("Initializing data loaders...")
        
        self.train_loader, self.val_loader = get_design_dataloaders(
            train_data_dir=self.configs.train_data_dir,
            val_data_dir=self.configs.val_data_dir if self.configs.val_data_dir else None,
            batch_size=self.configs.batch_size,
            num_workers=self.configs.num_workers,
            use_pdb_directly=self.configs.get("use_pdb_directly", False),
            pdb_file_pattern=self.configs.get("pdb_file_pattern", "*.pdb"),
            atom_selection=self.configs.get("atom_selection", "c4prime"),
            max_length=self.configs.max_length,
            min_length=self.configs.min_length,
            use_ss_constraints=self.configs.use_ss_constraints,
            use_distance_constraints=self.configs.use_distance_constraints,
            augment_coords=self.configs.augment_coords,
        )
        
        self.print(f"Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            self.print(f"Val samples: {len(self.val_loader.dataset)}")
    
    def try_load_checkpoint(self):
        """Try to load checkpoint if specified."""
        if not self.configs.load_checkpoint_path:
            return
        
        self.print(f"Loading checkpoint from {self.configs.load_checkpoint_path}")
        
        checkpoint = torch.load(
            self.configs.load_checkpoint_path,
            map_location=self.device,
        )
        
        model_without_ddp = self.model.module if hasattr(self.model, "module") else self.model
        
        # Load model weights
        if "model" in checkpoint:
            model_state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]
        else:
            model_state = checkpoint
        
        # Filter out incompatible keys for design model
        current_state = model_without_ddp.state_dict()
        filtered_state = {}
        skipped_keys = []
        
        for key, value in model_state.items():
            # Remove 'module.' prefix if present
            clean_key = key.replace("module.", "")
            
            if clean_key in current_state:
                if current_state[clean_key].shape == value.shape:
                    filtered_state[clean_key] = value
                else:
                    skipped_keys.append(f"{clean_key} (shape mismatch)")
            else:
                skipped_keys.append(f"{clean_key} (not in model)")
        
        if skipped_keys:
            self.print(f"Skipped {len(skipped_keys)} incompatible keys")
            for key in skipped_keys[:10]:
                self.print(f"  - {key}")
            if len(skipped_keys) > 10:
                self.print(f"  ... and {len(skipped_keys) - 10} more")
        
        model_without_ddp.load_state_dict(filtered_state, strict=False)
        self.print(f"Loaded {len(filtered_state)} parameters from checkpoint")
        
        # Load optimizer and scheduler if not params_only
        if not self.configs.load_params_only:
            if "optimizer" in checkpoint and not self.configs.skip_load_optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            
            if "scheduler" in checkpoint and not self.configs.skip_load_scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            
            if "step" in checkpoint and not self.configs.skip_load_step:
                self.step = checkpoint["step"]
                self.start_step = self.step
                self.global_step = self.step * self.iters_to_accumulate
        
        # Load EMA
        if self.ema is not None and "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"])
    
    def save_checkpoint(self, suffix: str = ""):
        """Save checkpoint."""
        if DIST_WRAPPER.rank != 0:
            return
        
        model_without_ddp = self.model.module if hasattr(self.model, "module") else self.model
        
        checkpoint = {
            "step": self.step,
            "model": model_without_ddp.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "configs": dict(self.configs),
        }
        
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()
        
        filename = f"checkpoint_{self.step}{suffix}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        self.print(f"Saved checkpoint to {filepath}")
        
        # Also save latest
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        torch.save(checkpoint, latest_path)
    
    def train_step(
        self,
        design_conditions: Dict[str, Any],
        label_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            design_conditions: Design conditions batch.
            label_dict: Label batch.
            
        Returns:
            Dictionary of loss values and metrics for wandb logging.
        """
        self.model.train()
        
        # Move to device
        design_conditions = to_device(design_conditions, self.device)
        label_dict = to_device(label_dict, self.device)
        
        # Add device info
        design_conditions["device"] = self.device
        design_conditions["dtype"] = self.dtype
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(dtype=self.dtype):
            pred_dict, log_dict = self.model(
                design_conditions=design_conditions,
                label_dict=label_dict,
                mode="train",
            )
            
            # Compute loss
            losses = self.loss_fn(
                pred_dict=pred_dict,
                label_dict=label_dict,
                design_conditions=design_conditions,
            )
        
        # Backward pass
        loss = losses["total_loss"] / self.iters_to_accumulate
        loss.backward()
        
        # Combine losses and log_dict for wandb
        metrics = {k: v.item() for k, v in losses.items()}
        metrics.update(log_dict)  # Add flow matching metrics from model
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        # Use EMA model if available
        if self.ema is not None:
            model_without_ddp = self.model.module if hasattr(self.model, "module") else self.model
            self.ema.copy_to(model_without_ddp)
        
        all_metrics = []
        all_losses = []
        
        for design_conditions, label_dict in tqdm(
            self.val_loader, desc="Validation", disable=DIST_WRAPPER.rank != 0
        ):
            design_conditions = to_device(design_conditions, self.device)
            label_dict = to_device(label_dict, self.device)
            design_conditions["device"] = self.device
            design_conditions["dtype"] = self.dtype
            
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Compute validation loss
                pred_dict, log_dict = self.model(
                    design_conditions=design_conditions,
                    label_dict=label_dict,
                    mode="train",  # Use train mode to get loss
                )
                
                losses = self.loss_fn(
                    pred_dict=pred_dict,
                    label_dict=label_dict,
                    design_conditions=design_conditions,
                )
                all_losses.append({k: v.item() for k, v in losses.items()})
                
                # Generate samples for quality metrics
                pred_dict_infer, _ = self.model(
                    design_conditions=design_conditions,
                    label_dict=None,
                    mode="inference",
                )
            
            # Compute metrics
            pred_coords = pred_dict_infer["coordinate"]
            target_coords = label_dict["coordinate"]
            mask = label_dict.get("coordinate_mask")
            
            # Handle N_sample dimension
            if pred_coords.dim() > target_coords.dim():
                pred_coords = pred_coords[:, 0]  # Take first sample
            
            metrics = compute_design_metrics(
                pred_coords=pred_coords,
                target_coords=target_coords,
                mask=mask,
            )
            all_metrics.append(metrics)
            
            # Store first batch for visualization
            if len(all_metrics) == 1:
                first_pred = pred_coords[0].clone()
                first_target = target_coords[0].clone()
                first_mask = mask[0].clone() if mask is not None else None
        
        # Restore non-EMA model
        if self.ema is not None:
            self.ema.restore(model_without_ddp)
        
        # Aggregate metrics
        avg_metrics = {}
        
        # Aggregate losses
        for key in all_losses[0].keys():
            values = [l[key] for l in all_losses]
            avg_metrics[f"loss/{key}"] = sum(values) / len(values)
        
        # Aggregate quality metrics
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[f"quality/{key}"] = sum(values) / len(values)
        
        # Log visualizations to wandb
        if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
            try:
                # 3D structure visualization
                vis_3d = create_structure_visualization(
                    pred_coords=first_pred,
                    target_coords=first_target,
                    title=f"Step {self.step}",
                )
                if vis_3d is not None:
                    wandb.log({"val/structure_3d": vis_3d}, step=self.step)
                
                # Distance error histogram
                hist = create_distance_histogram(
                    pred_coords=first_pred,
                    target_coords=first_target,
                    mask=first_mask,
                )
                if hist is not None:
                    wandb.log({"val/distance_error_hist": hist}, step=self.step)
                    
            except Exception as e:
                logging.warning(f"Failed to log visualizations: {e}")
        
        return avg_metrics
    
    def train(self):
        """Main training loop."""
        self.print("Starting training...")
        
        if self.configs.eval_first:
            val_metrics = self.validate()
            self.log_metrics(val_metrics, prefix="val")
        
        data_iter = iter(self.train_loader)
        
        pbar = tqdm(
            range(self.start_step, self.configs.max_steps),
            desc="Training",
            disable=DIST_WRAPPER.rank != 0,
        )
        
        for step in pbar:
            self.step = step
            
            # Accumulate gradients
            self.optimizer.zero_grad()
            step_losses = []
            
            for _ in range(self.iters_to_accumulate):
                try:
                    design_conditions, label_dict = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    design_conditions, label_dict = next(data_iter)
                
                losses = self.train_step(design_conditions, label_dict)
                step_losses.append(losses)
                self.global_step += 1
            
            # Gradient clipping and norm computation
            grad_norm = 0.0
            if self.configs.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.configs.grad_clip_norm,
                ).item()
            else:
                # Compute grad norm without clipping for logging
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update EMA
            if self.ema is not None:
                model_without_ddp = self.model.module if hasattr(self.model, "module") else self.model
                self.ema.update(model_without_ddp)
            
            # Aggregate metrics from all accumulation steps
            avg_metrics = {}
            for key in step_losses[0].keys():
                avg_metrics[key] = sum(l[key] for l in step_losses) / len(step_losses)
            
            # Add training dynamics metrics
            avg_metrics["optim/grad_norm"] = grad_norm
            avg_metrics["optim/lr"] = self.scheduler.get_last_lr()[0]
            avg_metrics["optim/step"] = self.step
            
            # Logging
            if step % self.configs.log_interval == 0:
                self.log_metrics(avg_metrics, prefix="train")
                pbar.set_postfix(
                    loss=avg_metrics["total_loss"],
                    lr=f"{avg_metrics['optim/lr']:.2e}",
                    gnorm=f"{grad_norm:.2f}"
                )
            
            # Validation
            if step % self.configs.eval_interval == 0 and step > 0:
                val_metrics = self.validate()
                self.log_metrics(val_metrics, prefix="val")
            
            # Checkpointing
            if (
                self.configs.checkpoint_interval > 0 
                and step % self.configs.checkpoint_interval == 0 
                and step > 0
            ):
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint(suffix="_final")
        self.print("Training complete!")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to console and wandb."""
        if DIST_WRAPPER.rank != 0:
            return
        
        # Console logging (only key metrics)
        key_metrics = ["total_loss", "flow_loss", "optim/lr", "optim/grad_norm"]
        msg = f"Step {self.step}: "
        msg_parts = []
        for k in key_metrics:
            if k in metrics:
                if "lr" in k:
                    msg_parts.append(f"{k}={metrics[k]:.2e}")
                else:
                    msg_parts.append(f"{k}={metrics[k]:.4f}")
        msg += ", ".join(msg_parts)
        self.print(msg)
        
        # Wandb logging (all metrics with proper grouping)
        if self.configs.use_wandb:
            wandb_metrics = {}
            for k, v in metrics.items():
                # Handle nested keys (e.g., "timestep/mean" -> "train/timestep/mean")
                if "/" in k:
                    # Already has category, just add prefix
                    wandb_key = f"{prefix}/{k}" if prefix else k
                else:
                    # Add to loss category by default
                    wandb_key = f"{prefix}/loss/{k}" if prefix else f"loss/{k}"
                wandb_metrics[wandb_key] = v
            
            wandb.log(wandb_metrics, step=self.step)
    
    def print(self, msg: str):
        """Print message (rank 0 only)."""
        if DIST_WRAPPER.rank == 0:
            logging.info(msg)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_sys_args()
    configs = parse_configs(design_configs, args)
    
    # Create trainer and run
    trainer = DesignTrainer(configs)
    
    if configs.eval_only:
        metrics = trainer.validate()
        trainer.print(f"Validation metrics: {metrics}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()
