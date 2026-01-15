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
Loss functions for RNA de novo structure design.

These losses are designed for flow matching / diffusion-based generative models
that learn to predict structure without sequence conditioning.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rnapro.utils.logger import get_logger

logger = get_logger(__name__)


class FlowMatchingLoss(nn.Module):
    """
    Loss function for Flow Matching training.
    
    Computes MSE loss between predicted and target coordinates,
    with support for masked atoms and weighted timestep loss.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",  # "mse" or "smooth_l1"
        reduction: str = "mean",
        per_atom_weighting: bool = False,
        timestep_weighting: bool = False,
        sigma_data: float = 16.0,
    ):
        """
        Args:
            loss_type: Type of loss ("mse" or "smooth_l1").
            reduction: Reduction method ("mean", "sum", "none").
            per_atom_weighting: Whether to weight loss per atom type.
            timestep_weighting: Whether to apply timestep-dependent weighting.
            sigma_data: Data standard deviation for normalization.
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.per_atom_weighting = per_atom_weighting
        self.timestep_weighting = timestep_weighting
        self.sigma_data = sigma_data
        
        logger.info(
            f"FlowMatchingLoss initialized: type={loss_type}, "
            f"timestep_weighting={timestep_weighting}"
        )
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        atom_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            x_pred: Predicted coordinates [..., N_sample, N_atom, 3].
            x_target: Target coordinates [..., N_sample, N_atom, 3].
            mask: Atom mask [..., N_atom] or [..., N_sample, N_atom].
            timestep: Timesteps [..., N_sample] for timestep weighting.
            atom_weights: Per-atom weights [..., N_atom].
            
        Returns:
            Loss value (scalar or per-sample depending on reduction).
        """
        # Compute per-coordinate error
        if self.loss_type == "mse":
            error = (x_pred - x_target).pow(2)
        elif self.loss_type == "smooth_l1":
            error = F.smooth_l1_loss(x_pred, x_target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Sum over coordinates (x, y, z)
        error = error.sum(dim=-1)  # [..., N_sample, N_atom]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == error.dim() - 1:
                mask = mask.unsqueeze(-2)  # [..., 1, N_atom]
            error = error * mask
        
        # Apply per-atom weights if provided
        if atom_weights is not None:
            if atom_weights.dim() == error.dim() - 1:
                atom_weights = atom_weights.unsqueeze(-2)
            error = error * atom_weights
        
        # Apply timestep weighting if enabled
        if self.timestep_weighting and timestep is not None:
            # Weight by 1 / (1 - t + eps) to emphasize later timesteps
            t_weight = 1.0 / (1.0 - timestep + 0.1)
            t_weight = t_weight[..., None]  # [..., N_sample, 1]
            error = error * t_weight
        
        # Reduce
        if self.reduction == "mean":
            if mask is not None:
                n_valid = mask.sum() * error.shape[-2]  # account for N_sample
                return error.sum() / (n_valid + 1e-8)
            else:
                return error.mean()
        elif self.reduction == "sum":
            return error.sum()
        else:
            return error


class VelocityLoss(nn.Module):
    """
    Direct velocity prediction loss for flow matching.
    
    Instead of predicting denoised coordinates and computing
    implicit velocity, this loss directly supervises velocity prediction.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
    ):
        """
        Args:
            reduction: Reduction method.
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute velocity prediction loss.
        
        Args:
            v_pred: Predicted velocity [..., N_sample, N_atom, 3].
            v_target: Target velocity [..., N_sample, N_atom, 3].
            mask: Atom mask.
            
        Returns:
            Loss value.
        """
        error = (v_pred - v_target).pow(2).sum(dim=-1)
        
        if mask is not None:
            if mask.dim() == error.dim() - 1:
                mask = mask.unsqueeze(-2)
            error = error * mask
            
            if self.reduction == "mean":
                return error.sum() / (mask.sum() * error.shape[-2] + 1e-8)
        
        if self.reduction == "mean":
            return error.mean()
        elif self.reduction == "sum":
            return error.sum()
        else:
            return error


class StructuralConsistencyLoss(nn.Module):
    """
    Auxiliary loss for structural consistency.
    
    Encourages predicted structures to have physically reasonable:
    - Bond lengths
    - Bond angles
    - Non-clashing atoms
    """
    
    def __init__(
        self,
        bond_loss_weight: float = 1.0,
        clash_loss_weight: float = 0.1,
        ideal_bond_length: float = 3.8,  # Approximate backbone distance
        clash_threshold: float = 2.0,  # Minimum distance before clash
    ):
        """
        Args:
            bond_loss_weight: Weight for bond length loss.
            clash_loss_weight: Weight for clash loss.
            ideal_bond_length: Target bond length in Angstroms.
            clash_threshold: Distance threshold for clash detection.
        """
        super().__init__()
        self.bond_loss_weight = bond_loss_weight
        self.clash_loss_weight = clash_loss_weight
        self.ideal_bond_length = ideal_bond_length
        self.clash_threshold = clash_threshold
    
    def forward(
        self,
        x_pred: torch.Tensor,
        token_bonds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute structural consistency losses.
        
        Args:
            x_pred: Predicted coordinates [..., N_atom, 3].
            token_bonds: Adjacency matrix [..., N, N] indicating bonded pairs.
            mask: Atom mask [..., N_atom].
            
        Returns:
            Dictionary of loss components.
        """
        losses = {}
        
        # Compute pairwise distances
        # [..., N_sample, N_atom, 3] -> [..., N_sample, N_atom, N_atom]
        diff = x_pred[..., :, None, :] - x_pred[..., None, :, :]
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        
        # Bond length loss
        if token_bonds is not None and self.bond_loss_weight > 0:
            # Get bonded pairs
            bond_mask = token_bonds > 0.5  # [..., N, N]
            if bond_mask.dim() < distances.dim():
                bond_mask = bond_mask.unsqueeze(-3)  # Add N_sample dim
            
            # Compute deviation from ideal bond length
            bond_distances = distances * bond_mask.float()
            bond_deviation = (bond_distances - self.ideal_bond_length).pow(2) * bond_mask.float()
            
            n_bonds = bond_mask.float().sum() + 1e-8
            bond_loss = bond_deviation.sum() / n_bonds
            
            losses["bond_loss"] = self.bond_loss_weight * bond_loss
        
        # Clash loss (penalize atoms that are too close)
        if self.clash_loss_weight > 0:
            # Create non-bonded mask
            if token_bonds is not None:
                nonbond_mask = (token_bonds < 0.5).float()
                if nonbond_mask.dim() < distances.dim():
                    nonbond_mask = nonbond_mask.unsqueeze(-3)
            else:
                nonbond_mask = 1.0 - torch.eye(
                    distances.shape[-1], 
                    device=distances.device, 
                    dtype=distances.dtype
                )
            
            # Compute clash penalty
            clash_violations = F.relu(self.clash_threshold - distances) * nonbond_mask
            
            if mask is not None:
                # Mask out invalid atoms
                pair_mask = mask[..., :, None] * mask[..., None, :]
                if pair_mask.dim() < clash_violations.dim():
                    pair_mask = pair_mask.unsqueeze(-3)
                clash_violations = clash_violations * pair_mask
                n_pairs = pair_mask.sum() + 1e-8
            else:
                n_pairs = clash_violations.numel() / clash_violations.shape[-1]
            
            clash_loss = clash_violations.pow(2).sum() / n_pairs
            losses["clash_loss"] = self.clash_loss_weight * clash_loss
        
        # Total loss
        losses["structural_loss"] = sum(losses.values())
        
        return losses


class RNAProDesignLoss(nn.Module):
    """
    Combined loss function for RNAProDesign training.
    
    Combines flow matching loss with optional structural regularization.
    """
    
    def __init__(
        self,
        flow_loss_weight: float = 1.0,
        structural_loss_weight: float = 0.1,
        bond_loss_weight: float = 1.0,
        clash_loss_weight: float = 0.1,
        loss_type: str = "mse",
        timestep_weighting: bool = False,
        sigma_data: float = 16.0,
    ):
        """
        Args:
            flow_loss_weight: Weight for main flow matching loss.
            structural_loss_weight: Weight for structural consistency loss.
            bond_loss_weight: Weight for bond length regularization.
            clash_loss_weight: Weight for clash avoidance.
            loss_type: Type of coordinate loss.
            timestep_weighting: Whether to weight by timestep.
            sigma_data: Data standard deviation.
        """
        super().__init__()
        
        self.flow_loss_weight = flow_loss_weight
        self.structural_loss_weight = structural_loss_weight
        
        self.flow_loss = FlowMatchingLoss(
            loss_type=loss_type,
            timestep_weighting=timestep_weighting,
            sigma_data=sigma_data,
        )
        
        self.structural_loss = StructuralConsistencyLoss(
            bond_loss_weight=bond_loss_weight,
            clash_loss_weight=clash_loss_weight,
        )
        
        logger.info(
            f"RNAProDesignLoss initialized: flow_weight={flow_loss_weight}, "
            f"structural_weight={structural_loss_weight}"
        )
    
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        label_dict: Dict[str, torch.Tensor],
        design_conditions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_dict: Predictions from model containing:
                - "x_pred" or "coordinate": Predicted coordinates
                - "v_target": Target velocity (optional)
                - "timestep": Sampled timesteps (optional)
            label_dict: Labels containing:
                - "coordinate": Ground truth coordinates
                - "coordinate_mask": Atom validity mask
            design_conditions: Design conditions containing:
                - "token_bonds": Bond adjacency matrix (optional)
                
        Returns:
            Dictionary of loss values.
        """
        losses = {}
        
        # Get predictions and targets
        x_pred = pred_dict.get("x_pred", pred_dict.get("coordinate"))
        x_target = pred_dict.get("v_target")  # For flow matching, target is augmented GT
        if x_target is None:
            # Fall back to label coordinates
            x_target = label_dict["coordinate"]
            # Expand for N_sample if needed
            if x_target.dim() < x_pred.dim():
                x_target = x_target.unsqueeze(-3).expand_as(x_pred)
        
        mask = label_dict.get("coordinate_mask")
        timestep = pred_dict.get("timestep")
        
        # Flow matching loss
        flow_loss = self.flow_loss(
            x_pred=x_pred,
            x_target=x_target,
            mask=mask,
            timestep=timestep,
        )
        losses["flow_loss"] = self.flow_loss_weight * flow_loss
        
        # Structural consistency loss (on final predictions)
        if self.structural_loss_weight > 0:
            token_bonds = None
            if design_conditions is not None:
                token_bonds = design_conditions.get("token_bonds")
            
            structural_losses = self.structural_loss(
                x_pred=x_pred,
                token_bonds=token_bonds,
                mask=mask,
            )
            
            for key, value in structural_losses.items():
                losses[key] = self.structural_loss_weight * value
        
        # Total loss
        losses["total_loss"] = sum(
            v for k, v in losses.items() 
            if k != "structural_loss"  # Avoid double counting
        )
        
        # Add detailed metrics for wandb logging
        with torch.no_grad():
            # Per-sample loss statistics
            per_sample_error = ((x_pred - x_target) ** 2).sum(dim=-1).mean(dim=-1)  # [..., N_sample]
            losses["flow_loss_std"] = per_sample_error.std()
            
            # Loss at different coordinate dimensions
            coord_error_per_dim = ((x_pred - x_target) ** 2).mean(dim=(-3, -2))  # [3]
            losses["flow_loss_x"] = coord_error_per_dim[..., 0].mean()
            losses["flow_loss_y"] = coord_error_per_dim[..., 1].mean()
            losses["flow_loss_z"] = coord_error_per_dim[..., 2].mean()
        
        return losses


def compute_design_metrics(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for structure design.
    
    Args:
        pred_coords: Predicted coordinates [..., N_atom, 3].
        target_coords: Target coordinates [..., N_atom, 3].
        mask: Atom mask.
        
    Returns:
        Dictionary of metric values.
    """
    metrics = {}
    
    # Flatten batch dimensions if needed
    pred_flat = pred_coords.reshape(-1, pred_coords.shape[-2], 3)
    target_flat = target_coords.reshape(-1, target_coords.shape[-2], 3)
    
    if mask is not None:
        mask_flat = mask.reshape(-1, mask.shape[-1])
    else:
        mask_flat = torch.ones(pred_flat.shape[:-1], device=pred_flat.device)
    
    # RMSD (after alignment)
    rmsd_values = []
    for i in range(pred_flat.shape[0]):
        p = pred_flat[i][mask_flat[i].bool()]
        t = target_flat[i][mask_flat[i].bool()]
        
        if p.shape[0] > 0:
            # Center
            p_centered = p - p.mean(dim=0)
            t_centered = t - t.mean(dim=0)
            
            # Kabsch alignment
            H = p_centered.T @ t_centered
            U, S, Vt = torch.linalg.svd(H)
            R = Vt.T @ U.T
            
            if torch.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            p_aligned = p_centered @ R
            
            rmsd = torch.sqrt(((p_aligned - t_centered) ** 2).sum(dim=-1).mean())
            rmsd_values.append(rmsd.item())
    
    if rmsd_values:
        metrics["rmsd"] = sum(rmsd_values) / len(rmsd_values)
        metrics["rmsd_min"] = min(rmsd_values)
        metrics["rmsd_max"] = max(rmsd_values)
        metrics["rmsd_std"] = (sum((r - metrics["rmsd"])**2 for r in rmsd_values) / len(rmsd_values)) ** 0.5
    
    # Coordinate error (without alignment)
    coord_error = ((pred_flat - target_flat) ** 2).sum(dim=-1)
    if mask is not None:
        coord_error = coord_error * mask_flat
        metrics["coord_error"] = (coord_error.sum() / (mask_flat.sum() + 1e-8)).item()
    else:
        metrics["coord_error"] = coord_error.mean().item()
    
    # Per-atom distance error (GDT-like metric)
    per_atom_dist = torch.sqrt(coord_error + 1e-8)  # [..., N_atom]
    
    # Fraction of atoms within various distance thresholds
    for threshold in [1.0, 2.0, 4.0, 8.0]:
        if mask is not None:
            within = ((per_atom_dist < threshold) * mask_flat).sum()
            total = mask_flat.sum()
        else:
            within = (per_atom_dist < threshold).sum()
            total = per_atom_dist.numel()
        metrics[f"frac_within_{threshold}A"] = (within / (total + 1e-8)).item()
    
    # Radius of gyration comparison
    for i in range(min(pred_flat.shape[0], 10)):  # Limit computation
        p = pred_flat[i][mask_flat[i].bool()] if mask is not None else pred_flat[i]
        t = target_flat[i][mask_flat[i].bool()] if mask is not None else target_flat[i]
        
        if p.shape[0] > 0:
            p_centered = p - p.mean(dim=0)
            t_centered = t - t.mean(dim=0)
            
            rg_pred = torch.sqrt((p_centered ** 2).sum(dim=-1).mean())
            rg_target = torch.sqrt((t_centered ** 2).sum(dim=-1).mean())
            
            if i == 0:
                metrics["radius_gyration_pred"] = rg_pred.item()
                metrics["radius_gyration_target"] = rg_target.item()
                metrics["radius_gyration_error"] = abs(rg_pred.item() - rg_target.item())
    
    return metrics
