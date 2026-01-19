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
RNAProDesign: De novo RNA 3D structure design model using Flow Matching.

This model removes sequence-dependent components (MSA, templates, sequence embedder)
and replaces them with structure condition embedders for unconditional or
constraint-conditioned 3D structure generation.

Key features:
- Supports fine-tuning from pretrained RNAPro checkpoints
- Operates at C4' level (one atom per residue) for simplicity
- Uses Flow Matching for training instead of diffusion score matching
- Retains pretrained Pairformer and Diffusion module weights

Architecture:
    Constraints → ConditionEmbedder → Pairformer (pretrained) → FlowMatching → Structure
"""

import time
from typing import Any, Optional

import torch
import torch.nn as nn

from rnapro.model.generator.flow_matching import (
    FlowMatchingScheduler,
    FlowMatchingInferenceSampler,
    sample_flow_matching_training,
    flow_matching_loss,
)
from rnapro.model.modules.design_embedders import (
    StructureConditionEmbedder,
    UnconditionalEmbedder,
)
from rnapro.model.modules.diffusion import DiffusionModule
from rnapro.model.modules.embedders import RelativePositionEncoding
from rnapro.model.modules.pairformer import PairformerStack
from rnapro.model.modules.primitives import LinearNoBias
from rnapro.openfold_local.model.primitives import LayerNorm
from rnapro.utils.logger import get_logger
from rnapro.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


class RNAProDesign(nn.Module):
    """
    De novo RNA 3D structure design model using Flow Matching.
    
    This is a simplified version of RNAPro that removes:
    - InputFeatureEmbedder (sequence-dependent)
    - MSAModule (evolutionary information)
    - TemplateEmbedder (structural templates)
    - RibonanzaNet (RNA language model)
    - DistogramHead (sequence-based predictions)
    - ConfidenceHead (sequence-based confidence)
    
    And retains (with pretrained weights):
    - PairformerStack (core representation learning)
    - DiffusionModule (structure generation)
    - RelativePositionEncoding (positional information)
    - Recycling layers (layernorm_z_cycle, linear_no_bias_z_cycle, etc.)
    
    For conditioning, it uses:
    - StructureConditionEmbedder (optional structural constraints)
    - UnconditionalEmbedder (for fully unconditional generation)
    
    Representation:
    - C4' level: one atom per residue (n_atoms == n_tokens)
    - Can be extended to all-atom later
    """
    
    def __init__(self, configs) -> None:
        """
        Initialize RNAProDesign model.
        
        Args:
            configs: Configuration object containing model hyperparameters.
        """
        super(RNAProDesign, self).__init__()
        self.configs = configs
        
        # Core dimensions
        self.c_s = configs.c_s
        self.c_z = configs.c_z
        self.c_s_inputs = configs.c_s_inputs
        
        # Hyperparameters
        self.N_cycle = configs.model.N_cycle
        self.diffusion_batch_size = configs.diffusion_batch_size
        
        # Classifier-Free Guidance (CFG) settings
        # During training: randomly drop conditioning with probability cfg_drop_prob
        # During inference: interpolate between conditional and unconditional predictions
        self.cfg_drop_prob = configs.get("cfg_drop_prob", 0.1)  # 10% unconditional
        self.cfg_scale = configs.get("cfg_scale", 1.0)  # Guidance scale at inference
        
        # Design mode: "unconditional", "conditional", or "cfg" (classifier-free guidance)
        self.design_mode = configs.get("design_mode", "cfg")
        
        # Single condition embedder that handles both modes
        # For unconditional: sequence/ss_constraints are set to None
        # For conditional: sequence/ss_constraints are provided
        # For CFG: randomly drop conditioning during training
        self.condition_embedder = StructureConditionEmbedder(
            c_s=self.c_s,
            c_z=self.c_z,
            c_s_inputs=self.c_s_inputs,
            max_length=configs.get("max_length", 1024),
            n_ss_classes=configs.get("n_ss_classes", 4),
            n_distance_bins=configs.get("n_distance_bins", 64),
        )
        
        # Relative position encoding (KEEP from original)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        
        # Token bond embedding (optional, for backbone connectivity)
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        
        # Initial projections from s_inputs to s_init
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        
        # Pairformer stack (KEEP from original)
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        
        # Diffusion module (KEEP from original) - used for denoising
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        
        # Flow matching schedulers
        self.flow_scheduler = FlowMatchingScheduler(
            sigma_min=configs.get("flow_sigma_min", 0.001),
            sigma_data=configs.sigma_data,
        )
        self.inference_sampler = FlowMatchingInferenceSampler(
            n_steps=configs.get("flow_n_steps", 50),
            sigma_data=configs.sigma_data,
        )
        
        # Recycling layers
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s_cycle = LayerNorm(self.c_s)
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s_cycle = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        
        # Zero-initialize recycling connections
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s_cycle.weight)
        
        logger.info(
            f"RNAProDesign initialized: design_mode={self.design_mode}, "
            f"c_s={self.c_s}, c_z={self.c_z}, N_cycle={self.N_cycle}"
        )
    
    def load_pretrained_rnapro(
        self,
        checkpoint_path: str,
        device: torch.device = None,
        freeze_pairformer: bool = False,
        freeze_diffusion: bool = False,
    ) -> dict[str, int]:
        """
        Load pretrained weights from RNAPro structure prediction model.
        
        This loads weights for:
        - pairformer_stack (core representation learning)
        - diffusion_module (structure generation)
        - relative_position_encoding (positional information)
        - Recycling layers (layernorm_z_cycle, linear_no_bias_z_cycle, etc.)
        
        Args:
            checkpoint_path: Path to RNAPro checkpoint.
            device: Device to load checkpoint to.
            freeze_pairformer: If True, freeze pairformer weights.
            freeze_diffusion: If True, freeze diffusion module weights.
            
        Returns:
            Dictionary with counts of loaded/skipped parameters.
        """
        if device is None:
            device = next(self.parameters()).device
        
        logger.info(f"Loading pretrained RNAPro weights from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict
        if "model" in checkpoint:
            source_state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            source_state = checkpoint["state_dict"]
        else:
            source_state = checkpoint
        
        # Clean keys (remove 'module.' prefix from DDP)
        cleaned_source = {}
        for key, value in source_state.items():
            clean_key = key.replace("module.", "")
            cleaned_source[clean_key] = value
        
        # Define which modules to load from pretrained
        # These are the modules that exist in both RNAPro and RNAProDesign
        loadable_prefixes = [
            "pairformer_stack.",
            "diffusion_module.",
            "relative_position_encoding.",
            "layernorm_z_cycle.",
            "layernorm_s_cycle.",
            "linear_no_bias_z_cycle.",
            "linear_no_bias_s_cycle.",
            "linear_no_bias_token_bond.",
        ]
        
        # Note: In RNAPro, the recycling layer is called linear_no_bias_s
        # but in RNAProDesign we call it linear_no_bias_s_cycle for clarity
        key_remapping = {
            "linear_no_bias_s.": "linear_no_bias_s_cycle.",
            "layernorm_s.": "layernorm_s_cycle.",
        }
        
        current_state = self.state_dict()
        loaded_state = {}
        loaded_count = 0
        skipped_count = 0
        shape_mismatch = []
        not_found = []
        
        for source_key, source_value in cleaned_source.items():
            # Apply key remapping
            target_key = source_key
            for old_prefix, new_prefix in key_remapping.items():
                if source_key.startswith(old_prefix):
                    target_key = source_key.replace(old_prefix, new_prefix, 1)
                    break
            
            # Check if this key should be loaded
            should_load = any(target_key.startswith(prefix) for prefix in loadable_prefixes)
            
            if should_load and target_key in current_state:
                if current_state[target_key].shape == source_value.shape:
                    loaded_state[target_key] = source_value
                    loaded_count += 1
                else:
                    shape_mismatch.append(
                        f"{target_key}: expected {current_state[target_key].shape}, "
                        f"got {source_value.shape}"
                    )
                    skipped_count += 1
            elif should_load:
                not_found.append(target_key)
                skipped_count += 1
        
        # Load the filtered state dict
        missing, unexpected = self.load_state_dict(loaded_state, strict=False)
        
        logger.info(f"Loaded {loaded_count} parameters from pretrained RNAPro")
        logger.info(f"Skipped {skipped_count} parameters")
        
        if shape_mismatch:
            logger.warning(f"Shape mismatches ({len(shape_mismatch)}):")
            for msg in shape_mismatch[:5]:
                logger.warning(f"  {msg}")
            if len(shape_mismatch) > 5:
                logger.warning(f"  ... and {len(shape_mismatch) - 5} more")
        
        # Optionally freeze loaded modules
        if freeze_pairformer:
            logger.info("Freezing pairformer_stack parameters")
            for name, param in self.pairformer_stack.named_parameters():
                param.requires_grad = False
        
        if freeze_diffusion:
            logger.info("Freezing diffusion_module parameters")
            for name, param in self.diffusion_module.named_parameters():
                param.requires_grad = False
        
        return {
            "loaded": loaded_count,
            "skipped": skipped_count,
            "shape_mismatch": len(shape_mismatch),
        }
    
    def get_trainable_params_info(self) -> dict[str, Any]:
        """
        Get information about trainable parameters.
        
        Returns:
            Dictionary with parameter counts per module.
        """
        info = {}
        total_params = 0
        trainable_params = 0
        
        module_names = [
            "condition_embedder",
            "relative_position_encoding",
            "pairformer_stack",
            "diffusion_module",
        ]
        
        for module_name in module_names:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                module_total = sum(p.numel() for p in module.parameters())
                module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                info[module_name] = {
                    "total": module_total,
                    "trainable": module_trainable,
                    "frozen": module_total - module_trainable,
                }
                total_params += module_total
                trainable_params += module_trainable
        
        info["total"] = total_params
        info["trainable"] = trainable_params
        info["frozen"] = total_params - trainable_params
        
        return info
    
    def get_pairformer_output(
        self,
        design_conditions: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass from design conditions to pairformer output.
        
        This replaces the original get_pairformer_output which used
        sequence features. Instead, we use structural constraints
        or unconditional embeddings.
        
        Args:
            design_conditions: Dictionary containing:
                - length: int, sequence length
                - device: torch.device
                - dtype: torch.dtype
                - batch_size: int (optional, default 1)
                - ss_constraints: Optional[Tensor], secondary structure
                - distance_constraints: Optional[Tensor], distance constraints
                - asym_id, residue_index, entity_id, token_index, sym_id: for RelPosEnc
                - token_bonds: Optional[Tensor], backbone connectivity
            N_cycle: Number of recycling iterations.
            inplace_safe: Whether inplace operations are safe.
            chunk_size: Chunk size for memory efficiency.
            
        Returns:
            s_inputs: [B, N, c_s_inputs] single input embeddings
            s: [B, N, c_s] single trunk embeddings
            z: [B, N, N, c_z] pair trunk embeddings
        """
        length = design_conditions["length"]
        device = design_conditions["device"]
        dtype = design_conditions["dtype"]
        batch_size = design_conditions.get("batch_size", 1)
        
        # Get conditioning inputs
        sequence = design_conditions.get("sequence")
        pair_compat = design_conditions.get("pair_compat")
        ss_constraints = design_conditions.get("ss_constraints")
        distance_constraints = design_conditions.get("distance_constraints")
        
        # Apply Classifier-Free Guidance (CFG) conditioning dropout during training
        # Randomly drop conditioning to learn both conditional and unconditional generation
        if self.training and self.design_mode == "cfg" and self.cfg_drop_prob > 0:
            if torch.rand(1).item() < self.cfg_drop_prob:
                # Drop all conditioning -> unconditional
                sequence = None
                pair_compat = None
                ss_constraints = None
                distance_constraints = None
        elif self.design_mode == "unconditional":
            # Always unconditional
            sequence = None
            pair_compat = None
            ss_constraints = None
            distance_constraints = None
        
        # Get initial embeddings from condition embedder
        s_inputs, z_init = self.condition_embedder(
            length=length,
            device=device,
            dtype=dtype,
            sequence=sequence,
            pair_compat=pair_compat,
            ss_constraints=ss_constraints,
            distance_constraints=distance_constraints,
            batch_size=batch_size,
        )
        
        # Project s_inputs to s_init dimension
        s_init = self.linear_no_bias_sinit(s_inputs)  # [B, N, c_s]
        
        # Add relative position encoding
        if all(k in design_conditions for k in ["asym_id", "residue_index", "entity_id", "token_index", "sym_id"]):
            rel_pos_enc = self.relative_position_encoding(
                design_conditions["asym_id"],
                design_conditions["residue_index"],
                design_conditions["entity_id"],
                design_conditions["token_index"],
                design_conditions["sym_id"],
            )
            if inplace_safe:
                z_init += rel_pos_enc
            else:
                z_init = z_init + rel_pos_enc
        
        # Add token bond information if provided
        if "token_bonds" in design_conditions:
            token_bond_emb = self.linear_no_bias_token_bond(
                design_conditions["token_bonds"].unsqueeze(dim=-1)
            )
            if inplace_safe:
                z_init += token_bond_emb
            else:
                z_init = z_init + token_bond_emb
        
        # Initialize recycling tensors
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)
        
        # Recycling through pairformer
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training and cycle_no == (N_cycle - 1)
            ):
                # Update with recycled features
                z_recycle = self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                s_recycle = self.linear_no_bias_s_cycle(self.layernorm_s_cycle(s))
                
                if inplace_safe:
                    z = z_init + z_recycle
                    s = s_init + s_recycle
                else:
                    z = z_init + z_recycle
                    s = s_init + s_recycle
                
                # Pairformer stack
                s, z = self.pairformer_stack(
                    s, z,
                    pair_mask=design_conditions.get("pair_mask"),
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
        
        return s_inputs, s, z
    
    def main_train_loop(
        self,
        design_conditions: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Training loop using flow matching loss.
        
        Args:
            design_conditions: Design conditions dictionary.
            label_dict: Dictionary containing:
                - "coordinate": Ground truth coordinates [B, N_atom, 3]
                - "coordinate_mask": Atom mask [B, N_atom]
            N_cycle: Number of recycling iterations.
            inplace_safe: Whether inplace operations are safe.
            chunk_size: Chunk size for memory efficiency.
            
        Returns:
            pred_dict: Predictions including x_pred, v_target, timesteps.
            log_dict: Logging information.
        """
        # Get pairformer representations
        s_inputs, s, z = self.get_pairformer_output(
            design_conditions=design_conditions,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        
        # Flow matching training step
        N_sample = self.diffusion_batch_size
        diffusion_chunk_size = self.configs.get("diffusion_chunk_size")
        
        x_t, x_pred, v_target, t = sample_flow_matching_training(
            flow_scheduler=self.flow_scheduler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=design_conditions,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            N_sample=N_sample,
            use_conditioning=True,
            diffusion_chunk_size=diffusion_chunk_size,
        )
        
        pred_dict = {
            "x_t": x_t,
            "x_pred": x_pred,
            "v_target": v_target,
            "timestep": t,
            "coordinate": x_pred,  # For compatibility with loss functions
        }
        
        # Compute detailed logging metrics for wandb
        with torch.no_grad():
            # Timestep statistics
            log_dict = {
                "timestep/mean": t.mean().item(),
                "timestep/std": t.std().item(),
                "timestep/min": t.min().item(),
                "timestep/max": t.max().item(),
            }
            
            # Velocity statistics (target velocity = x_1 - x_0)
            v_magnitude = torch.sqrt((v_target ** 2).sum(dim=-1))  # [..., N_sample, N_atom]
            log_dict["velocity/target_magnitude_mean"] = v_magnitude.mean().item()
            log_dict["velocity/target_magnitude_std"] = v_magnitude.std().item()
            
            # Prediction error at different timesteps
            coord_error = ((x_pred - v_target) ** 2).sum(dim=-1)  # [..., N_sample, N_atom]
            log_dict["flow/coord_error_mean"] = coord_error.mean().item()
            
            # Error by timestep bins (early/mid/late)
            t_flat = t.flatten()
            error_flat = coord_error.mean(dim=-1).flatten()  # Average over atoms
            
            early_mask = t_flat < 0.33
            mid_mask = (t_flat >= 0.33) & (t_flat < 0.67)
            late_mask = t_flat >= 0.67
            
            if early_mask.any():
                log_dict["flow/error_t_early"] = error_flat[early_mask].mean().item()
            if mid_mask.any():
                log_dict["flow/error_t_mid"] = error_flat[mid_mask].mean().item()
            if late_mask.any():
                log_dict["flow/error_t_late"] = error_flat[late_mask].mean().item()
            
            # Noisy input statistics
            x_t_magnitude = torch.sqrt((x_t ** 2).sum(dim=-1))
            log_dict["flow/x_t_magnitude_mean"] = x_t_magnitude.mean().item()
            
            # Prediction statistics
            x_pred_magnitude = torch.sqrt((x_pred ** 2).sum(dim=-1))
            log_dict["flow/x_pred_magnitude_mean"] = x_pred_magnitude.mean().item()
            
            # Ground truth statistics
            x_gt = label_dict["coordinate"]
            x_gt_magnitude = torch.sqrt((x_gt ** 2).sum(dim=-1))
            log_dict["data/gt_magnitude_mean"] = x_gt_magnitude.mean().item()
            log_dict["data/n_atoms"] = x_gt.shape[-2]
        
        return pred_dict, log_dict
    
    def _get_pairformer_conditional(
        self,
        design_conditions: dict[str, Any],
        use_conditioning: bool,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get pairformer output with or without conditioning.
        
        Helper for CFG sampling.
        """
        # Make a copy of conditions
        cond = dict(design_conditions)
        
        if not use_conditioning:
            # Remove conditioning for unconditional path
            cond["sequence"] = None
            cond["pair_compat"] = None
            cond["ss_constraints"] = None
            cond["distance_constraints"] = None
        
        return self.get_pairformer_output(
            design_conditions=cond,
            N_cycle=self.N_cycle,
            inplace_safe=True,
            chunk_size=chunk_size,
        )
    
    @torch.no_grad()
    def sample_with_cfg(
        self,
        design_conditions: dict[str, Any],
        n_samples: int = 1,
        n_steps: Optional[int] = None,
        cfg_scale: float = 1.5,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate RNA structures using Classifier-Free Guidance.
        
        At each step, computes both conditional and unconditional predictions,
        then interpolates: v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        Args:
            design_conditions: Design conditions dictionary with sequence/SS.
            n_samples: Number of structures to generate.
            n_steps: Number of integration steps.
            cfg_scale: Guidance scale (1.0 = no guidance, >1.0 = stronger conditioning).
            chunk_size: Chunk size for attention.
            
        Returns:
            coordinates: [n_samples, N_atom, 3] generated coordinates.
        """
        self.eval()
        
        n_atoms = design_conditions["n_atoms"]
        device = design_conditions["device"]
        dtype = design_conditions["dtype"]
        batch_size = design_conditions.get("batch_size", 1)
        
        # Get BOTH conditional and unconditional pairformer outputs
        s_inputs_cond, s_cond, z_cond = self._get_pairformer_conditional(
            design_conditions, use_conditioning=True, chunk_size=chunk_size
        )
        s_inputs_uncond, s_uncond, z_uncond = self._get_pairformer_conditional(
            design_conditions, use_conditioning=False, chunk_size=chunk_size
        )
        
        # Update sampler steps if provided
        if n_steps is not None:
            sampler = FlowMatchingInferenceSampler(
                n_steps=n_steps,
                sigma_data=self.configs.sigma_data,
            )
        else:
            sampler = self.inference_sampler
        
        # Sample using CFG-aware Euler integration
        coordinates = sampler.sample_with_cfg(
            velocity_net=self.diffusion_module,
            shape=(batch_size, n_samples, n_atoms, 3),
            device=device,
            dtype=dtype,
            input_feature_dict=design_conditions,
            s_inputs_cond=s_inputs_cond,
            s_trunk_cond=s_cond,
            z_trunk_cond=z_cond,
            s_inputs_uncond=s_inputs_uncond,
            s_trunk_uncond=s_uncond,
            z_trunk_uncond=z_uncond,
            cfg_scale=cfg_scale,
            chunk_size=chunk_size,
            inplace_safe=True,
        )
        
        return coordinates
    
    @torch.no_grad()
    def sample(
        self,
        design_conditions: dict[str, Any],
        n_samples: int = 1,
        n_steps: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate RNA structures from design conditions.
        
        Args:
            design_conditions: Design conditions dictionary.
            n_samples: Number of structures to generate.
            n_steps: Number of integration steps (overrides config if provided).
            chunk_size: Chunk size for attention.
            
        Returns:
            coordinates: [n_samples, N_atom, 3] generated coordinates.
        """
        self.eval()
        
        # Get pairformer representations
        s_inputs, s, z = self.get_pairformer_output(
            design_conditions=design_conditions,
            N_cycle=self.N_cycle,
            inplace_safe=True,
            chunk_size=chunk_size,
        )
        
        n_atoms = design_conditions["n_atoms"]
        device = design_conditions["device"]
        dtype = design_conditions["dtype"]
        batch_size = design_conditions.get("batch_size", 1)
        
        # Update sampler steps if provided
        if n_steps is not None:
            sampler = FlowMatchingInferenceSampler(
                n_steps=n_steps,
                sigma_data=self.configs.sigma_data,
            )
        else:
            sampler = self.inference_sampler
        
        # Sample using Euler integration
        coordinates = sampler.sample(
            velocity_net=self.diffusion_module,
            shape=(batch_size, n_samples, n_atoms, 3),
            device=device,
            dtype=dtype,
            input_feature_dict=design_conditions,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            chunk_size=chunk_size,
            inplace_safe=True,
        )
        
        return coordinates
    
    def forward(
        self,
        design_conditions: dict[str, Any],
        label_dict: Optional[dict[str, Any]] = None,
        mode: str = "inference",
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """
        Forward pass for training or inference.
        
        Args:
            design_conditions: Design conditions dictionary.
            label_dict: Label dictionary (required for training).
            mode: "train" or "inference".
            
        Returns:
            pred_dict: Predictions dictionary.
            log_dict: Logging dictionary.
        """
        if mode == "train":
            if label_dict is None:
                raise ValueError("label_dict is required for training mode")
            return self.main_train_loop(
                design_conditions=design_conditions,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
            )
        else:  # inference
            n_samples = self.configs.get("n_samples", 1)
            n_steps = self.configs.get("flow_n_steps", 50)
            
            coords = self.sample(
                design_conditions=design_conditions,
                n_samples=n_samples,
                n_steps=n_steps,
            )
            
            pred_dict = {"coordinate": coords}
            log_dict = {}
            
            return pred_dict, log_dict


def create_design_conditions(
    length: int,
    n_atoms: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 1,
    sequence: Optional[torch.Tensor] = None,
    pair_compat: Optional[torch.Tensor] = None,
    ss_constraints: Optional[torch.Tensor] = None,
    distance_constraints: Optional[torch.Tensor] = None,
    atom_to_token_idx: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """
    Helper function to create design conditions dictionary.
    
    This creates a minimal feature dictionary for unconditional
    or constraint-conditioned generation.
    
    Args:
        length: Number of tokens (residues).
        n_atoms: Number of atoms.
        device: Target device.
        dtype: Target dtype.
        batch_size: Batch size.
        sequence: Optional nucleotide indices [N] (0=A, 1=U, 2=G, 3=C, 4=N).
        pair_compat: Optional base-pair compatibility matrix [N, N].
        ss_constraints: Optional secondary structure constraints [N, N].
        distance_constraints: Optional distance constraints [N, N, n_bins].
        atom_to_token_idx: Mapping from atoms to tokens [N_atom].
        
    Returns:
        design_conditions: Dictionary ready for RNAProDesign.
    """
    design_conditions = {
        "length": length,
        "n_atoms": n_atoms,
        "device": device,
        "dtype": dtype,
        "batch_size": batch_size,
        
        # Positional information (single chain, simple indexing)
        "asym_id": torch.zeros(batch_size, length, device=device, dtype=torch.long),
        "residue_index": torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1),
        "entity_id": torch.zeros(batch_size, length, device=device, dtype=torch.long),
        "token_index": torch.arange(length, device=device).unsqueeze(0).expand(batch_size, -1),
        "sym_id": torch.zeros(batch_size, length, device=device, dtype=torch.long),
        
        # Token bonds (backbone connectivity)
        "token_bonds": torch.zeros(batch_size, length, length, device=device, dtype=dtype),
    }
    
    # Add backbone connectivity (adjacent residues)
    for i in range(length - 1):
        design_conditions["token_bonds"][:, i, i + 1] = 1.0
        design_conditions["token_bonds"][:, i + 1, i] = 1.0
    
    # Add atom to token mapping
    if atom_to_token_idx is not None:
        design_conditions["atom_to_token_idx"] = atom_to_token_idx
    else:
        # Assume one-to-one mapping for simplicity
        design_conditions["atom_to_token_idx"] = torch.arange(
            min(n_atoms, length), device=device
        ).unsqueeze(0).expand(batch_size, -1)
    
    # Add sequence information
    if sequence is not None:
        design_conditions["sequence"] = sequence
    
    # Add pair compatibility matrix
    if pair_compat is not None:
        design_conditions["pair_compat"] = pair_compat
    
    # Add optional constraints
    if ss_constraints is not None:
        design_conditions["ss_constraints"] = ss_constraints
    
    if distance_constraints is not None:
        design_conditions["distance_constraints"] = distance_constraints
    
    return design_conditions
