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
Flow Matching implementation for RNA structure generation.

Flow Matching is a simpler alternative to diffusion that learns a vector field
to transport noise to data along straight-line paths.

References:
- Lipman et al., "Flow Matching for Generative Modeling" (2023)
- Tong et al., "Improving and Generalizing Flow-Based Generative Models" (2023)
"""

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from rnapro.model.utils import centre_random_augmentation


class FlowMatchingScheduler:
    """
    Optimal Transport Conditional Flow Matching scheduler.
    
    Learns straight-line interpolation paths between noise and data:
        x_t = (1 - t) * x_0 + t * x_1
        
    where x_0 ~ N(0, I) is noise and x_1 is data.
    """
    
    def __init__(
        self,
        sigma_min: float = 0.001,
        sigma_data: float = 16.0,
    ):
        """
        Args:
            sigma_min: Minimum noise scale for numerical stability.
            sigma_data: Standard deviation of the data distribution (for compatibility).
        """
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        print(f"FlowMatchingScheduler initialized with sigma_min={sigma_min}, sigma_data={sigma_data}")
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample timesteps uniformly from [0, 1].
        
        Args:
            batch_size: Number of timesteps to sample.
            device: Target device.
            dtype: Target dtype.
            
        Returns:
            Tensor of shape [batch_size] with values in [0, 1].
        """
        return torch.rand(batch_size, device=device, dtype=dtype)
    
    def sample_timesteps_with_shape(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample timesteps uniformly from [0, 1] with arbitrary shape.
        
        Args:
            shape: Target shape for timesteps.
            device: Target device.
            dtype: Target dtype.
            
        Returns:
            Tensor of given shape with values in [0, 1].
        """
        return torch.rand(shape, device=device, dtype=dtype)
    
    def interpolate(
        self,
        x_0: torch.Tensor,  # noise
        x_1: torch.Tensor,  # data
        t: torch.Tensor,    # timestep [0, 1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute interpolated point and target velocity.
        
        Optimal transport path:
            x_t = (1 - t) * x_0 + t * x_1
            v_target = x_1 - x_0
        
        Args:
            x_0: Noise samples [..., N_atom, 3]
            x_1: Data samples [..., N_atom, 3]
            t: Timesteps [...] (will be expanded for broadcasting)
            
        Returns:
            x_t: Interpolated points [..., N_atom, 3]
            v_target: Target velocity [..., N_atom, 3]
        """
        # Expand t for broadcasting: [...] -> [..., 1, 1]
        t_expanded = t[..., None, None]
        
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_target = x_1 - x_0
        
        return x_t, v_target
    
    def add_noise(
        self,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise to data for training.
        
        Args:
            x_1: Data samples [..., N_atom, 3]
            t: Timesteps [...]
            
        Returns:
            x_t: Noisy samples [..., N_atom, 3]
            v_target: Target velocity [..., N_atom, 3]
            x_0: Initial noise [..., N_atom, 3]
        """
        x_0 = torch.randn_like(x_1) * self.sigma_data
        x_t, v_target = self.interpolate(x_0, x_1, t)
        return x_t, v_target, x_0
    
    def get_noise_level_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching timestep to equivalent noise level.
        
        This provides compatibility with the existing diffusion module
        which expects noise levels rather than timesteps.
        
        At t=0, noise level is high (sigma_data * s_max equivalent)
        At t=1, noise level is low (sigma_min)
        
        Args:
            t: Timesteps in [0, 1]
            
        Returns:
            Equivalent noise levels for diffusion conditioning.
        """
        # Linear interpolation in log space for smoother transition
        log_sigma_max = torch.log(torch.tensor(self.sigma_data * 10.0))
        log_sigma_min = torch.log(torch.tensor(self.sigma_min))
        
        # At t=0, high noise; at t=1, low noise
        log_sigma = (1 - t) * log_sigma_max + t * log_sigma_min
        return torch.exp(log_sigma)


class FlowMatchingInferenceSampler:
    """
    Euler integration sampler for flow matching inference.
    
    Integrates the learned velocity field from t=0 (noise) to t=1 (data):
        dx/dt = v(x, t)
        x_{t+dt} = x_t + v(x_t, t) * dt
    """
    
    def __init__(
        self,
        n_steps: int = 50,
        sigma_data: float = 16.0,
    ):
        """
        Args:
            n_steps: Number of integration steps.
            sigma_data: Standard deviation for initial noise sampling.
        """
        self.n_steps = n_steps
        self.sigma_data = sigma_data
        print(f"FlowMatchingInferenceSampler initialized with n_steps={n_steps}")
    
    def sample(
        self,
        velocity_net: nn.Module,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        use_center_augmentation: bool = True,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via Euler integration.
        
        Args:
            velocity_net: Network that predicts velocity given (x, t).
            shape: Output shape (batch_size, N_sample, N_atom, 3).
            device: Target device.
            dtype: Target dtype.
            input_feature_dict: Feature dictionary for conditioning.
            s_inputs: Single input embeddings.
            s_trunk: Single trunk embeddings from pairformer.
            z_trunk: Pair trunk embeddings from pairformer.
            use_center_augmentation: Whether to center coordinates at each step.
            chunk_size: Chunk size for attention (optional).
            inplace_safe: Whether inplace operations are safe.
            
        Returns:
            Generated coordinates [batch_size, N_sample, N_atom, 3]
        """
        batch_shape = shape[:-3]
        n_sample = shape[-3]
        n_atom = shape[-2]
        
        # Start from noise at t=0
        x = torch.randn(shape, device=device, dtype=dtype) * self.sigma_data
        
        dt = 1.0 / self.n_steps
        
        for step in range(self.n_steps):
            t = step / self.n_steps
            
            # Center coordinates for stability (optional)
            if use_center_augmentation:
                x = centre_random_augmentation(
                    x_input_coords=x, 
                    N_sample=1
                ).squeeze(dim=-3).to(dtype)
            
            # Create timestep tensor
            t_tensor = torch.full(
                (*batch_shape, n_sample), t, 
                device=device, dtype=dtype
            )
            
            # Convert to noise level for diffusion module compatibility
            t_hat = self._t_to_noise_level(t_tensor)
            
            # Predict velocity (using diffusion module which predicts denoised x)
            # The velocity can be approximated as (x_pred - x) / (1 - t) at each step
            x_pred = velocity_net(
                x_noisy=x,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
            )
            
            # Euler step: x_{t+dt} = x_t + v * dt
            # For flow matching with denoising network: v ≈ (x_1 - x_0) = (x_pred - x_noise) / t
            # But we can also directly interpolate toward prediction
            if t < 0.999:
                v = (x_pred - x) / (1.0 - t + 1e-8)
                x = x + v * dt
            else:
                x = x_pred
        
        return x
    
    def _t_to_noise_level(self, t: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching timestep to noise level for diffusion module.
        
        Maps t in [0, 1] to noise level compatible with diffusion conditioning.
        At t=0 (pure noise), noise_level is high.
        At t=1 (data), noise_level is low.
        """
        # Use exponential schedule similar to diffusion
        s_max = 160.0
        s_min = 4e-4
        rho = 7.0
        
        noise_level = self.sigma_data * (
            s_max ** (1 / rho) + 
            t * (s_min ** (1 / rho) - s_max ** (1 / rho))
        ) ** rho
        
        return noise_level
    
    def sample_with_cfg(
        self,
        velocity_net: Callable,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
        input_feature_dict: dict[str, Any],
        s_inputs_cond: torch.Tensor,
        s_trunk_cond: torch.Tensor,
        z_trunk_cond: torch.Tensor,
        s_inputs_uncond: torch.Tensor,
        s_trunk_uncond: torch.Tensor,
        z_trunk_uncond: torch.Tensor,
        cfg_scale: float = 1.5,
        use_center_augmentation: bool = True,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via Euler integration with Classifier-Free Guidance.
        
        At each step:
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        Args:
            velocity_net: Network that predicts velocity given (x, t).
            shape: Output shape (batch_size, N_sample, N_atom, 3).
            device: Target device.
            dtype: Target dtype.
            input_feature_dict: Feature dictionary for conditioning.
            s_inputs_cond: Conditional single input embeddings.
            s_trunk_cond: Conditional single trunk embeddings.
            z_trunk_cond: Conditional pair trunk embeddings.
            s_inputs_uncond: Unconditional single input embeddings.
            s_trunk_uncond: Unconditional single trunk embeddings.
            z_trunk_uncond: Unconditional pair trunk embeddings.
            cfg_scale: Guidance scale (1.0 = conditional only, >1 = stronger guidance).
            use_center_augmentation: Whether to center coordinates at each step.
            chunk_size: Chunk size for attention (optional).
            inplace_safe: Whether inplace operations are safe.
            
        Returns:
            Generated coordinates [batch_size, N_sample, N_atom, 3]
        """
        batch_shape = shape[:-3]
        n_sample = shape[-3]
        n_atom = shape[-2]
        
        # Start from noise at t=0
        x = torch.randn(shape, device=device, dtype=dtype) * self.sigma_data
        
        dt = 1.0 / self.n_steps
        
        for step in range(self.n_steps):
            t = step / self.n_steps
            
            # Center coordinates for stability (optional)
            if use_center_augmentation:
                x = centre_random_augmentation(
                    x_input_coords=x, 
                    N_sample=1
                ).squeeze(dim=-3).to(dtype)
            
            # Create timestep tensor
            t_tensor = torch.full(
                (*batch_shape, n_sample), t, 
                device=device, dtype=dtype
            )
            
            # Convert to noise level for diffusion module compatibility
            t_hat = self._t_to_noise_level(t_tensor)
            
            # Predict CONDITIONAL velocity
            x_pred_cond = velocity_net(
                x_noisy=x,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs_cond,
                s_trunk=s_trunk_cond,
                z_trunk=z_trunk_cond,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
            )
            
            # Predict UNCONDITIONAL velocity
            x_pred_uncond = velocity_net(
                x_noisy=x,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs_uncond,
                s_trunk=s_trunk_uncond,
                z_trunk=z_trunk_uncond,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
            )
            
            # CFG interpolation: pred = uncond + scale * (cond - uncond)
            x_pred = x_pred_uncond + cfg_scale * (x_pred_cond - x_pred_uncond)
            
            # Euler step
            if t < 0.999:
                v = (x_pred - x) / (1.0 - t + 1e-8)
                x = x + v * dt
            else:
                x = x_pred
        
        return x


def sample_flow_matching_training(
    flow_scheduler: FlowMatchingScheduler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    use_conditioning: bool = True,
    diffusion_chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Training step for flow matching.
    
    Args:
        flow_scheduler: Flow matching scheduler.
        denoise_net: Denoising network (DiffusionModule).
        label_dict: Dictionary containing:
            - "coordinate": Ground truth coordinates [..., N_atom, 3]
            - "coordinate_mask": Atom mask [..., N_atom]
        input_feature_dict: Input features for conditioning.
        s_inputs: Single input embeddings [..., N_tokens, c_s_inputs].
        s_trunk: Single trunk embeddings [..., N_tokens, c_s].
        z_trunk: Pair trunk embeddings [..., N_tokens, N_tokens, c_z].
        N_sample: Number of training samples.
        use_conditioning: Whether to use trunk conditioning.
        diffusion_chunk_size: Chunk size for memory efficiency.
        
    Returns:
        x_t: Interpolated coordinates [..., N_sample, N_atom, 3]
        x_pred: Predicted coordinates [..., N_sample, N_atom, 3]
        v_target: Target velocity [..., N_sample, N_atom, 3]
        t: Sampled timesteps [..., N_sample]
    """
    batch_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    
    # Create N_sample augmented versions of ground truth
    x_1 = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict.get("coordinate_mask"),
    ).to(dtype)  # [..., N_sample, N_atom, 3]
    
    # Sample timesteps for each sample
    t = flow_scheduler.sample_timesteps_with_shape(
        shape=(*batch_shape, N_sample),
        device=device,
        dtype=dtype,
    )  # [..., N_sample]
    
    # Generate noise and interpolate
    x_t, v_target, x_0 = flow_scheduler.add_noise(x_1, t)
    
    # Convert timestep to noise level for diffusion module
    t_hat = flow_scheduler.get_noise_level_from_t(t)
    
    # Get predictions from network
    if diffusion_chunk_size is None:
        x_pred = denoise_net(
            x_noisy=x_t,
            t_hat_noise_level=t_hat,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            use_conditioning=use_conditioning,
        )
    else:
        x_pred = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            start_idx = i * diffusion_chunk_size
            end_idx = min((i + 1) * diffusion_chunk_size, N_sample)
            
            x_noisy_i = x_t[..., start_idx:end_idx, :, :]
            t_hat_i = t_hat[..., start_idx:end_idx]
            
            x_pred_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                use_conditioning=use_conditioning,
            )
            x_pred.append(x_pred_i)
        x_pred = torch.cat(x_pred, dim=-3)
    
    return x_t, x_pred, v_target, t


def flow_matching_loss(
    x_pred: torch.Tensor,
    x_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute flow matching loss (MSE between predicted and target coordinates).
    
    For flow matching, we can either:
    1. Predict velocity and compute MSE with target velocity
    2. Predict final coordinates and compute MSE with ground truth
    
    This implementation uses coordinate prediction for compatibility
    with the existing diffusion module.
    
    Args:
        x_pred: Predicted coordinates [..., N_sample, N_atom, 3]
        x_target: Target coordinates [..., N_sample, N_atom, 3]  
        mask: Atom mask [..., N_atom] or [..., N_sample, N_atom]
        reduction: Reduction method ("mean", "sum", "none")
        
    Returns:
        Loss value.
    """
    # Squared error
    sq_error = (x_pred - x_target).pow(2).sum(dim=-1)  # [..., N_sample, N_atom]
    
    if mask is not None:
        # Expand mask if needed
        if mask.dim() == sq_error.dim() - 1:
            mask = mask.unsqueeze(-2)  # [..., 1, N_atom]
        
        sq_error = sq_error * mask
        
        if reduction == "mean":
            return sq_error.sum() / (mask.sum() * sq_error.shape[-2] + 1e-8)
        elif reduction == "sum":
            return sq_error.sum()
        else:
            return sq_error
    else:
        if reduction == "mean":
            return sq_error.mean()
        elif reduction == "sum":
            return sq_error.sum()
        else:
            return sq_error