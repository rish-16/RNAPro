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
Embedders for de novo RNA structure design.

These modules replace sequence-dependent embedders (InputFeatureEmbedder, MSA, templates)
with structure condition embedders for unconditional or constraint-conditioned generation.
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rnapro.model.modules.primitives import LinearNoBias, Transition
from rnapro.openfold_local.model.primitives import LayerNorm


class StructureConditionEmbedder(nn.Module):
    """
    Embeds structural constraints for de novo RNA design.
    
    Replaces sequence-based InputFeatureEmbedder with learnable embeddings
    conditioned on optional structural constraints like:
    - Secondary structure constraints (base pairing)
    - Distance constraints
    - Motif specifications
    
    For unconditional generation, produces learnable position-dependent embeddings.
    """
    
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        max_length: int = 1024,
        n_ss_classes: int = 4,  # unpaired, paired, any, mask
        n_distance_bins: int = 64,
        use_learnable_base: bool = True,
    ):
        """
        Args:
            c_s: Single representation dimension.
            c_z: Pair representation dimension.
            c_s_inputs: Input single representation dimension (for compatibility).
            max_length: Maximum sequence length supported.
            n_ss_classes: Number of secondary structure constraint classes.
            n_distance_bins: Number of distance bins for distance constraints.
            use_learnable_base: Whether to use learnable base embeddings.
        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.max_length = max_length
        self.n_ss_classes = n_ss_classes
        self.n_distance_bins = n_distance_bins
        
        # Learnable base single representation (no sequence info)
        if use_learnable_base:
            self.base_single = nn.Parameter(torch.randn(1, c_s_inputs) * 0.02)
        else:
            self.register_buffer(
                "base_single", 
                torch.zeros(1, c_s_inputs)
            )
        
        # Position embedding (learnable)
        self.pos_embed = nn.Embedding(max_length, c_s_inputs)
        
        # Length embedding (optional, for length conditioning)
        self.length_embed = nn.Embedding(max_length, c_s_inputs)
        
        # Secondary structure constraint embedding (pairwise)
        # Classes: 0=unpaired, 1=paired, 2=any/unknown, 3=mask
        self.ss_embed = nn.Embedding(n_ss_classes, c_z)
        
        # Distance constraint embedding
        self.distance_constraint_embed = LinearNoBias(n_distance_bins, c_z)
        
        # Pairwise position difference embedding
        self.pair_pos_embed = nn.Embedding(2 * max_length + 1, c_z)
        
        # Projection layers for single representation
        self.proj_s = nn.Sequential(
            LayerNorm(c_s_inputs),
            LinearNoBias(c_s_inputs, c_s_inputs),
            nn.SiLU(),
            LinearNoBias(c_s_inputs, c_s_inputs),
        )
        
        # Projection for pair representation initialization
        self.linear_zinit1 = LinearNoBias(c_s_inputs, c_z)
        self.linear_zinit2 = LinearNoBias(c_s_inputs, c_z)
        
        # Final pair projection
        self.proj_z = nn.Sequential(
            LayerNorm(c_z),
            Transition(c_in=c_z, n=2),
        )
        
        print(f"StructureConditionEmbedder initialized: c_s={c_s}, c_z={c_z}, max_length={max_length}")
    
    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        ss_constraints: Optional[torch.Tensor] = None,
        distance_constraints: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate initial single and pair representations from constraints.
        
        Args:
            length: Sequence length.
            device: Target device.
            dtype: Target dtype.
            ss_constraints: Secondary structure constraints [N, N] or [B, N, N].
                Values in {0, 1, 2, 3} for unpaired/paired/any/mask.
            distance_constraints: Distance constraints [N, N, n_bins] or [B, N, N, n_bins].
                One-hot encoded distance bins.
            batch_size: Batch size for output tensors.
            
        Returns:
            s_inputs: [B, N, c_s_inputs] single input representation
            z_init: [B, N, N, c_z] pair representation
        """
        # Position indices
        pos_idx = torch.arange(length, device=device)
        
        # Single representation: base + position
        # [1, c_s_inputs] -> [N, c_s_inputs]
        s = self.base_single.expand(length, -1).to(dtype)
        s = s + self.pos_embed(pos_idx).to(dtype)
        
        # Add length conditioning
        length_idx = torch.tensor(min(length - 1, self.max_length - 1), device=device)
        s = s + self.length_embed(length_idx).unsqueeze(0).to(dtype)
        
        # Apply projection
        s = self.proj_s(s)
        
        # Expand for batch
        s = s.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, c_s_inputs]
        
        # Pair representation: outer product initialization
        z = (
            self.linear_zinit1(s)[..., :, None, :] +  # [B, N, 1, c_z]
            self.linear_zinit2(s)[..., None, :, :]    # [B, 1, N, c_z]
        )  # [B, N, N, c_z]
        
        # Add relative position information
        rel_pos = pos_idx[:, None] - pos_idx[None, :]  # [N, N]
        rel_pos = rel_pos.clamp(-self.max_length, self.max_length) + self.max_length  # Shift to positive
        z = z + self.pair_pos_embed(rel_pos).unsqueeze(0).to(dtype)
        
        # Add secondary structure constraints if provided
        if ss_constraints is not None:
            if ss_constraints.dim() == 2:
                ss_constraints = ss_constraints.unsqueeze(0)  # [1, N, N]
            z = z + self.ss_embed(ss_constraints.long()).to(dtype)
        
        # Add distance constraints if provided
        if distance_constraints is not None:
            if distance_constraints.dim() == 3:
                distance_constraints = distance_constraints.unsqueeze(0)  # [1, N, N, n_bins]
            z = z + self.distance_constraint_embed(distance_constraints.to(dtype))
        
        # Final projection
        z = self.proj_z(z)
        
        return s, z


class UnconditionalEmbedder(nn.Module):
    """
    Minimal embedder for fully unconditional structure generation.
    
    Produces learnable embeddings based only on position and length.
    No sequence or structural constraint information.
    """
    
    def __init__(
        self,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        max_length: int = 1024,
    ):
        """
        Args:
            c_s: Single representation dimension.
            c_z: Pair representation dimension.
            c_s_inputs: Input single dimension.
            max_length: Maximum length supported.
        """
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.max_length = max_length
        
        # Learnable position embeddings
        self.single_pos_embed = nn.Embedding(max_length, c_s_inputs)
        
        # Learnable relative position embeddings for pairs
        self.pair_rel_pos_embed = nn.Embedding(2 * max_length + 1, c_z)
        
        # Projections
        self.linear_zinit1 = LinearNoBias(c_s_inputs, c_z)
        self.linear_zinit2 = LinearNoBias(c_s_inputs, c_z)
        
        # Initialize with small values
        nn.init.normal_(self.single_pos_embed.weight, std=0.02)
        nn.init.normal_(self.pair_rel_pos_embed.weight, std=0.02)
        
        print(f"UnconditionalEmbedder initialized: c_s={c_s}, c_z={c_z}")
    
    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate unconditional embeddings.
        
        Args:
            length: Sequence length.
            device: Target device.
            dtype: Target dtype.
            batch_size: Batch size.
            
        Returns:
            s_inputs: [B, N, c_s_inputs]
            z_init: [B, N, N, c_z]
        """
        pos_idx = torch.arange(length, device=device)
        
        # Single embeddings from position only
        s = self.single_pos_embed(pos_idx).to(dtype)  # [N, c_s_inputs]
        s = s.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, c_s_inputs]
        
        # Pair embeddings from outer product + relative position
        z = (
            self.linear_zinit1(s)[..., :, None, :] +
            self.linear_zinit2(s)[..., None, :, :]
        )  # [B, N, N, c_z]
        
        # Add relative position encoding
        rel_pos = pos_idx[:, None] - pos_idx[None, :]
        rel_pos = rel_pos.clamp(-self.max_length, self.max_length) + self.max_length
        z = z + self.pair_rel_pos_embed(rel_pos).unsqueeze(0).to(dtype)
        
        return s, z


class SecondaryStructureConstraintEncoder(nn.Module):
    """
    Encodes secondary structure constraints (base pairing information).
    
    Takes a dot-bracket notation or pairing matrix and produces
    pairwise embeddings indicating paired/unpaired constraints.
    """
    
    def __init__(
        self,
        c_z: int = 128,
        n_classes: int = 4,
    ):
        """
        Args:
            c_z: Pair representation dimension.
            n_classes: Number of constraint classes (unpaired, paired, any, mask).
        """
        super().__init__()
        self.c_z = c_z
        self.n_classes = n_classes
        
        # Embedding for constraint classes
        self.embed = nn.Embedding(n_classes, c_z)
        
        # Projection
        self.proj = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_z, c_z),
        )
        
        # Zero init for residual connection
        nn.init.zeros_(self.proj[1].weight)
    
    def forward(
        self,
        ss_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode secondary structure constraints.
        
        Args:
            ss_matrix: [B, N, N] constraint matrix with values in {0, 1, 2, 3}.
                0: unpaired constraint
                1: paired constraint  
                2: any (no constraint)
                3: mask (ignore)
                
        Returns:
            z_ss: [B, N, N, c_z] pair embedding from SS constraints.
        """
        z_ss = self.embed(ss_matrix.long())  # [B, N, N, c_z]
        z_ss = self.proj(z_ss)
        return z_ss
    
    @staticmethod
    def dot_bracket_to_matrix(
        dot_bracket: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convert dot-bracket notation to constraint matrix.
        
        Args:
            dot_bracket: String like "(((...)))".
            device: Target device.
            
        Returns:
            matrix: [N, N] constraint matrix.
        """
        n = len(dot_bracket)
        matrix = torch.full((n, n), 2, device=device)  # Default: any
        
        # Find matching pairs
        stack = []
        for i, char in enumerate(dot_bracket):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    j = stack.pop()
                    matrix[i, j] = 1  # paired
                    matrix[j, i] = 1  # symmetric
            elif char == '.':
                matrix[i, i] = 0  # unpaired
        
        return matrix


class DistanceConstraintEncoder(nn.Module):
    """
    Encodes distance constraints for structure generation.
    
    Takes pairwise distance constraints and produces embeddings.
    """
    
    def __init__(
        self,
        c_z: int = 128,
        n_bins: int = 64,
        min_dist: float = 0.0,
        max_dist: float = 50.0,
    ):
        """
        Args:
            c_z: Pair representation dimension.
            n_bins: Number of distance bins.
            min_dist: Minimum distance.
            max_dist: Maximum distance.
        """
        super().__init__()
        self.c_z = c_z
        self.n_bins = n_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        
        # Register bin boundaries
        self.register_buffer(
            "bin_edges",
            torch.linspace(min_dist, max_dist, n_bins + 1)
        )
        
        # Embedding layer
        self.embed = LinearNoBias(n_bins, c_z)
        
        # Projection
        self.proj = nn.Sequential(
            LayerNorm(c_z),
            LinearNoBias(c_z, c_z),
        )
        
        # Zero init
        nn.init.zeros_(self.proj[1].weight)
    
    def discretize_distances(
        self,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert continuous distances to one-hot bins.
        
        Args:
            distances: [B, N, N] distance matrix.
            
        Returns:
            one_hot: [B, N, N, n_bins] one-hot encoded distances.
        """
        # Clamp to valid range
        distances = distances.clamp(self.min_dist, self.max_dist - 1e-6)
        
        # Bucketize
        indices = torch.bucketize(distances, self.bin_edges[:-1]) - 1
        indices = indices.clamp(0, self.n_bins - 1)
        
        # One-hot encode
        one_hot = F.one_hot(indices, self.n_bins).float()
        
        return one_hot
    
    def forward(
        self,
        distance_constraints: torch.Tensor,
        is_one_hot: bool = False,
    ) -> torch.Tensor:
        """
        Encode distance constraints.
        
        Args:
            distance_constraints: Either [B, N, N] continuous distances
                or [B, N, N, n_bins] one-hot encoded.
            is_one_hot: Whether input is already one-hot encoded.
            
        Returns:
            z_dist: [B, N, N, c_z] pair embedding from distance constraints.
        """
        if not is_one_hot:
            distance_constraints = self.discretize_distances(distance_constraints)
        
        z_dist = self.embed(distance_constraints)  # [B, N, N, c_z]
        z_dist = self.proj(z_dist)
        
        return z_dist
