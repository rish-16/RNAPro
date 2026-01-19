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
Sampling script for RNA de novo structure design.

This script generates RNA 3D structures using a trained RNAProDesign model.
Supports unconditional generation and constraint-conditioned generation.
"""

import argparse
import os
import logging
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from rnapro.model.RNAProDesign import RNAProDesign, create_design_conditions
from rnapro.model.modules.design_embedders import SecondaryStructureConstraintEncoder
from rnapro.config import parse_configs
from configs.configs_design import configs as design_configs
from rnapro.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: torch.device,
    configs: Optional[Dict] = None,
) -> RNAProDesign:
    """
    Load trained RNAProDesign model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        device: Target device.
        configs: Optional config override.
        
    Returns:
        Loaded model.
    """
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configs from checkpoint or use provided
    if configs is None:
        if "configs" in checkpoint:
            from ml_collections.config_dict import ConfigDict
            configs = ConfigDict(checkpoint["configs"])
        else:
            from ml_collections.config_dict import ConfigDict
            configs = ConfigDict(design_configs)
    
    # Create model
    model = RNAProDesign(configs)
    
    # Load weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    cleaned_state = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "")
        cleaned_state[clean_key] = value
    
    model.load_state_dict(cleaned_state, strict=False)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, configs


def generate_unconditional(
    model: RNAProDesign,
    length: int,
    n_samples: int = 5,
    n_steps: int = 50,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate RNA structures unconditionally.
    
    Args:
        model: Trained RNAProDesign model.
        length: Target sequence length.
        n_samples: Number of structures to generate.
        n_steps: Number of integration steps.
        device: Target device.
        seed: Random seed (optional).
        
    Returns:
        coordinates: [n_samples, N_atom, 3] generated coordinates.
    """
    if seed is not None:
        seed_everything(seed=seed, deterministic=False)
    
    logger.info(f"Generating {n_samples} structures of length {length}")
    
    # Create design conditions
    design_conditions = create_design_conditions(
        length=length,
        n_atoms=length,  # Assume 1 atom per residue for simplicity
        device=device,
        dtype=torch.float32,
        batch_size=1,
    )
    
    # Generate
    with torch.no_grad():
        coordinates = model.sample(
            design_conditions=design_conditions,
            n_samples=n_samples,
            n_steps=n_steps,
        )
    
    # Remove batch dimension
    coordinates = coordinates.squeeze(0)  # [n_samples, N_atom, 3]
    
    logger.info(f"Generated coordinates shape: {coordinates.shape}")
    return coordinates


def generate_with_ss_constraint(
    model: RNAProDesign,
    dot_bracket: str,
    n_samples: int = 5,
    n_steps: int = 50,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate RNA structures conditioned on secondary structure.
    
    Args:
        model: Trained RNAProDesign model.
        dot_bracket: Secondary structure in dot-bracket notation.
        n_samples: Number of structures to generate.
        n_steps: Number of integration steps.
        device: Target device.
        seed: Random seed (optional).
        
    Returns:
        coordinates: [n_samples, N_atom, 3] generated coordinates.
    """
    if seed is not None:
        seed_everything(seed=seed, deterministic=False)
    
    length = len(dot_bracket)
    logger.info(f"Generating {n_samples} structures for SS: {dot_bracket}")
    
    # Convert dot-bracket to constraint matrix
    ss_matrix = SecondaryStructureConstraintEncoder.dot_bracket_to_matrix(
        dot_bracket, device
    )
    
    # Create design conditions
    design_conditions = create_design_conditions(
        length=length,
        n_atoms=length,
        device=device,
        dtype=torch.float32,
        batch_size=1,
        ss_constraints=ss_matrix,
    )
    
    # Generate
    with torch.no_grad():
        coordinates = model.sample(
            design_conditions=design_conditions,
            n_samples=n_samples,
            n_steps=n_steps,
        )
    
    coordinates = coordinates.squeeze(0)
    
    logger.info(f"Generated coordinates shape: {coordinates.shape}")
    return coordinates


def generate_with_cfg(
    model: RNAProDesign,
    length: int,
    dot_bracket: Optional[str] = None,
    sequence: Optional[str] = None,
    n_samples: int = 5,
    n_steps: int = 50,
    cfg_scale: float = 1.5,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate RNA structures using Classifier-Free Guidance.
    
    CFG interpolates between conditional and unconditional predictions:
    pred = uncond + cfg_scale * (cond - uncond)
    
    - cfg_scale = 1.0: Pure conditional (same as generate_with_ss_constraint)
    - cfg_scale > 1.0: Stronger conditioning (higher fidelity to constraints)
    - cfg_scale < 1.0: More diversity, less adherence to constraints
    - cfg_scale = 0.0: Pure unconditional
    
    Args:
        model: Trained RNAProDesign model.
        length: Sequence length (used if dot_bracket not provided).
        dot_bracket: Secondary structure in dot-bracket notation (optional).
        sequence: RNA sequence string (optional, e.g., "AUGCAUGC").
        n_samples: Number of structures to generate.
        n_steps: Number of integration steps.
        cfg_scale: Guidance scale.
        device: Target device.
        seed: Random seed (optional).
        
    Returns:
        coordinates: [n_samples, N_atom, 3] generated coordinates.
    """
    if seed is not None:
        seed_everything(seed=seed, deterministic=False)
    
    # Determine length from dot_bracket if provided
    if dot_bracket is not None:
        length = len(dot_bracket)
    
    logger.info(f"Generating {n_samples} structures with CFG (scale={cfg_scale})")
    if dot_bracket:
        logger.info(f"  SS constraint: {dot_bracket}")
    if sequence:
        logger.info(f"  Sequence: {sequence[:20]}..." if len(sequence) > 20 else f"  Sequence: {sequence}")
    
    # Build design conditions
    ss_matrix = None
    if dot_bracket is not None:
        ss_matrix = SecondaryStructureConstraintEncoder.dot_bracket_to_matrix(
            dot_bracket, device
        )
    
    # Convert sequence to indices
    seq_tensor = None
    pair_compat = None
    if sequence is not None:
        nuc_to_idx = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4, "T": 1}  # T->U
        seq_list = [nuc_to_idx.get(c.upper(), 4) for c in sequence]
        seq_tensor = torch.tensor(seq_list, dtype=torch.long, device=device)
        
        # Compute pair compatibility
        valid_pairs = {(0, 1), (1, 0), (2, 3), (3, 2), (2, 1), (1, 2)}  # A-U, G-C, G-U
        n = len(sequence)
        pair_compat = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(n):
                if (seq_list[i], seq_list[j]) in valid_pairs:
                    pair_compat[i, j] = 1.0
    
    # Create design conditions
    design_conditions = create_design_conditions(
        length=length,
        n_atoms=length,
        device=device,
        dtype=torch.float32,
        batch_size=1,
        sequence=seq_tensor,
        pair_compat=pair_compat,
        ss_constraints=ss_matrix,
    )
    
    # Generate with CFG
    with torch.no_grad():
        coordinates = model.sample_with_cfg(
            design_conditions=design_conditions,
            n_samples=n_samples,
            n_steps=n_steps,
            cfg_scale=cfg_scale,
        )
    
    coordinates = coordinates.squeeze(0)
    
    logger.info(f"Generated coordinates shape: {coordinates.shape}")
    return coordinates


def generate_with_distance_constraint(
    model: RNAProDesign,
    length: int,
    distance_constraints: torch.Tensor,
    n_samples: int = 5,
    n_steps: int = 50,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate RNA structures conditioned on distance constraints.
    
    Args:
        model: Trained RNAProDesign model.
        length: Sequence length.
        distance_constraints: [N, N, n_bins] distance constraints.
        n_samples: Number of structures to generate.
        n_steps: Number of integration steps.
        device: Target device.
        seed: Random seed (optional).
        
    Returns:
        coordinates: [n_samples, N_atom, 3] generated coordinates.
    """
    if seed is not None:
        seed_everything(seed=seed, deterministic=False)
    
    logger.info(f"Generating {n_samples} structures with distance constraints")
    
    # Create design conditions
    design_conditions = create_design_conditions(
        length=length,
        n_atoms=length,
        device=device,
        dtype=torch.float32,
        batch_size=1,
        distance_constraints=distance_constraints,
    )
    
    # Generate
    with torch.no_grad():
        coordinates = model.sample(
            design_conditions=design_conditions,
            n_samples=n_samples,
            n_steps=n_steps,
        )
    
    coordinates = coordinates.squeeze(0)
    
    logger.info(f"Generated coordinates shape: {coordinates.shape}")
    return coordinates


def save_coordinates_pdb(
    coordinates: torch.Tensor,
    output_path: str,
    atom_name: str = "C4'",
    residue_name: str = "N",
):
    """
    Save coordinates as PDB file.
    
    Args:
        coordinates: [N_atom, 3] coordinates.
        output_path: Output file path.
        atom_name: Atom name for PDB.
        residue_name: Residue name for PDB.
    """
    coords = coordinates.cpu().numpy()
    
    with open(output_path, "w") as f:
        f.write("REMARK Generated by RNAProDesign\n")
        
        for i, (x, y, z) in enumerate(coords):
            f.write(
                f"ATOM  {i+1:5d}  {atom_name:4s}{residue_name:3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        
        f.write("END\n")
    
    logger.info(f"Saved PDB to {output_path}")


def save_coordinates_npz(
    coordinates: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict] = None,
):
    """
    Save coordinates as NPZ file.
    
    Args:
        coordinates: [n_samples, N_atom, 3] coordinates.
        output_path: Output file path.
        metadata: Optional metadata to save.
    """
    data = {
        "coordinates": coordinates.cpu().numpy(),
    }
    
    if metadata is not None:
        data.update(metadata)
    
    np.savez(output_path, **data)
    logger.info(f"Saved NPZ to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RNA Structure Design Sampling")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./generated_structures",
        help="Output directory for generated structures"
    )
    parser.add_argument(
        "--length", type=int, default=50,
        help="Length of RNA to generate (for unconditional)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of structures to generate"
    )
    parser.add_argument(
        "--n_steps", type=int, default=50,
        help="Number of flow matching integration steps"
    )
    parser.add_argument(
        "--dot_bracket", type=str, default=None,
        help="Secondary structure constraint in dot-bracket notation"
    )
    parser.add_argument(
        "--sequence", type=str, default=None,
        help="RNA sequence (e.g., 'AUGCAUGC')"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=1.5,
        help="Classifier-Free Guidance scale (1.0=conditional, >1.0=stronger guidance)"
    )
    parser.add_argument(
        "--use_cfg", action="store_true",
        help="Use Classifier-Free Guidance for sampling"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--save_pdb", action="store_true",
        help="Save individual PDB files"
    )
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, configs = load_model(args.checkpoint, device)
    
    # Generate structures
    if args.use_cfg or args.cfg_scale != 1.0:
        # Use Classifier-Free Guidance
        coordinates = generate_with_cfg(
            model=model,
            length=args.length,
            dot_bracket=args.dot_bracket,
            sequence=args.sequence,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            cfg_scale=args.cfg_scale,
            device=device,
            seed=args.seed,
        )
        prefix = f"cfg_{args.cfg_scale}"
    elif args.dot_bracket is not None:
        # Constraint-conditioned generation (without CFG)
        coordinates = generate_with_ss_constraint(
            model=model,
            dot_bracket=args.dot_bracket,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            device=device,
            seed=args.seed,
        )
        prefix = "ss_constrained"
    else:
        # Unconditional generation
        coordinates = generate_unconditional(
            model=model,
            length=args.length,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            device=device,
            seed=args.seed,
        )
        prefix = "unconditional"
    
    # Save all coordinates as NPZ
    npz_path = os.path.join(args.output_dir, f"{prefix}_samples.npz")
    metadata = {
        "n_samples": args.n_samples,
        "n_steps": args.n_steps,
        "seed": args.seed,
    }
    if args.dot_bracket is not None:
        metadata["dot_bracket"] = args.dot_bracket
    else:
        metadata["length"] = args.length
    
    save_coordinates_npz(coordinates, npz_path, metadata)
    
    # Optionally save individual PDB files
    if args.save_pdb:
        for i in range(coordinates.shape[0]):
            pdb_path = os.path.join(args.output_dir, f"{prefix}_sample_{i}.pdb")
            save_coordinates_pdb(coordinates[i], pdb_path)
    
    logger.info(f"Generation complete. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()

"""
from rnapro.model.RNAProDesign import RNAProDesign, create_design_conditions

# Create model
model = RNAProDesign(configs)

# Generate unconditionally
design_conditions = create_design_conditions(length=50, n_atoms=50, device=device, dtype=dtype)
coordinates = model.sample(design_conditions, n_samples=5, n_steps=50)
"""