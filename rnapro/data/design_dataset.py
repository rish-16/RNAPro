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
Dataset classes for RNA de novo structure design.

These datasets provide structure data with sequence and secondary structure
information for training generative models with proper pair representations.
"""

import os
import glob
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from rnapro.utils.logger import get_logger

logger = get_logger(__name__)

# Nucleotide mapping
NUC_TO_IDX = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4}  # N = unknown
IDX_TO_NUC = {0: "A", 1: "U", 2: "G", 3: "C", 4: "N"}

# Standard residue name mapping to single letter
RESNAME_TO_NUC = {
    "A": "A", "ADE": "A", "RA": "A", "DA": "A",
    "U": "U", "URA": "U", "RU": "U", "URI": "U",
    "G": "G", "GUA": "G", "RG": "G", "DG": "G",
    "C": "C", "CYT": "C", "RC": "C", "DC": "C",
    "T": "U", "THY": "U", "DT": "U",  # Treat T as U for RNA
}

# Secondary structure constraint classes
SS_UNPAIRED = 0    # Definitely unpaired
SS_PAIRED = 1      # Definitely paired (with another residue)
SS_ANY = 2         # Unknown/any
SS_MASK = 3        # Masked (for training)

# Watson-Crick and wobble pair compatibility
# (A-U, U-A, G-C, C-G, G-U, U-G)
VALID_PAIRS = {
    (0, 1), (1, 0),  # A-U, U-A
    (2, 3), (3, 2),  # G-C, C-G
    (2, 1), (1, 2),  # G-U, U-G (wobble)
}


def detect_base_pairs_from_coords(
    c4_coords: torch.Tensor,
    sequence: torch.Tensor,
    max_pair_dist: float = 25.0,
    min_seq_sep: int = 4,
) -> torch.Tensor:
    """
    Detect base pairs from C4' coordinates using distance heuristics.
    
    For proper base pairing, the C4'-C4' distance is typically:
    - Watson-Crick pairs: ~15-18 Å
    - Wobble pairs: ~16-19 Å
    
    We use a relaxed threshold and filter by sequence compatibility.
    
    Args:
        c4_coords: [N, 3] C4' atom coordinates.
        sequence: [N] nucleotide indices (0=A, 1=U, 2=G, 3=C, 4=N).
        max_pair_dist: Maximum C4'-C4' distance to consider.
        min_seq_sep: Minimum sequence separation for valid pairs.
        
    Returns:
        ss_matrix: [N, N] secondary structure matrix.
            Values: 0=unpaired, 1=paired, 2=any/unknown.
    """
    n = c4_coords.shape[0]
    
    # Compute pairwise distances
    diff = c4_coords[:, None, :] - c4_coords[None, :, :]  # [N, N, 3]
    distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [N, N]
    
    # Initialize with "any" (unknown)
    ss_matrix = torch.full((n, n), SS_ANY, dtype=torch.long)
    
    # Set diagonal to unpaired (residue can't pair with itself)
    ss_matrix.fill_diagonal_(SS_UNPAIRED)
    
    # Find potential pairs
    for i in range(n):
        for j in range(i + min_seq_sep, n):
            dist = distances[i, j].item()
            
            # Check distance criterion (typical C4'-C4' for base pairs)
            if 12.0 <= dist <= max_pair_dist:
                # Check sequence compatibility
                nuc_i = sequence[i].item()
                nuc_j = sequence[j].item()
                
                if (nuc_i, nuc_j) in VALID_PAIRS:
                    # Mark as paired
                    ss_matrix[i, j] = SS_PAIRED
                    ss_matrix[j, i] = SS_PAIRED
    
    return ss_matrix


def compute_pair_compatibility_matrix(sequence: torch.Tensor) -> torch.Tensor:
    """
    Compute which pairs of residues CAN form base pairs based on sequence.
    
    This encodes Watson-Crick (A-U, G-C) and wobble (G-U) pairing rules.
    
    Args:
        sequence: [N] nucleotide indices.
        
    Returns:
        compat_matrix: [N, N] binary compatibility matrix.
    """
    n = sequence.shape[0]
    compat = torch.zeros(n, n, dtype=torch.float32)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                nuc_i = sequence[i].item()
                nuc_j = sequence[j].item()
                if (nuc_i, nuc_j) in VALID_PAIRS:
                    compat[i, j] = 1.0
    
    return compat


class RNADesignDataset(Dataset):
    """
    Dataset for structure-conditioned RNA design at C4' level.
    
    Each sample contains:
    - Target 3D coordinates (C4' atoms only, one per residue)
    - Sequence information (nucleotide identity)
    - Secondary structure constraints (detected from coordinates)
    - Structural metadata (lengths, masks)
    
    C4' Level Representation:
    - n_atoms == n_tokens == n_residues
    - Each residue is represented by its C4' atom position
    - This simplifies the model and is suitable for initial experiments
    - Can be extended to all-atom representation later
    """
    
    def __init__(
        self,
        data_dir: str,
        file_pattern: str = "*.pt",
        max_length: int = 512,
        min_length: int = 10,
        use_ss_constraints: bool = False,
        use_distance_constraints: bool = False,
        augment_coords: bool = True,
        random_seed: int = 42,
        atom_level: str = "c4prime",  # "c4prime" or "all_atom"
    ):
        """
        Args:
            data_dir: Directory containing preprocessed structure files.
            file_pattern: Glob pattern for structure files.
            max_length: Maximum sequence length to include.
            min_length: Minimum sequence length to include.
            use_ss_constraints: Whether to include secondary structure constraints.
            use_distance_constraints: Whether to include distance constraints.
            augment_coords: Whether to apply random rotations/translations.
            random_seed: Random seed for reproducibility.
            atom_level: Atom representation level ("c4prime" or "all_atom").
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.min_length = min_length
        self.use_ss_constraints = use_ss_constraints
        self.use_distance_constraints = use_distance_constraints
        self.augment_coords = augment_coords
        self.atom_level = atom_level
        
        # Find all structure files
        self.structure_files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        # Filter by length if metadata available
        self.structure_files = self._filter_by_length(self.structure_files)
        
        logger.info(
            f"RNADesignDataset initialized with {len(self.structure_files)} structures "
            f"from {data_dir} (atom_level={atom_level})"
        )
        
        random.seed(random_seed)
    
    def _filter_by_length(self, files: List[str]) -> List[str]:
        """Filter files by sequence length."""
        filtered = []
        for f in files:
            # Try to get length from filename or metadata
            # Format: {pdb_id}_{length}.pt
            try:
                basename = os.path.basename(f)
                parts = basename.replace(".pt", "").split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    length = int(parts[-1])
                    if self.min_length <= length <= self.max_length:
                        filtered.append(f)
                else:
                    # Can't determine length, include by default
                    filtered.append(f)
            except Exception:
                filtered.append(f)
        
        return filtered
    
    def __len__(self) -> int:
        return len(self.structure_files)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load a structure sample.
        
        Returns:
            design_conditions: Dictionary with design inputs.
            label_dict: Dictionary with target coordinates.
        """
        # Load preprocessed data
        data = torch.load(self.structure_files[idx], map_location="cpu")
        
        # Extract coordinates
        coordinates = data["coordinates"]  # [N_atom, 3]
        coordinate_mask = data.get("coordinate_mask", torch.ones(coordinates.shape[0]))
        
        # Get dimensions
        n_atoms = coordinates.shape[0]
        n_tokens = data.get("n_tokens", n_atoms)  # Default: one atom per token
        
        # Extract sequence (nucleotide indices)
        if "sequence" in data:
            sequence = data["sequence"]  # [N] tensor of indices
        else:
            # Default to unknown nucleotides if not provided
            sequence = torch.full((n_tokens,), NUC_TO_IDX["N"], dtype=torch.long)
        
        # Augment coordinates (random rotation/translation)
        if self.augment_coords and self.training:
            coordinates = self._augment_coordinates(coordinates, coordinate_mask)
        
        # Build design conditions with sequence info
        design_conditions = {
            "length": n_tokens,
            "n_atoms": n_atoms,
            # Sequence information
            "sequence": sequence,  # [N] nucleotide indices (0=A, 1=U, 2=G, 3=C, 4=N)
            # Positional information
            "asym_id": torch.zeros(n_tokens, dtype=torch.long),
            "residue_index": torch.arange(n_tokens),
            "entity_id": torch.zeros(n_tokens, dtype=torch.long),
            "token_index": torch.arange(n_tokens),
            "sym_id": torch.zeros(n_tokens, dtype=torch.long),
            # Atom to token mapping
            "atom_to_token_idx": data.get(
                "atom_to_token_idx", 
                torch.arange(min(n_atoms, n_tokens))
            ),
        }
        
        # Add token bonds (backbone connectivity)
        token_bonds = torch.zeros(n_tokens, n_tokens)
        for i in range(n_tokens - 1):
            token_bonds[i, i + 1] = 1.0
            token_bonds[i + 1, i] = 1.0
        design_conditions["token_bonds"] = token_bonds
        
        # Add base pair compatibility (from sequence)
        design_conditions["pair_compat"] = compute_pair_compatibility_matrix(sequence)
        
        # Compute or load secondary structure constraints
        if self.use_ss_constraints:
            if "ss_matrix" in data:
                design_conditions["ss_constraints"] = data["ss_matrix"]
            else:
                # Detect base pairs from 3D coordinates
                design_conditions["ss_constraints"] = detect_base_pairs_from_coords(
                    coordinates, sequence
                )
        
        if self.use_distance_constraints:
            # Compute distance constraints from coordinates
            design_conditions["distance_constraints"] = self._compute_distance_constraints(
                coordinates, coordinate_mask
            )
        
        # Build label dictionary
        label_dict = {
            "coordinate": coordinates,
            "coordinate_mask": coordinate_mask,
        }
        
        return design_conditions, label_dict
    
    def _augment_coordinates(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply random rotation and translation to coordinates.
        
        Args:
            coords: [N, 3] coordinates.
            mask: [N] validity mask.
            
        Returns:
            Augmented coordinates.
        """
        # Center coordinates
        valid_coords = coords[mask.bool()]
        center = valid_coords.mean(dim=0)
        coords = coords - center
        
        # Random rotation
        rotation = self._random_rotation_matrix()
        coords = coords @ rotation.T
        
        # Random translation (small)
        translation = torch.randn(3) * 0.1
        coords = coords + translation
        
        return coords
    
    def _random_rotation_matrix(self) -> torch.Tensor:
        """Generate a random 3D rotation matrix."""
        # Use QR decomposition of random matrix
        random_matrix = torch.randn(3, 3)
        q, r = torch.linalg.qr(random_matrix)
        # Ensure proper rotation (det = 1)
        q = q * torch.sign(torch.diag(r))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q
    
    def _compute_distance_constraints(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        n_bins: int = 64,
        max_dist: float = 50.0,
    ) -> torch.Tensor:
        """
        Compute binned distance constraints from coordinates.
        
        Args:
            coords: [N, 3] coordinates.
            mask: [N] validity mask.
            n_bins: Number of distance bins.
            max_dist: Maximum distance.
            
        Returns:
            distance_constraints: [N, N, n_bins] one-hot distances.
        """
        n = coords.shape[0]
        
        # Compute pairwise distances
        diff = coords[:, None, :] - coords[None, :, :]  # [N, N, 3]
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [N, N]
        
        # Clamp and discretize
        distances = distances.clamp(0, max_dist - 1e-6)
        bin_edges = torch.linspace(0, max_dist, n_bins + 1)
        indices = torch.bucketize(distances, bin_edges[:-1]) - 1
        indices = indices.clamp(0, n_bins - 1)
        
        # One-hot encode
        one_hot = F.one_hot(indices, n_bins).float()  # [N, N, n_bins]
        
        return one_hot
    
    @property
    def training(self) -> bool:
        """Check if in training mode."""
        return getattr(self, "_training", True)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self._training = mode
        return self
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)


class RNADesignDatasetFromPDB(RNADesignDataset):
    """
    Dataset that loads structures directly from PDB/mmCIF files.
    
    This is a convenience class for when preprocessed .pt files
    are not available.
    """
    
    def __init__(
        self,
        pdb_dir: str,
        file_pattern: str = "*.pdb",
        atom_selection: str = "backbone",  # "backbone", "all", "c4prime"
        **kwargs,
    ):
        """
        Args:
            pdb_dir: Directory containing PDB files.
            file_pattern: Glob pattern for PDB files.
            atom_selection: Which atoms to include.
            **kwargs: Additional arguments for parent class.
        """
        self.atom_selection = atom_selection
        
        # Don't call parent __init__ yet
        self.data_dir = pdb_dir
        self.max_length = kwargs.get("max_length", 512)
        self.min_length = kwargs.get("min_length", 10)
        self.use_ss_constraints = kwargs.get("use_ss_constraints", False)
        self.use_distance_constraints = kwargs.get("use_distance_constraints", False)
        self.augment_coords = kwargs.get("augment_coords", True)
        
        # Find PDB files
        self.structure_files = sorted(glob.glob(os.path.join(pdb_dir, file_pattern)))
        
        logger.info(
            f"RNADesignDatasetFromPDB initialized with {len(self.structure_files)} files"
        )
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load structure from PDB file."""
        pdb_path = self.structure_files[idx]
        
        try:
            coordinates, sequence, n_residues = self._parse_pdb(pdb_path)
        except Exception as e:
            logger.warning(f"Error parsing {pdb_path}: {e}")
            # Return dummy data with proper sequence
            coordinates = torch.randn(10, 3)
            sequence = torch.full((10,), NUC_TO_IDX["N"], dtype=torch.long)
            n_residues = 10
        
        n_atoms = coordinates.shape[0]
        coordinate_mask = torch.ones(n_atoms)
        
        # Augment
        if self.augment_coords and self.training:
            coordinates = self._augment_coordinates(coordinates, coordinate_mask)
        
        # Build design conditions with sequence
        design_conditions = {
            "length": n_residues,
            "n_atoms": n_atoms,
            "sequence": sequence,  # [N] nucleotide indices
            "asym_id": torch.zeros(n_residues, dtype=torch.long),
            "residue_index": torch.arange(n_residues),
            "entity_id": torch.zeros(n_residues, dtype=torch.long),
            "token_index": torch.arange(n_residues),
            "sym_id": torch.zeros(n_residues, dtype=torch.long),
            "atom_to_token_idx": self._create_atom_to_token_mapping(n_atoms, n_residues),
        }
        
        # Token bonds (backbone connectivity)
        token_bonds = torch.zeros(n_residues, n_residues)
        for i in range(n_residues - 1):
            token_bonds[i, i + 1] = 1.0
            token_bonds[i + 1, i] = 1.0
        design_conditions["token_bonds"] = token_bonds
        
        # Add base pair compatibility (from sequence)
        design_conditions["pair_compat"] = compute_pair_compatibility_matrix(sequence)
        
        # Compute secondary structure from coordinates
        if self.use_ss_constraints:
            design_conditions["ss_constraints"] = detect_base_pairs_from_coords(
                coordinates[:n_residues] if self.atom_selection == "c4prime" else coordinates[:n_residues],
                sequence
            )
        
        if self.use_distance_constraints:
            design_conditions["distance_constraints"] = self._compute_distance_constraints(
                coordinates, coordinate_mask
            )
        
        label_dict = {
            "coordinate": coordinates,
            "coordinate_mask": coordinate_mask,
        }
        
        return design_conditions, label_dict
    
    def _parse_pdb(self, pdb_path: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Parse PDB file and extract coordinates and sequence.
        
        Args:
            pdb_path: Path to PDB file.
            
        Returns:
            coordinates: [N_atom, 3] tensor.
            sequence: [N_res] tensor of nucleotide indices.
            n_residues: Number of residues.
        """
        coordinates = []
        residue_data = {}  # res_id -> (nuc_char, coords_list)
        
        # Define which atoms to select based on atom_selection
        if self.atom_selection == "c4prime":
            target_atoms = {"C4'", "C4*"}
        elif self.atom_selection == "backbone":
            target_atoms = {"P", "C4'", "C4*", "O5'", "O5*", "C5'", "C5*", "O3'", "O3*"}
        else:  # all
            target_atoms = None
        
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    res_id = int(line[22:26].strip())
                    chain_id = line[21]
                    unique_res_id = (chain_id, res_id)
                    
                    # Check if this is an RNA residue
                    res_name = line[17:20].strip()
                    
                    # Get nucleotide character
                    nuc_char = RESNAME_TO_NUC.get(res_name, None)
                    if nuc_char is None:
                        continue
                    
                    # Apply atom selection
                    if target_atoms is not None and atom_name not in target_atoms:
                        continue
                    
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        if unique_res_id not in residue_data:
                            residue_data[unique_res_id] = (nuc_char, [])
                        residue_data[unique_res_id][1].append([x, y, z])
                        
                    except ValueError:
                        continue
        
        if len(residue_data) == 0:
            raise ValueError(f"No valid RNA residues found in {pdb_path}")
        
        # Sort by residue ID and extract coordinates/sequence
        sorted_res_ids = sorted(residue_data.keys(), key=lambda x: (x[0], x[1]))
        
        sequence = []
        all_coords = []
        
        for res_id in sorted_res_ids:
            nuc_char, coords_list = residue_data[res_id]
            sequence.append(NUC_TO_IDX[nuc_char])
            all_coords.extend(coords_list)
        
        n_residues = len(sorted_res_ids)
        coordinates = torch.tensor(all_coords, dtype=torch.float32)
        sequence = torch.tensor(sequence, dtype=torch.long)
        
        return coordinates, sequence, n_residues
    
    def _create_atom_to_token_mapping(
        self,
        n_atoms: int,
        n_residues: int,
    ) -> torch.Tensor:
        """Create mapping from atoms to tokens (residues)."""
        if self.atom_selection == "c4prime":
            # One atom per residue
            return torch.arange(min(n_atoms, n_residues))
        else:
            # Distribute atoms across residues
            atoms_per_residue = n_atoms // n_residues
            mapping = []
            for res_idx in range(n_residues):
                for _ in range(atoms_per_residue):
                    if len(mapping) < n_atoms:
                        mapping.append(res_idx)
            # Handle remaining atoms
            while len(mapping) < n_atoms:
                mapping.append(n_residues - 1)
            
            return torch.tensor(mapping[:n_atoms], dtype=torch.long)


def collate_design_batch(
    batch: List[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Collate function for design dataset.
    
    Handles variable-length sequences by padding.
    
    Args:
        batch: List of (design_conditions, label_dict) tuples.
        
    Returns:
        Batched design_conditions and label_dict.
    """
    design_conditions_list, label_dict_list = zip(*batch)
    
    # Get max lengths
    max_tokens = max(d["length"] for d in design_conditions_list)
    max_atoms = max(d["n_atoms"] for d in design_conditions_list)
    
    batch_size = len(batch)
    
    # Initialize batched tensors
    batched_design = {
        "length": max_tokens,
        "n_atoms": max_atoms,
        "batch_size": batch_size,
        "device": design_conditions_list[0].get("device", torch.device("cpu")),
        "dtype": design_conditions_list[0].get("dtype", torch.float32),
    }
    
    # Pad and stack token-level features (including sequence)
    token_features = ["asym_id", "residue_index", "entity_id", "token_index", "sym_id", "sequence"]
    for key in token_features:
        if key not in design_conditions_list[0]:
            continue
        padded = []
        for d in design_conditions_list:
            tensor = d[key]
            pad_len = max_tokens - tensor.shape[0]
            if pad_len > 0:
                # Pad sequence with N (unknown) = 4
                pad_value = 4 if key == "sequence" else 0
                tensor = F.pad(tensor, (0, pad_len), value=pad_value)
            padded.append(tensor)
        batched_design[key] = torch.stack(padded, dim=0)
    
    # Pad token bonds
    padded_bonds = []
    for d in design_conditions_list:
        bonds = d["token_bonds"]
        n = bonds.shape[0]
        if n < max_tokens:
            padded = F.pad(bonds, (0, max_tokens - n, 0, max_tokens - n), value=0)
        else:
            padded = bonds
        padded_bonds.append(padded)
    batched_design["token_bonds"] = torch.stack(padded_bonds, dim=0)
    
    # Pad atom_to_token_idx
    padded_mapping = []
    for d in design_conditions_list:
        mapping = d["atom_to_token_idx"]
        pad_len = max_atoms - mapping.shape[0]
        if pad_len > 0:
            mapping = F.pad(mapping, (0, pad_len), value=0)
        padded_mapping.append(mapping)
    batched_design["atom_to_token_idx"] = torch.stack(padded_mapping, dim=0)
    
    # Handle optional constraints
    if "ss_constraints" in design_conditions_list[0]:
        padded_ss = []
        for d in design_conditions_list:
            ss = d["ss_constraints"]
            n = ss.shape[0]
            if n < max_tokens:
                ss = F.pad(ss, (0, max_tokens - n, 0, max_tokens - n), value=2)  # pad with "any"
            padded_ss.append(ss)
        batched_design["ss_constraints"] = torch.stack(padded_ss, dim=0)
    
    if "distance_constraints" in design_conditions_list[0]:
        n_bins = design_conditions_list[0]["distance_constraints"].shape[-1]
        padded_dist = []
        for d in design_conditions_list:
            dist = d["distance_constraints"]
            n = dist.shape[0]
            if n < max_tokens:
                dist = F.pad(dist, (0, 0, 0, max_tokens - n, 0, max_tokens - n), value=0)
            padded_dist.append(dist)
        batched_design["distance_constraints"] = torch.stack(padded_dist, dim=0)
    
    # Handle pair compatibility matrix
    if "pair_compat" in design_conditions_list[0]:
        padded_compat = []
        for d in design_conditions_list:
            compat = d["pair_compat"]
            n = compat.shape[0]
            if n < max_tokens:
                compat = F.pad(compat, (0, max_tokens - n, 0, max_tokens - n), value=0)
            padded_compat.append(compat)
        batched_design["pair_compat"] = torch.stack(padded_compat, dim=0)
    
    # Batch labels
    padded_coords = []
    padded_masks = []
    for l in label_dict_list:
        coords = l["coordinate"]
        mask = l["coordinate_mask"]
        
        pad_len = max_atoms - coords.shape[0]
        if pad_len > 0:
            coords = F.pad(coords, (0, 0, 0, pad_len), value=0)
            mask = F.pad(mask, (0, pad_len), value=0)
        
        padded_coords.append(coords)
        padded_masks.append(mask)
    
    batched_labels = {
        "coordinate": torch.stack(padded_coords, dim=0),
        "coordinate_mask": torch.stack(padded_masks, dim=0),
    }
    
    return batched_design, batched_labels


def get_design_dataloaders(
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    use_pdb_directly: bool = False,
    pdb_file_pattern: str = "*.pdb",
    atom_selection: str = "c4prime",
    **dataset_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create DataLoaders for design training.
    
    Args:
        train_data_dir: Directory with training data.
        val_data_dir: Directory with validation data (optional).
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        use_pdb_directly: If True, load from PDB/CIF files directly.
                          If False, load from preprocessed .pt files.
        pdb_file_pattern: Glob pattern for PDB files (e.g., "*.pdb", "*.cif").
        atom_selection: Atom selection for PDB loading ("c4prime", "backbone", "all").
        **dataset_kwargs: Additional arguments for dataset.
        
    Returns:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (or None).
    """
    if use_pdb_directly:
        train_dataset = RNADesignDatasetFromPDB(
            pdb_dir=train_data_dir,
            file_pattern=pdb_file_pattern,
            atom_selection=atom_selection,
            **dataset_kwargs,
        )
    else:
        train_dataset = RNADesignDataset(train_data_dir, **dataset_kwargs)
    
    train_dataset.train()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_design_batch,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_data_dir is not None:
        if use_pdb_directly:
            val_dataset = RNADesignDatasetFromPDB(
                pdb_dir=val_data_dir,
                file_pattern=pdb_file_pattern,
                atom_selection=atom_selection,
                **dataset_kwargs,
            )
        else:
            val_dataset = RNADesignDataset(val_data_dir, **dataset_kwargs)
        
        val_dataset.eval()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_design_batch,
            pin_memory=True,
        )
    
    return train_loader, val_loader
