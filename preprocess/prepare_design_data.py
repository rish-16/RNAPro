# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocessing script to convert PDB/CIF files to .pt format for RNAProDesign training.

Usage:
    python preprocess/prepare_design_data.py \
        --input_dir ./raw_structures \
        --output_dir ./data/train \
        --atom_level c4prime \
        --file_format pdb

This script extracts C4' coordinates from RNA structures and saves them
as PyTorch tensors ready for training.
"""

import argparse
import glob
import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# RNA residue names to recognize
RNA_RESIDUES = {
    "A", "U", "G", "C",  # Standard
    "DA", "DU", "DG", "DC",  # DNA (sometimes in RNA structures)
    "ADE", "URA", "GUA", "CYT",  # Alternative names
    "RA", "RU", "RG", "RC",  # Ribosomal
    "A", "U", "G", "C", "I",  # Including inosine
}

# Atom names for C4' selection
C4_PRIME_ATOMS = {"C4'", "C4*"}


def parse_pdb_c4prime(filepath: str) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Parse PDB file and extract C4' coordinates.
    
    Args:
        filepath: Path to PDB file.
        
    Returns:
        coordinates: [N_residues, 3] C4' coordinates.
        n_residues: Number of residues.
        chain_ids: List of chain IDs.
    """
    coordinates = []
    residue_info = []  # (chain_id, res_id) to track unique residues
    
    with open(filepath, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22].strip() or "A"
            
            try:
                res_id = int(line[22:26].strip())
            except ValueError:
                continue
            
            # Check if RNA residue
            if res_name not in RNA_RESIDUES:
                continue
            
            # Check if C4' atom
            if atom_name not in C4_PRIME_ATOMS:
                continue
            
            # Check for duplicate residues (keep first occurrence)
            res_key = (chain_id, res_id)
            if res_key in residue_info:
                continue
            
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coordinates.append([x, y, z])
                residue_info.append(res_key)
            except ValueError:
                continue
    
    if len(coordinates) == 0:
        raise ValueError(f"No C4' atoms found in {filepath}")
    
    chain_ids = list(set(r[0] for r in residue_info))
    return torch.tensor(coordinates, dtype=torch.float32), len(coordinates), chain_ids


def parse_cif_c4prime(filepath: str) -> Tuple[torch.Tensor, int, List[str]]:
    """
    Parse mmCIF file and extract C4' coordinates.
    
    Args:
        filepath: Path to CIF file.
        
    Returns:
        coordinates: [N_residues, 3] C4' coordinates.
        n_residues: Number of residues.
        chain_ids: List of chain IDs.
    """
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
    except ImportError:
        raise ImportError("BioPython is required for CIF parsing. Install with: pip install biopython")
    
    structure = parser.get_structure("rna", filepath)
    
    coordinates = []
    residue_info = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                
                # Check if RNA residue
                if res_name not in RNA_RESIDUES:
                    continue
                
                # Look for C4' atom
                for atom in residue:
                    if atom.get_name() in C4_PRIME_ATOMS:
                        coord = atom.get_coord()
                        coordinates.append(coord.tolist())
                        residue_info.append((chain.id, residue.id[1]))
                        break  # Only one C4' per residue
    
    if len(coordinates) == 0:
        raise ValueError(f"No C4' atoms found in {filepath}")
    
    chain_ids = list(set(r[0] for r in residue_info))
    return torch.tensor(coordinates, dtype=torch.float32), len(coordinates), chain_ids


def center_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Center coordinates at origin."""
    center = coords.mean(dim=0)
    return coords - center


def process_structure(
    filepath: str,
    center: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Process a single structure file.
    
    Args:
        filepath: Path to PDB/CIF file.
        center: Whether to center coordinates at origin.
        
    Returns:
        Dictionary with processed data.
    """
    ext = Path(filepath).suffix.lower()
    
    if ext in [".pdb", ".ent"]:
        coords, n_residues, chains = parse_pdb_c4prime(filepath)
    elif ext in [".cif", ".mmcif"]:
        coords, n_residues, chains = parse_cif_c4prime(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    if center:
        coords = center_coordinates(coords)
    
    # Create output dictionary
    data = {
        "coordinates": coords,  # [N, 3]
        "coordinate_mask": torch.ones(n_residues),  # [N]
        "n_tokens": n_residues,
        "n_atoms": n_residues,  # Same for C4' level
        "n_chains": len(chains),
    }
    
    return data


def process_directory(
    input_dir: str,
    output_dir: str,
    file_patterns: List[str] = ["*.pdb", "*.cif"],
    center: bool = True,
    min_length: int = 5,
    max_length: int = 1024,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Process all structure files in a directory.
    
    Args:
        input_dir: Input directory with PDB/CIF files.
        output_dir: Output directory for .pt files.
        file_patterns: Glob patterns for input files.
        center: Whether to center coordinates.
        min_length: Minimum sequence length.
        max_length: Maximum sequence length.
        verbose: Print progress.
        
    Returns:
        Statistics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all input files
    input_files = []
    for pattern in file_patterns:
        input_files.extend(glob.glob(os.path.join(input_dir, "**", pattern), recursive=True))
    
    input_files = sorted(set(input_files))
    logger.info(f"Found {len(input_files)} structure files in {input_dir}")
    
    stats = {
        "total": len(input_files),
        "processed": 0,
        "skipped_length": 0,
        "skipped_error": 0,
    }
    
    for i, filepath in enumerate(input_files):
        if verbose and i % 100 == 0:
            logger.info(f"Processing {i}/{len(input_files)}...")
        
        try:
            data = process_structure(filepath, center=center)
            n_residues = data["n_tokens"]
            
            # Check length
            if n_residues < min_length or n_residues > max_length:
                stats["skipped_length"] += 1
                continue
            
            # Generate output filename
            basename = Path(filepath).stem
            output_filename = f"{basename}_{n_residues}.pt"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save
            torch.save(data, output_path)
            stats["processed"] += 1
            
        except Exception as e:
            if verbose:
                logger.warning(f"Error processing {filepath}: {e}")
            stats["skipped_error"] += 1
    
    logger.info(f"Processing complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDB/CIF files to .pt format for RNAProDesign training."
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Input directory containing PDB/CIF files.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for .pt files.",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default="both",
        choices=["pdb", "cif", "both"],
        help="Input file format to process.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum sequence length to include.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length to include.",
    )
    parser.add_argument(
        "--no_center",
        action="store_true",
        help="Don't center coordinates at origin.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    
    args = parser.parse_args()
    
    # Set file patterns
    if args.file_format == "pdb":
        patterns = ["*.pdb", "*.PDB", "*.ent", "*.ENT"]
    elif args.file_format == "cif":
        patterns = ["*.cif", "*.CIF", "*.mmcif", "*.mmCIF"]
    else:
        patterns = ["*.pdb", "*.PDB", "*.ent", "*.cif", "*.CIF", "*.mmcif"]
    
    stats = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_patterns=patterns,
        center=not args.no_center,
        min_length=args.min_length,
        max_length=args.max_length,
        verbose=not args.quiet,
    )
    
    print(f"\n{'='*50}")
    print("Processing Summary:")
    print(f"  Total files found: {stats['total']}")
    print(f"  Successfully processed: {stats['processed']}")
    print(f"  Skipped (length): {stats['skipped_length']}")
    print(f"  Skipped (error): {stats['skipped_error']}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
