#!/usr/bin/env python3

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

import os
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert template submission CSV to .pt file usable by RNAPro."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to template submission CSV (columns: ID, resname, x_1..x_5, y_1..y_5, z_1..z_5).",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=6,
        help="Number of coordinate triplets + 1 (default: 6 to read x_1..x_5 etc.).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="template_features.pt",
        help="Name of the output .pt file (default: template_features.pt).",
    )
    return parser.parse_args()


def build_structure_data(
    df: pd.DataFrame, max_n: int = 6
) -> Dict[str, Dict[str, Any]]:
    """
    Build structure_data dict from a dataframe with columns:
      - ID: e.g., R1107_1 (target_id and residue index separated by '_')
      - resname: residue name (A/U/G/C)
      - x_i, y_i, z_i for i in [1..max_n-1]
    Produces:
      structure_data[target_id] = {
        "seq": str of residue letters,
        "xyz": np.ndarray of shape (L, max_n-1, 3)
      }
    Args:
        df (pd.DataFrame): DataFrame with columns: ID, resname, x_1..x_5, y_1..y_5, z_1..z_5
        max_n (int): Number of coordinate triplets + 1 (default: 6 to read x_1..x_5 etc.)

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with target_id as keys and seq and xyz as values
    """
    # Fill missing numeric values, to avoid NaNs in tensors
    df = df.copy()
    df.fillna(0.0, inplace=True)

    structure_data: Dict[str, Dict[str, Any]] = {}

    # Iterate rows to aggregate per target_id
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting templates"):
        full_id = row["ID"]
        resname = row["resname"]
        # target_id is everything before the trailing underscore suffix
        target_id = "_".join(str(full_id).split("_")[:-1])

        if target_id not in structure_data:
            structure_data[target_id] = {"seq": "", "xyz": []}

        structure_data[target_id]["seq"] += str(resname)

        # Collect up to (max_n - 1) template coordinates per residue
        xyzs = []
        for i in range(1, max_n):
            x, y, z = row[f"x_{i}"], row[f"y_{i}"], row[f"z_{i}"]
            xyzs.append([float(x), float(y), float(z)])

        structure_data[target_id]["xyz"].append(xyzs)

    # Convert lists to numpy arrays
    for k in structure_data.keys():
        structure_data[k]["xyz"] = np.array(structure_data[k]["xyz"], dtype=np.float32)

    return structure_data


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    structure_data = build_structure_data(df=df, max_n=args.max_n)

    os.makedirs("data/processed", exist_ok=True)
    file_path = os.path.join("data/processed", args.output_name)


    torch.save(structure_data, file_path)
    print(f"Saved template features to: {file_path}")
    
if __name__ == "__main__":
    main()


