# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Loss functions for RNAPro models.

Re-exports loss classes from the parent loss.py module.
"""

# Import from the sibling loss.py file
import importlib.util
from pathlib import Path

# Load the loss.py module from the parent directory
_loss_path = Path(__file__).parent.parent / "loss.py"
_spec = importlib.util.spec_from_file_location("_loss_module", _loss_path)
_loss_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss_module)

# Re-export the classes
SmoothLDDTLoss = _loss_module.SmoothLDDTLoss

__all__ = [
    "SmoothLDDTLoss",
]
