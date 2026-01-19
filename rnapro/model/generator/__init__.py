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
__init__.py for generator subpackage.

Re-exports diffusion functions from the parent generator.py module.
"""

# Import from the sibling generator.py file
# This allows `from rnapro.model.generator import ...` to work
import sys
import importlib.util
from pathlib import Path

# Load the generator.py module from the parent directory
_generator_path = Path(__file__).parent.parent / "generator.py"
_spec = importlib.util.spec_from_file_location("_generator_module", _generator_path)
_generator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_generator_module)

# Re-export the classes and functions
TrainingNoiseSampler = _generator_module.TrainingNoiseSampler
InferenceNoiseScheduler = _generator_module.InferenceNoiseScheduler
sample_diffusion = _generator_module.sample_diffusion
sample_diffusion_training = _generator_module.sample_diffusion_training

__all__ = [
    "TrainingNoiseSampler",
    "InferenceNoiseScheduler", 
    "sample_diffusion",
    "sample_diffusion_training",
]
