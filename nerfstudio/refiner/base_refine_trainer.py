from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from nerfstudio.configs.base_config import InstantiateConfig

@dataclass
class RefineTrainerConfig(InstantiateConfig):
    
    _target: Type = field(default_factory=lambda: RefineTrainer)


class RefineTrainer(nn.Module):
    def __init__(self, config: RefineTrainerConfig):
        super().__init__()

    def train_step(self, step, data):
        return None

    def get_param_groups(self):  # no trainable parameters by default
        return {}
