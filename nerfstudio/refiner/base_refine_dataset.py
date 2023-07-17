from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from nerfstudio.configs.base_config import InstantiateConfig

@dataclass
class RefineDatasetConfig(InstantiateConfig):
    
    _target: Type = field(default_factory=lambda: RefineDataset)


class RefineDataset:
    def __init__(self, config: RefineDatasetConfig):
        super().__init__()

    def collate(self, index):
        return None
