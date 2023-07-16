
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import numpy as np
from PIL import Image
import os

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig

from nerfstudio.inpainter.base_inpainter import Inpainter
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras, CameraType





@dataclass
class InitializerConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: Initializer)
    """target class to instantiate"""

class Initializer:
    """Model class for initializing the scene."""


    def __init__(self) -> None:
        super().__init__()


    def initialize_scene(self, **kwargs):
        return
        
