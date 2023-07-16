"""
Code for inpainting with diffusion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig

import torch
import tyro
from rich.progress import Console
from torch import nn

class Inpainter(nn.Module):
    """Model class for inpainting.

    Args:
        config: configuration for instantiating.
    """

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()
        self.kwargs = kwargs
    
    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for inpainting.

        Args:
            images: input images.
            mask: mask for inpainting.

        Returns:
            outputs: outputs from the inpainting model.
        """
        return {"pred_images": images}
