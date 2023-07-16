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
from torchtyping import TensorType
from nerfstudio.inpainter.base_inpainter import Inpainter
from diffusers import StableDiffusionInpaintPipeline

import kornia
from kornia.geometry.depth import depth_to_3d, project_points, DepthWarper
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import compose_transformations, \
        convert_points_to_homogeneous, inverse_transformation, transform_points

from PIL import Image
import cv2
import numpy as np


def concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def blend(img, img_ori, mask):
    mask = np.array(mask)
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    img = np.array(img) * (mask / 255) + np.array(img_ori) * (1 - mask / 255)
    img = Image.fromarray(img.astype(np.uint8))
    return img


class StableDiffusionInpainter(Inpainter):
    """Model class for inpainting.

    Args:
        config: configuration for instantiating.
    """

    def __init__(
        self,
        device: str = "cuda",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.camera_selector = None
        self.device = device

        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
        self.model.to(self.device)


    def forward(
        self,
        image: TensorType["hw":..., "rgb":3],
        mask: TensorType["hw":..., 1],
        step: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for inpainting.

        Args:
            images: input images.
            mask: mask for inpainting.

        Returns:
            outputs: outputs from the inpainting model.
        """

        image = (image + 1.) * 127.5
        image = image.detach().cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image).resize((512, 512))

        # import ipdb; ipdb.set_trace()
        mask = mask.detach().cpu().numpy()[..., 0]
        mask = np.floor(mask) * 255
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).resize((512, 512))
        # mask.save(f"temp/vis-temp/mask_{step}.png")

        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        else:
            prompt = "Real estate photo"

        inpaint_image = self.model(prompt=prompt, image=image, mask_image=mask).images[0]
        # inpaint_image = torch.from_numpy(np.array(inpaint_image)).to(self.device).permute(2, 0, 1).float() / 127.5 - 1.

        return inpaint_image
