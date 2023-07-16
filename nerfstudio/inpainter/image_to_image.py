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
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline
# from diffusers.utils import pt_to_pil
# from huggingface_hub import login
# login()

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


class ImageToImageInpainter(Inpainter):
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
        self.model_type = 'stable_diffusion'

        if self.model_type == 'stable_diffusion':
            self.model = StableDiffusionImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
            self.model.to(self.device)
        
        elif self.model_type == 'floyd':

            stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
            stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
            stage_1.enable_model_cpu_offload()
            self.stage_1 = stage_1

            stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
            )
            stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
            stage_2.enable_model_cpu_offload()
            self.stage_2 = stage_2

            safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
            stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
            stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
            stage_3.enable_model_cpu_offload()
            self.stage_3 = stage_3

        else:
            raise NotImplementedError


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

        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        else:
            prompt = "Real estate photo"

        if self.model_type == 'stable_diffusion':
            image = (image + 1.) * 127.5
            image = image.detach().cpu().numpy().astype(np.uint8)
            image = Image.fromarray(image).resize((512, 512))

            # import ipdb; ipdb.set_trace()
            # mask = mask.detach().cpu().numpy()[..., 0]
            # mask = np.floor(mask) * 255
            # mask = mask.astype(np.uint8)
            # mask = Image.fromarray(mask).resize((512, 512))
            # mask.save(f"temp/vis-temp/mask_{step}.png")

            inpaint_image = self.model(
                prompt=prompt, 
                image=image, 
                strength=0.5,
                guidance_scale=7.5,  # default value
            ).images[0]
            # inpaint_image = torch.from_numpy(np.array(inpaint_image)).to(self.device).permute(2, 0, 1).float() / 127.5 - 1.
        
        elif self.model_type == 'floyd':

            prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)

            generator = torch.manual_seed(0)

            image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
            # pt_to_pil(image)[0].save("./if_stage_I.png")

            # stage 2
            image = self.stage_2(
                image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
            ).images
            # pt_to_pil(image)[0].save("./if_stage_II.png")

            # stage 3
            image = self.stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
            # image[0].save("./if_stage_III.png")
            import ipdb; ipdb.set_trace()

            inpaint_image = image[0]

        return inpaint_image
