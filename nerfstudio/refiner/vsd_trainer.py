import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from jaxtyping import Float

import re
import os
import cv2
import gc
import tinycudann as tcnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.refiner.utils.if_utils import IF
from nerfstudio.refiner.utils.sd_utils import StableDiffusion
from nerfstudio.refiner.base_refine_trainer import RefineTrainer, RefineTrainerConfig

from nerfstudio.refiner.utils.vsd_prompt_processor import PromptProcessorOutput, PromptProcessor

# import threestudio
# from threestudio.models.prompt_processors.base import PromptProcessorOutput
# from threestudio.utils.base import BaseModule
# from threestudio.utils.misc import C, cleanup, parse_version
# from threestudio.utils.typing import *


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


def load_module_weights(
    path, module_name=None, ignore_modules=None, map_location=None
) -> Tuple[dict, int, int]:
    if module_name is not None and ignore_modules is not None:
        raise ValueError("module_name and ignore_modules cannot be both set")
    if map_location is None:
        raise ValueError

    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["state_dict"]
    state_dict_to_load = state_dict

    if ignore_modules is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            ignore = any(
                [k.startswith(ignore_module + ".") for ignore_module in ignore_modules]
            )
            if ignore:
                continue
            state_dict_to_load[k] = v

    if module_name is not None:
        state_dict_to_load = {}
        for k, v in state_dict.items():
            m = re.match(rf"^{module_name}\.(.*)$", k)
            if m is None:
                continue
            state_dict_to_load[m.group(1)] = v

    return state_dict_to_load, ckpt["epoch"], ckpt["global_step"]


def C(value: Any, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        # value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            raise ValueError("end_step must be an integer, got", end_step)
            # current_step = epoch
            # value = start_value + (end_value - start_value) * max(
            #     min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            # )
    return value


@dataclass
class VSDTrainerConfig(RefineTrainerConfig):

    _target: Type = field(default_factory=lambda: VSDTrainer)
    """target class to instantiate"""
    prompt: str = "living room with a lit furnace, couch and cozy curtains, bright lamps that make the room look well-lit"
    """Prompt for sds"""
    negative_prompt: str = "blurry, bad art, blurred, text, watermark, plant, nature"
    """Negative prompt for sds"""
    sds_save_dir: str = "vis_refiner/"
    """Directory to save sds visualization"""
    access_token: str = "none"
    """Access token for DeepFloyd huggingface model hub"""
    sds_start_step: int = 1000
    """Start step for sds training"""
    sds_end_step: int = -1
    """End step for sds training"""
    nerf_start_step: int = 0
    """Start step for nerf training"""
    nerf_end_step: int = -1
    """End step for nerf training"""
    lambda_vsd: float = 1.0
    """Lambda for vsd loss"""
    lambda_lora: float = 1.0
    """Lambda for lora loss"""
    save_state_dict: bool = True
    """Whether to save state dict"""

    weights: Optional[str] = None
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-1"
    enable_memory_efficient_attention: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_attention_slicing: bool = False
    enable_channels_last_format: bool = False

    guidance_scale: float = 7.5
    guidance_scale_lora: float = 1.0

    grad_clip: Optional[Any] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = True
    lora_cfg_training: bool = True
    lora_n_timestamp_samples: int = 1

    min_step_percent: float = 0.02
    max_step_percent: float = 0.98

    view_dependent_prompting: bool = True
    camera_condition_type: str = "extrinsics"


class VSDTrainer(RefineTrainer):

    def __init__(
        self,
        config,
        sds_iters, # sds iterations
        total_iters, # total iterations
        timestamp, # timestamp
        device: str = "cuda", # device
    ):
        super().__init__(config)
        self.config = config
        self.device = device
        self.time_stamp = timestamp

        self.sds_iters = sds_iters
        self.total_iters = total_iters

        self.guidance_images = None
        self.lora_images = None
        self.target_images = None
        self.workspace = os.path.join(config.sds_save_dir, self.time_stamp)
        self.save_guidance_path = os.path.join(self.workspace, 'guidance_vsd')
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.save_guidance_path, exist_ok=True)

        self.configure()
        if self.config.weights is not None:
            # format: path/to/weights:module_name
            weights_path, module_name = self.config.weights.split(":")
            state_dict, epoch, global_step = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.load_state_dict(state_dict)
            self.do_update_step(
                epoch, global_step, on_load_weights=True
            )  # restore states
        # dummy tensor to indicate model state
        self._dummy: Float[Tensor, "..."]
        self.register_buffer("_dummy", torch.zeros(0).float(), persistent=False)


    def configure(self) -> None:
        print(f"[INFO] Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.config.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline
        
        def cleanup():
            gc.collect()
            torch.cuda.empty_cache()
            tcnn.free_temporary_memory()

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        if (
            self.config.pretrained_model_name_or_path
            == self.config.pretrained_model_name_or_path_lora
        ):
            self.single_model = True
            pipe_lora = pipe
        else:
            self.single_model = False
            pipe_lora = StableDiffusionPipeline.from_pretrained(
                self.config.pretrained_model_name_or_path_lora,
                **pipe_lora_kwargs,
            ).to(self.device)
            del pipe_lora.vae
            cleanup()
            pipe_lora.vae = pipe.vae
        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        if self.config.enable_memory_efficient_attention:
            if torch.__version__[0] == '2':
                print(f"[INFO] PyTorch2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                print(f"[WARN] xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.config.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.config.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.config.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        ).to(self.device)
        self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).to(self.device)

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)  
        # alphas: shape: [1000], mono-decrease, 0.9991~0.0047

        self.grad_clip_val: Optional[float] = None

        self.prompt_utils: PromptProcessorOutput = PromptProcessor(
            prompt=self.config.prompt,
            pretrained_model_name_or_path=self.config.pretrained_model_name_or_path,
        ).process()

        print(f"[INFO] Loaded Stable Diffusion!")

    def get_param_groups(self):
        lora_params = []
        for param in self.parameters():
            if param.requires_grad:
                lora_params.append(param)
        return {
            "lora": lora_params
        }

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = self.prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.config.view_dependent_prompting,
        )
        cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.config.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    def sample_lora(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = self.prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.config.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.config.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.config.camera_condition_type}"
            )

        B = elevation.shape[0]
        camera_condition_cfg = torch.cat(
            [
                camera_condition.view(B, -1),
                torch.zeros_like(camera_condition.view(B, -1)),
            ],
            dim=0,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.config.guidance_scale_lora,
            class_labels=camera_condition_cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings_vd,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            # use view-independent text embeddings in LoRA
            noise_pred_est = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.config.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_text,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.config.guidance_scale_lora * (
            noise_pred_est_text - noise_pred_est_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)
        return grad

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.config.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.config.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings, _ = text_embeddings.chunk(2)
        if self.config.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings.repeat(
                self.config.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(B, -1).repeat(
                self.config.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    def train_step(self, step, outputs, data, **kwargs):
        B = 1
        H, W = data['H'], data['W']

        self.guidance_images = outputs['image']
        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        # if outputs['normal_image'] is not None:
        #     pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]

        as_latent = False
        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([pred_rgb, pred_mask], dim=1).contiguous() # [B, 4, H, W]

        losses = self.forward(
            rgb=pred_rgb,
            elevation=data['polar'],
            azimuth=data['azimuth'],
            camera_distances=data['radius'],
            mvp_mtx=None,
            c2w=data['c2w'],
            rgb_as_latents=as_latent,
            step=step,
        )
        loss_vsd, loss_lora = losses['loss_vsd'], losses['loss_lora']

        sds_loss_dict = {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
        }
        sds_outputs = {
            'sds_pred_rgb': pred_rgb, 
            'sds_pred_depth': pred_depth,
            'sds_pred_mask': pred_mask
        }
        sds_metrics = {
            "grad_norm": losses['grad_norm'],
            "min_step": losses['min_step'],
            "max_step": losses['max_step'],
        }

        return sds_loss_dict, sds_outputs, sds_metrics

    def forward(
        self,
        rgb: Float[Tensor, "B C H W"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]  # rgb: B3HW, [0, 1]
        latents = self.get_latents(rgb, rgb_as_latents=rgb_as_latents)

        # view-dependent text embeddings
        text_embeddings_vd = self.prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.config.view_dependent_prompting,
        )

        # input text embeddings, view-independent
        text_embeddings = self.prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.config.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.config.camera_condition_type == "mvp":
            raise NotImplementedError
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.config.camera_condition_type}"
            )

        grad = self.compute_grad_vsd(
            latents, text_embeddings_vd, text_embeddings, camera_condition
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss_vsd *= self.config.lambda_vsd

        loss_lora = self.train_lora(latents, text_embeddings, camera_condition)
        loss_lora *= self.config.lambda_lora

        return {
            "loss_vsd": loss_vsd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.config.grad_clip is not None:
            self.grad_clip_val = C(self.config.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.config.min_step_percent, epoch, global_step),
            max_step_percent=C(self.config.max_step_percent, epoch, global_step),
        )
    
    def save_guidance_images(self, step):
        img = self.guidance_images
        if img is None:
            return
        # tgt = self.target_images
        # if img is None or tgt is None:
        #     return
        img = img.contiguous().detach().cpu().numpy() * 255
        save_path = os.path.join(self.save_guidance_path, f'step_{step:07d}.png')
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # tgt = tgt.contiguous().detach().cpu().numpy() * 255
        # save_path = os.path.join(self.save_guidance_path, f'step_{step:07d}_tgt.png')
        # cv2.imwrite(save_path, cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR))