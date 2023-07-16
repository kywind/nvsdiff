import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms.functional as TF
from torchmetrics import PearsonCorrCoef

from rich.console import Console
from torch_ema import ExponentialMovingAverage

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.initializer.sds.utils.if_utils import IF
from nerfstudio.initializer.sds.utils.sd_utils import StableDiffusion
# from nerfstudio.initializer.sds.utils.utils import get_CPU_mem, get_GPU_mem
# from nerfstudio.initializer.sds.utils.optimizer import Adan

"""
ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(ray_samples, step, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs
"""

@dataclass
class SDSTrainerConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SDSTrainer)
    """target class to instantiate"""
    fp16: bool = True
    """Whether to use fp16"""
    sds_save_dir: str = "sds_vis/"
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
    guidance_scale: float = 20
    """Guidance scale"""
    lambda_guidance: float = 0.5
    """Guidance loss weight"""
    lambda_opacity: float = 0
    """Opacity loss weight"""
    lambda_entropy: float = 0.1
    """Entropy loss weight"""
    lambda_orient: float = 0
    """Orientation loss weight"""
    lambda_2d_normal_smooth: float = 0
    """2D normal smooth loss weight"""
    lambda_3d_normal_smooth: float = 0
    """3D normal smooth loss weight"""
    guidance_method: str = "IF"
    """Guidance methods"""
    latent_iter_ratio: float = 0
    """Ratio of iterations to use latent guidance"""

class SDSTrainer:
    def __init__(
        self,
        config,
        model, # network
        sds_iters, # sds iterations
        total_iters, # total iterations
        timestamp, # timestamp
        device: str = "cuda", # device
    ):
        # self.config = config
        self.device = device
        self.sds_iters = sds_iters
        self.total_iters = total_iters
        self.latent_iter_ratio = config.latent_iter_ratio
        self.fp16 = config.fp16
        self.time_stamp = timestamp
        self.guidance_method = config.guidance_method
        self.workspace = os.path.join(config.sds_save_dir, self.time_stamp)
        self.save_guidance_path = os.path.join(self.workspace, 'guidance')
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.save_guidance_path, exist_ok=True)

        self.prompt = "living room with a lit furnace, couch and cozy curtains, bright lamps that make the room look well-lit"
        self.negative_prompt = "blurry, bad art, blurred, text, watermark, plant, nature"

        # model.to(device)
        # if self.world_size > 1:
        #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        guidance = nn.ModuleDict()
        if self.guidance_method == 'IF':
            guidance = IF(device, vram_O=True, t_range=[0.02, 0.5], access_token=config.access_token)
        elif self.guidance_method == 'SD':
            guidance = StableDiffusion(device, vram_O=True, t_range=[0.02, 0.5], access_token=config.access_token)

        self.guidance = guidance
        for p in self.guidance.parameters():
            p.requires_grad = False

        self.embeddings = {}
        self.prepare_embeddings()

        # self.pearson = PearsonCorrCoef().to(self.device)

        # self.optimizer = Adan(model.get_params(lr=5e-3), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        # self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        # self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 0.1 ** min(iter / self.iters, 1))

        # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.95)
        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        # self.bg_radius = -1
        self.guidance_scale = config.guidance_scale
        self.lambda_guidance = config.lambda_guidance
        self.lambda_opacity = config.lambda_opacity
        self.lambda_entropy = config.lambda_entropy
        self.lambda_orient = config.lambda_orient
        self.lambda_2d_normal_smooth = config.lambda_2d_normal_smooth
        self.lambda_3d_normal_smooth = config.lambda_3d_normal_smooth

        self.guidance_images = None
        self.target_images = None

        print(f'[INFO] Trainer: {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        print(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')


    @torch.no_grad()
    def prepare_embeddings(self):
        # text embeddings (stable-diffusion)
        if self.guidance_method == 'SD':
            self.embeddings['default'] = self.guidance.get_text_embeds([self.prompt])
            self.embeddings['uncond'] = self.guidance.get_text_embeds([self.negative_prompt])
            for d in ['front', 'side', 'back']:
                self.embeddings[d] = self.guidance.get_text_embeds([f"{self.prompt}, {d} view"])

        elif self.guidance_method == 'IF':
            self.embeddings['default'] = self.guidance.get_text_embeds([self.prompt])
            self.embeddings['uncond'] = self.guidance.get_text_embeds([self.negative_prompt])
            for d in ['front', 'side', 'back']:
                self.embeddings[d] = self.guidance.get_text_embeds([f"{self.prompt}, {d} view"])

        elif self.guidance_method == 'CLIP':
            raise NotImplementedError
            self.embeddings['text'] = self.guidance.get_text_embeds(self.prompt)
        
        else:
            raise ValueError(f'Guidance method {self.guidance_method} not supported.')


    def train_step(self, step, data, save_guidance_path=None):
        """
            Args:
                save_guidance_path: an image that combines the NeRF render, the added latent noise,
                    the denoised result and optionally the fully-denoised image.
        """
        # rays_o = data['rays_o'] # [B, N, 3]
        # rays_d = data['rays_d'] # [B, N, 3]
        # mvp = data['mvp'] # [B, 4, 4]
        ray_bundle = data['ray_bundle']

        # B, N = rays_o.shape[:2]
        # B = ray_bundle.camera_indices.shape[0]
        B = 1
        H, W = data['H'], data['W']

        # random shading
        # ambient_ratio = 0.1 + 0.9 * random.random()
        # rand = random.random()
        # if rand > 0.8:
        #     shading = 'textureless'
        # else:
        #     shading = 'lambertian'

        # random weights binarization (like mobile-nerf) [NOT WORKING NOW]
        # binarize_thresh = min(0.5, -0.5 + step / self.iters)
        # binarize = random.random() < binarize_thresh
        # binarize = False

        # random background
        # rand = random.random()
        # if self.bg_radius > 0 and rand > 0.5:
        #     bg_color = None # use bg_net
        # else:
        #     bg_color = torch.rand(3).to(self.device) # single color random bg

        # TODO
        model_outputs = self.model(ray_bundle, step)
        outputs = {
            'image': model_outputs['rgb'].reshape(H, W, 3),
            'depth': model_outputs['depth'].reshape(H, W),
            'weights_sum': model_outputs['accumulation'].reshape(H, W),
            'weights': model_outputs['weights_list'][-1].reshape(H, W, -1),
            'normal_image': model_outputs['pred_normals'].reshape(H, W, 3) if 'pred_normals' in model_outputs else None,
        }

        # visualize
        # if step % 100 == 0:
        #     img = outputs['image'][0].permute(1, 2, 0).cpu().numpy()
        #     img = np.clip(img, 0, 1)
        #     img = (img * 255).astype(np.uint8)
        #     img = Image.fromarray(img)
        #     img.save(os.path.join(, f'{self.time_stamp}_{step}.png'))
        self.guidance_images = outputs['image']

        # outputs = self.model.render(rays_o, rays_d, mvp, H, W, staged=False, perturb=True, 
        #     bg_color=bg_color, ambient_ratio=ambient_ratio, shading=shading, binarize=binarize)

        pred_depth = outputs['depth'].reshape(B, 1, H, W)
        pred_mask = outputs['weights_sum'].reshape(B, 1, H, W)
        if outputs['normal_image'] is not None:
            pred_normal = outputs['normal_image'].reshape(B, H, W, 3)

        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]

        as_latent = self.guidance_method == 'SD' and step * 1. / self.total_iters < self.latent_iter_ratio
        if as_latent:
            # abuse normal & mask as latent code for faster geometry initialization (ref: fantasia3D)
            pred_rgb = torch.cat([pred_rgb, pred_mask], dim=1).contiguous() # [B, 4, H, W]

        # novel view loss
        loss = 0
        # interpolate text_z
        azimuth = data['azimuth'] # [-180, 180]
        # ENHANCE: remove loop to handle batch size > 1
        text_z = [self.embeddings['uncond']] * azimuth.shape[0]
        for b in range(azimuth.shape[0]):
            if azimuth[b] >= -90 and azimuth[b] < 90:
                if azimuth[b] >= 0:
                    r = 1 - azimuth[b] / 90
                else:
                    r = 1 + azimuth[b] / 90
                start_z = self.embeddings['front']
                end_z = self.embeddings['side']
            else:
                if azimuth[b] >= 0:
                    r = 1 - (azimuth[b] - 90) / 90
                else:
                    r = 1 + (azimuth[b] + 90) / 90
                start_z = self.embeddings['side']
                end_z = self.embeddings['back']
            text_z.append(r * start_z + (1 - r) * end_z)
        text_z = torch.cat(text_z, dim=0)

        # if step >= 45000:  # TODO remove hardcode
        #     min_t = 0.02
        #     max_t = 0.2
        # else:
        min_t = None
        max_t = None

        guidance_loss, target_image = self.guidance.train_step(text_z, pred_rgb, 
            guidance_scale=self.guidance_scale, grad_scale=self.lambda_guidance, as_latent=as_latent,
            min_t=min_t, max_t=max_t)

        loss = loss + guidance_loss
        self.target_images = target_image[0].permute(1, 2, 0)
        sds_metrics = {'sds_loss_guidance': guidance_loss}

        # regularizations
        if self.lambda_opacity > 0:
            loss_opacity = (outputs['weights_sum'] ** 2).mean()
            loss = loss + self.lambda_opacity * loss_opacity
            sds_metrics['sds_loss_opacity'] = self.lambda_opacity * loss_opacity

        if self.lambda_entropy > 0:
            alphas = outputs['weights'].clamp(1e-5, 1 - 1e-5)
            # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            lambda_entropy = self.lambda_entropy * min(1, 2 * step / self.sds_iters)
            loss = loss + lambda_entropy * loss_entropy
            sds_metrics['sds_lambda_entropy'] = lambda_entropy
            sds_metrics['sds_loss_entropy'] = lambda_entropy * loss_entropy

        if self.lambda_2d_normal_smooth > 0 and 'normal_image' in outputs:
            # pred_vals = outputs['normal_image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            # smoothed_vals = TF.gaussian_blur(pred_vals.detach(), kernel_size=9)
            # loss_smooth = F.mse_loss(pred_vals, smoothed_vals)
            # total-variation
            loss_smooth = (pred_normal[:, 1:, :, :] - pred_normal[:, :-1, :, :]).square().mean() + \
                          (pred_normal[:, :, 1:, :] - pred_normal[:, :, :-1, :]).square().mean()
            loss = loss + self.lambda_2d_normal_smooth * loss_smooth
            sds_metrics['sds_loss_smooth'] = self.lambda_2d_normal_smooth * loss_smooth

        if self.lambda_orient > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.lambda_orient * loss_orient
            sds_metrics['sds_loss_orient'] = loss_orient

        if self.lambda_3d_normal_smooth > 0 and 'loss_normal_perturb' in outputs:
            loss_normal_perturb = outputs['loss_normal_perturb']
            loss = loss + self.lambda_3d_normal_smooth * loss_normal_perturb
            sds_metrics['sds_loss_normal_perturb'] = self.lambda_3d_normal_smooth * loss_normal_perturb
        
        # return pred_rgb, pred_depth, loss
        
        sds_loss_dict = {'sds_loss': loss}
        sds_outputs = {'sds_pred_rgb': pred_rgb, 'sds_pred_depth': pred_depth, 'sds_pred_mask': pred_mask}

        return sds_loss_dict, sds_outputs, sds_metrics


    def save_guidance_images(self, step):
        img = self.guidance_images
        tgt = self.target_images
        if img is None or tgt is None:
            return
        img = img.contiguous().detach().cpu().numpy() * 255
        save_path = os.path.join(self.save_guidance_path, f'step_{step:07d}.png')
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        tgt = tgt.contiguous().detach().cpu().numpy() * 255
        save_path = os.path.join(self.save_guidance_path, f'step_{step:07d}_tgt.png')
        cv2.imwrite(save_path, cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR))

