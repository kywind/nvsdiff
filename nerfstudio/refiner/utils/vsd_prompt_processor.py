import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from jaxtyping import Float

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModel
import gc
import tinycudann as tcnn


@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


def shift_azimuth_deg(azimuth: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    # shift azimuth angle (in degrees), to [-180, 180]
    return (azimuth + 180) % 360 - 180


def shifted_expotional_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def hash_prompt(model: str, prompt: str) -> str:
    import hashlib

    identifier = f"{model}-{prompt}"
    return hashlib.md5(identifier.encode()).hexdigest()


@dataclass
class PromptProcessorOutput:
    text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]

    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0)

    def get_text_embeddings_perp_neg(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Tuple[Float[Tensor, "BBBB N Nf"], Float[Tensor, "B 2"]]:
        assert (
            view_dependent_prompting
        ), "Perp-Neg only works with view-dependent prompting"

        batch_size = elevation.shape[0]

        direction_idx = torch.zeros_like(elevation, dtype=torch.long)
        for d in self.directions:
            direction_idx[
                d.condition(elevation, azimuth, camera_distances)
            ] = self.direction2idx[d.name]
        # 0 - side view
        # 1 - front view
        # 2 - back view
        # 3 - overhead view

        pos_text_embeddings = []
        neg_text_embeddings = []
        neg_guidance_weights = []
        uncond_text_embeddings = []

        side_emb = self.text_embeddings_vd[0]
        front_emb = self.text_embeddings_vd[1]
        back_emb = self.text_embeddings_vd[2]
        overhead_emb = self.text_embeddings_vd[3]

        for idx, ele, azi, dis in zip(
            direction_idx, elevation, azimuth, camera_distances
        ):
            azi = shift_azimuth_deg(azi)  # to (-180, 180)
            uncond_text_embeddings.append(
                self.uncond_text_embeddings_vd[idx]
            )  # should be ""
            if idx.item() == 3:  # overhead view
                pos_text_embeddings.append(overhead_emb)  # side view
                # dummy
                neg_text_embeddings += [
                    self.uncond_text_embeddings_vd[idx],
                    self.uncond_text_embeddings_vd[idx],
                ]
                neg_guidance_weights += [0.0, 0.0]
            else:  # interpolating views
                if torch.abs(azi) < 90:
                    # front-side interpolation
                    # 0 - complete side, 1 - complete front
                    r_inter = 1 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * front_emb + (1 - r_inter) * side_emb
                    )
                    neg_text_embeddings += [front_emb, side_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_fs, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_sf, 1 - r_inter),
                    ]
                else:
                    # side-back interpolation
                    # 0 - complete back, 1 - complete side
                    r_inter = 2.0 - torch.abs(azi) / 90
                    pos_text_embeddings.append(
                        r_inter * side_emb + (1 - r_inter) * back_emb
                    )
                    neg_text_embeddings += [side_emb, front_emb]
                    neg_guidance_weights += [
                        -shifted_expotional_decay(*self.perp_neg_f_sb, r_inter),
                        -shifted_expotional_decay(*self.perp_neg_f_fsb, r_inter),
                    ]

        text_embeddings = torch.cat(
            [
                torch.stack(pos_text_embeddings, dim=0),
                torch.stack(uncond_text_embeddings, dim=0),
                torch.stack(neg_text_embeddings, dim=0),
            ],
            dim=0,
        )

        return text_embeddings, torch.as_tensor(
            neg_guidance_weights, device=elevation.device
        ).reshape(batch_size, 2)



class PromptProcessor:

    def __init__(self, prompt, negative_prompt="",
        pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base", device="cuda"):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device

        self.overhead_threshold: float = 60.0
        self.front_threshold: float = 45.0
        self.back_threshold: float = 45.0
        self.view_dependent_prompt_front: bool = False

        # view-dependent text embeddings
        self.directions = [
            DirectionConfig(
                "side",
                lambda s: f"{s}, side view",
                lambda s: s,
                lambda ele, azi, dis: torch.ones_like(ele, dtype=torch.bool),
            ),
            DirectionConfig(
                "front",
                lambda s: f"{s}, front view",
                lambda s: s,
                lambda ele, azi, dis: (
                    shift_azimuth_deg(azi) > -self.front_threshold
                )
                & (shift_azimuth_deg(azi) < self.front_threshold),
            ),
            DirectionConfig(
                "back",
                lambda s: f"{s}, back view",
                lambda s: s,
                lambda ele, azi, dis: (
                    shift_azimuth_deg(azi) > 180 - self.back_threshold
                )
                | (shift_azimuth_deg(azi) < -180 + self.back_threshold),
            ),
            DirectionConfig(
                "overhead",
                lambda s: f"{s}, overhead view",
                lambda s: s,
                lambda ele, azi, dis: ele > self.overhead_threshold,
            ),
        ]
        self.direction2idx = {d.name: i for i, d in enumerate(self.directions)}
        # self.prompts_vd = [
        #     d.prompt(self.cfg.get(f"prompt_{d.name}", None) or self.prompt)  # type: ignore
        #     for d in self.directions
        # ]  # view-dependent custom prompts
        self.prompts_vd = [d.prompt(self.prompt) for d in self.directions]
        self.negative_prompts_vd = [d.negative_prompt(self.negative_prompt) for d in self.directions]

        print(f"VSD trainer using prompt [{self.prompt}] and negative prompt [{self.negative_prompt}]")
        prompts_vd_display = " ".join([f"[{d.name}]:[{prompt}]" for prompt, d in zip(self.prompts_vd, self.directions)])
        print(f"Using view-dependent prompts {prompts_vd_display}")

        ## encode prompts
        all_prompts = [self.prompt] + [self.negative_prompt] + self.prompts_vd + self.negative_prompts_vd

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder", device_map="auto",
        )

        with torch.no_grad():
            tokens = tokenizer(
                all_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(tokens.input_ids)[0]

        del text_encoder

        embeddings = {}
        for prompt, embedding in zip(all_prompts, text_embeddings):
            embeddings[prompt] = embedding
    
        self.text_embeddings = embeddings[self.prompt][None, ...].to(self.device)
        self.uncond_text_embeddings = embeddings[self.negative_prompt][None, ...].to(self.device)
        self.text_embeddings_vd = torch.stack([embeddings[prompt] for prompt in self.prompts_vd], dim=0).to(self.device)
        self.uncond_text_embeddings_vd = torch.stack([embeddings[prompt] for prompt in self.negative_prompts_vd], dim=0).to(self.device)
        print(f"Loaded text embeddings.")

        cleanup()


    def process(self) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            uncond_text_embeddings=self.uncond_text_embeddings,
            text_embeddings_vd=self.text_embeddings_vd,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=False,
            perp_neg_f_sb=(1, 0.5, -0.606), # a*e(-b*r) + c, a * e(-b) + c = 0
            perp_neg_f_fsb=(1, 0.5, +0.967),
            perp_neg_f_fs=(4, 0.5, -2.426), # f_fs(1) = 0, a, b > 0
            perp_neg_f_sf=(4, 0.5, -2.426),
        )

