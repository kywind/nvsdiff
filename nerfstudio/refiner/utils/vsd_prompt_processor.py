import random
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from jaxtyping import Float
from torch import Tensor


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

