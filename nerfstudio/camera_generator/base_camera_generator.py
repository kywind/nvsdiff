
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

from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras, CameraType





@dataclass
class CameraGeneratorConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: CameraGenerator)
    """target class to instantiate"""
    pending_camera_only: bool = True
    """only use pending cameras for inpainting"""
    use_pending_images: bool = True
    """use pending images for inpainting. Only valid when pending_camera_only is True"""



class CameraGenerator:
    """Model class for generating camera."""


    def __init__(self, config: CameraGeneratorConfig, inpaint_save_dir: str = None) -> None:
        super().__init__()
        self.config = config
        self.camera_count = 0
        self.inpaint_save_dir = inpaint_save_dir
        if self.inpaint_save_dir is not None:
            os.makedirs(self.inpaint_save_dir, exist_ok=True)
        

    def generate_inpaint_cameras(self, step: int, num_inpaint_images: int,
            datamanager: DataManager, model: Model, inpainter: Inpainter):

        train_dataset = datamanager.train_dataset

        if self.config.pending_camera_only:
            self.add_existing_camera_views(step, num_inpaint_images, train_dataset)
        else:
            self.add_new_camera_views(step, num_inpaint_images, train_dataset, model, inpainter)

        cameras = train_dataset.cameras.to(model.device)
        for index in range(len(train_dataset.image_filenames)):
            if train_dataset.image_filenames[index] is not None:  # skip existing images
                continue

            camera_ray_bundle = cameras.generate_rays(camera_indices=index)

            outputs = model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            image, mask = model.get_novel_view_rendering(outputs)
            inpaint_image = inpainter(image, mask, step=step)

            save_path = os.path.join(self.inpaint_save_dir, f'frame_{step:04d}_{index:04d}_inpaint.png')
            inpaint_image.save(save_path)

            render_image = image.cpu().numpy()
            render_image = (render_image * 255).astype(np.uint8)
            save_path = os.path.join(self.inpaint_save_dir, f'frame_{step:04d}_{index:04d}.png')
            Image.fromarray(render_image).save(save_path)

            train_dataset.update_image(index, save_path)
        datamanager.setup_train()


    def add_new_camera_views(self, step: int, num_inpaint_images: int, 
            dataset: InputDataset, model: Model, inpainter: Inpainter):
        fx = dataset.cameras.fx[-1]
        fy = dataset.cameras.fy[-1]
        cx = dataset.cameras.cx[-1]
        cy = dataset.cameras.cy[-1]
        w = dataset.cameras.width[-1]
        h = dataset.cameras.height[-1]
        distort = dataset.cameras.distortion_params[-1]

        new_cams = []
        new_images = []

        num_cams = dataset.cameras.camera_to_worlds.shape[0]
        assert len(dataset._dataparser_outputs.image_filenames) == num_cams

        for i in range(num_cams):
            t = dataset.cameras.camera_to_worlds[i, :3, 3]
            R = dataset.cameras.camera_to_worlds[i, :3, :3]
            
            orig_cam = torch.cat([R, t.unsqueeze(-1)], dim=-1)
            new_cams.append(orig_cam.clone())
            new_images.append(dataset._dataparser_outputs.image_filenames[i])

        for i in range(num_inpaint_images):
            # TODO

            new_t = None
            new_R = None
            new_cam = torch.cat([new_R, new_t.unsqueeze(-1)], dim=-1)

            new_cams.append(new_cam.clone())
            new_images.append(None)

        assert dataset.cameras.times is None  # not supported
        dataset.cameras = Cameras(
            camera_to_worlds=torch.stack(new_cams, dim=0),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            camera_type=CameraType.PERSPECTIVE,
            distortion_params=distort[None].repeat(len(new_cams), 1),
        )  # update cameras

        dataset._dataparser_outputs.image_filenames = new_images  # update images

    
    def add_existing_camera_views(self, step: int, num_inpaint_images: int, dataset: InputDataset):
        new_cams = []
        new_images = []
        num_cams = dataset.cameras.camera_to_worlds.shape[0]
        assert len(dataset._dataparser_outputs.image_filenames) == num_cams

        # cam_samples = np.floor(np.linspace(0, num_cams-1, num_inpaint_images)).astype(np.int64)
        # cam_samples = list(range(num_inpaint_images))
        
        num_pending_cams = dataset.pending_cameras.camera_to_worlds.shape[0]
        new_pending_list = list(range(num_pending_cams))
        assert len(dataset._dataparser_outputs.pending_image_filenames) == num_pending_cams
        # assert num_pending_cams >= num_inpaint_images, "Not enough pending cameras to sample from"

        # cam_samples = np.floor(np.linspace(0, num_pending_cams-1, num_inpaint_images)).astype(np.int64)
        if len(new_pending_list) == 0:
            return
        elif len(new_pending_list) < num_inpaint_images:
            num_inpaint_images = len(new_pending_list)
        cam_samples = np.random.choice(new_pending_list, num_inpaint_images, replace=False)

        for i in range(num_cams):
            t = dataset.cameras.camera_to_worlds[i, :3, 3]
            R = dataset.cameras.camera_to_worlds[i, :3, :3]
            
            orig_cam = torch.cat([R, t.unsqueeze(-1)], dim=-1)
            new_cams.append(orig_cam.clone())
            new_images.append(dataset._dataparser_outputs.image_filenames[i])

        for j in cam_samples:
            t = dataset.pending_cameras.camera_to_worlds[j, :3, 3]
            R = dataset.pending_cameras.camera_to_worlds[j, :3, :3]

            new_cam = torch.cat([R, t.unsqueeze(-1)], dim=-1)
            new_cams.append(new_cam.clone())
            if self.config.use_pending_images:
                new_images.append(dataset._dataparser_outputs.pending_image_filenames[j])
            else:
                new_images.append(None)

            new_pending_list.remove(j)
        
        assert dataset.cameras.times is None  # not supported
        dataset.cameras = Cameras(
            camera_to_worlds=torch.stack(new_cams, dim=0),
            fx=dataset.cameras.fx[-1],
            fy=dataset.cameras.fy[-1],
            cx=dataset.cameras.cx[-1],
            cy=dataset.cameras.cy[-1],
            width=dataset.cameras.width[-1],
            height=dataset.cameras.height[-1],
            camera_type=CameraType.PERSPECTIVE,
            distortion_params=dataset.cameras.distortion_params[-1][None].repeat(len(new_cams), 1),
        )  # update cameras

        dataset.pending_cameras = Cameras(
            camera_to_worlds=dataset.pending_cameras.camera_to_worlds[new_pending_list],
            fx=dataset.pending_cameras.fx[new_pending_list],
            fy=dataset.pending_cameras.fy[new_pending_list],
            cx=dataset.pending_cameras.cx[new_pending_list],
            cy=dataset.pending_cameras.cy[new_pending_list],
            width=dataset.pending_cameras.width[new_pending_list],
            height=dataset.pending_cameras.height[new_pending_list],
            camera_type=CameraType.PERSPECTIVE,
            distortion_params=dataset.pending_cameras.distortion_params[new_pending_list],
        )  # update cameras

        dataset._dataparser_outputs.image_filenames = new_images  # update images
        dataset._dataparser_outputs.pending_image_filenames = [
            dataset._dataparser_outputs.pending_image_filenames[i] for i in new_pending_list]

