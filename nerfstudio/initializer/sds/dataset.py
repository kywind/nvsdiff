from pathlib import Path
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data import DataLoader

from nerfstudio.initializer.sds.utils.pose_utils import rand_poses, circle_poses, visualize_poses
from nerfstudio.initializer.sds.utils.utils import get_rays
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager


@dataclass
class SDSDatasetConfig(InstantiateConfig):
    """Configuration for model instantiation"""

    _target: Type = field(default_factory=lambda: SDSDataset)
    """target class to instantiate"""
    height: int = 64
    """image height"""
    width: int = 64
    """image width"""
    fovy_min: float = 40
    """minimum fovy"""
    fovy_max: float = 60
    """maximum fovy"""
    radius_min: float = 0.5
    """minimum radius"""
    radius_max: float = 0.6
    """maximum radius"""
    theta_min: float = 90
    """minimum theta"""
    theta_max: float = 90
    """maximum theta"""
    phi_min: float = -180
    """minimum phi"""
    phi_max: float = 180
    """maximum phi"""
    jitter_pose: bool = False
    """jitter pose"""

class SDSDataset:
    def __init__(
        self, 
        config,
        datamanager: DataManager = None,
        device: str = "cuda",
        max_iter: int = 10000,
    ):
        self.batch_size = 1
        self.min_near = 0.01
        self.angle_overhead = 30
        self.angle_front = 60
        self.jitter_pose = config.jitter_pose
        self.uniform_sphere_rate = 0
        # self.known_view_scale = 1.5

        # self.default_radius = 3.2
        # self.default_fovy = 20
        self.default_polar = 90
        self.default_azimuth = 0

        self.fovy_range = [config.fovy_min, config.fovy_max]
        self.radius_range = [config.radius_min, config.radius_max]
        self.theta_range = [config.theta_min, config.theta_max]
        self.phi_range = [config.phi_min, config.phi_max]

        self.device = device

        self.H = config.height
        self.W = config.width
        self.max_iter = max_iter

        # self.mode = mode # train, val, test
        # self.training = self.mode in ['train', 'all']

        self.cx = self.H / 2
        self.cy = self.W / 2

        self.near = self.min_near
        self.far = 1000 # infinite

        self.datamanager = datamanager
        self.count = 0
        ## visualize poses
        # poses, dirs, _, _, _ = rand_poses(100, self.device, 
        #     radius_range=self.radius_range, angle_overhead=self.angle_overhead, 
        #     angle_front=self.angle_front, jitter=self.jitter_pose, uniform_sphere_rate=1)
        # visualize_poses(poses.detach().cpu().numpy(), dirs.detach().cpu().numpy())


    def collate(self, index):
        B = len(index) # always 1
        existing_cameras = self.datamanager.train_dataset.cameras

        # if self.training:
        if True:
            # random pose on the fly
            poses, dirs, thetas, phis, radius = rand_poses(B, self.device, existing_cameras,
                radius_range=self.radius_range, theta_range=self.theta_range, phi_range=self.phi_range, 
                return_dirs=True, angle_overhead=self.angle_overhead, angle_front=self.angle_front, 
                jitter=self.jitter_pose, uniform_sphere_rate=self.uniform_sphere_rate)

            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
        # else:
        #     # circle pose
        #     thetas = torch.FloatTensor([self.default_polar]).to(self.device)
        #     phis = torch.FloatTensor([(index[0] / self.size) * 360]).to(self.device)
        #     radius = torch.FloatTensor([self.default_radius]).to(self.device)
        #     poses, dirs = circle_poses(self.device, radius=radius, theta=thetas, phi=phis, return_dirs=True, 
        #         angle_overhead=self.angle_overhead, angle_front=self.angle_front)
        #     # fixed focal
        #     fov = self.default_fovy

        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        # projection = torch.tensor([
        #     [2*focal/self.W, 0, 0, 0],
        #     [0, -2*focal/self.H, 0, 0],
        #     [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
        #     [0, 0, -1, 0]
        # ], dtype=torch.float32, device=self.device).unsqueeze(0)
        # mvp = projection @ torch.inverse(poses) # [1, 4, 4]

        # sample a low-resolution but full image
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)
        fx, fy, cx, cy = intrinsics

        # delta polar/azimuth/radius to default view
        delta_polar = thetas - self.default_polar
        delta_azimuth = phis - self.default_azimuth
        delta_azimuth[delta_azimuth > 180] -= 360 # range in [-180, 180]
        # delta_radius = radius - self.default_radius

        # distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=self.H,
            width=self.W,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )
        ## DEBUG
        # if self.count % 10 == 0:
        if False:
            # import ipdb; ipdb.set_trace()
            orig_cameras = self.datamanager.train_dataset.cameras
            n = orig_cameras.camera_to_worlds.shape[0]
            self.datamanager.train_dataset.cameras = Cameras(
                fx=orig_cameras.fx[-1],
                fy=orig_cameras.fy[-1],
                cx=orig_cameras.cx[-1],
                cy=orig_cameras.cy[-1],
                height=orig_cameras.height[-1],
                width=orig_cameras.width[-1],
                camera_to_worlds=torch.cat((orig_cameras.camera_to_worlds, 
                    poses[:, :3, :4].to(orig_cameras.camera_to_worlds.device)), dim=0),
                camera_type=CameraType.PERSPECTIVE,
            )
            self.datamanager.train_dataset._dataparser_outputs.cameras = self.datamanager.train_dataset.cameras
            # dummy image
            self.datamanager.train_dataset.depth_filenames.append(
                Path('data/text2room_generate/test/0702-171330/00001_depth.png'))
            self.datamanager.train_dataset._dataparser_outputs.image_filenames.append(
                Path('data/text2room_generate/test/0702-171330/00001.png'))
            
            self.datamanager.setup_train()
        self.count += 1
        # import ipdb; ipdb.set_trace()
        # c = ray_indices[:, 0]  # camera indices
        # y = ray_indices[:, 1]  # row indices
        # x = ray_indices[:, 2]  # col indices
        # coords = image_coords[y, x]
        image_coords = cameras.get_image_coords()# .to(cameras.device)
        # pose_optimizer: CameraOptimizer
        # camera_opt_to_camera = self.pose_optimizer(c)
        ray_bundle = cameras.generate_rays(
            camera_indices=0,
            coords=image_coords.reshape(-1, 2),
        )
        # import ipdb; ipdb.set_trace()
        return {
            'ray_bundle': ray_bundle,
            'H': self.H,
            'W': self.W,
            'polar': delta_polar,
            'azimuth': delta_azimuth,
            # 'radius': delta_radius,
        }
        # import ipdb; ipdb.set_trace()

        # data = {
        #     'H': self.H,
        #     'W': self.W,
        #     'rays_o': rays['rays_o'],
        #     'rays_d': rays['rays_d'],
        #     'dir': dirs,
        #     'mvp': mvp,
        #     'polar': delta_polar,
        #     'azimuth': delta_azimuth,
        #     'radius': delta_radius,
        # }
        # return data

    def dataloader(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        loader = DataLoader(list(range(self.max_iter)), batch_size=batch_size, 
            collate_fn=self.collate, shuffle=True, num_workers=0)
            # collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self
        return iter(loader)