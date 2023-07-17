# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from time import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import profiler

from nerfstudio.inpainter.stable_diffusion import StableDiffusionInpainter
from nerfstudio.inpainter.image_to_image import ImageToImageInpainter
from nerfstudio.camera_generator.base_camera_generator import CameraGenerator, CameraGeneratorConfig
from nerfstudio.initializer.base_initializer import Initializer, InitializerConfig
from nerfstudio.refiner.base_refine_dataset import RefineDataset, RefineDatasetConfig
from nerfstudio.refiner.base_refine_trainer import RefineTrainer, RefineTrainerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType

import os
import numpy as np
from PIL import Image
from datetime import datetime


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the data manager and model classes from the trainer,
    worrying about:
    1) Fetching data with the data manager
    2) Feeding the model the data and fetching the loss
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'train': loads train/eval datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    # pylint: disable=abstract-method

    datamanager: DataManager
    _model: Model

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        model_state = {
            key.replace("_model.", ""): value for key, value in state_dict.items() if key.startswith("_model.")
        }
        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}
        self._model.load_state_dict(model_state, strict=strict)
        super().load_state_dict(pipeline_state, strict=False)

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average."""

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """


@dataclass
class VanillaPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""
    camera_generator: CameraGeneratorConfig = CameraGeneratorConfig()
    """specifies the camera generator config"""
    initializer: InitializerConfig = InitializerConfig()
    """specifies the initializer config"""
    refine_dataset: RefineDatasetConfig = RefineDatasetConfig()
    """specifies the refine dataset config"""
    refine_trainer: RefineTrainerConfig = RefineTrainerConfig()
    """specifies the refine trainer config"""


class VanillaPipeline(Pipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        timestamp: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        gen_data: bool = False,
        use_sds: bool = False,  # whether to use refinement
        max_num_cameras: int = 500,
        max_iter: int = 10000,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.gen_data = gen_data
        self.use_sds = use_sds

        self.timestamp = timestamp
        # self.timestamp = datetime.now().strftime("%m%d-%H%M%S")

        if self.gen_data:
            self.data_gen_dir = Path(os.path.join(self.config.datamanager.data, self.timestamp))
            self.data_gen_dir.mkdir(exist_ok=True, parents=True)
            self.config.datamanager.data_gen_dir = self.data_gen_dir

            # initializing stage
            self.initializer = config.initializer.setup(
                initialize_save_dir=self.data_gen_dir,
                timestamp=self.timestamp,
            )
            self.initializer.initialize_scene()
        else:
            self.data_gen_dir = None
            self.config.datamanager.data_gen_dir = None

        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, 
            max_num_cameras=max_num_cameras
        )
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=max_num_cameras,
            # num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
        )
        self.model.to(device)

        # Inpainter (containing the diffusion model)
        # self.inpainter = ImageToImageInpainter(
        #     prompt='Real estate photo',
        #     device=device,
        # )
        self.inpainter = None
        
        # Camera Generator (for updating the dataset)
        self.camera_generator: CameraGenerator = config.camera_generator.setup(
            inpaint_save_dir=self.data_gen_dir
        )

        # TODO SDS Loss
        if self.use_sds:
            self.refine_dataset = config.refine_dataset.setup(
                datamanager=self.datamanager,
                device=device,
                max_iter=max_iter
            )
            self.refine_dataloader = self.refine_dataset.dataloader()

            self.sds_start_step = config.refine_trainer.sds_start_step
            self.sds_end_step = config.refine_trainer.sds_end_step
            self.nerf_start_step = config.refine_trainer.nerf_start_step
            self.nerf_end_step = config.refine_trainer.nerf_end_step
            if self.sds_end_step == -1:
                self.sds_end_step = max_iter
            if self.nerf_end_step == -1:
                self.nerf_end_step = max_iter

            self.refine_trainer = config.refine_trainer.setup(
                timestamp=self.timestamp,
                model=self.model,
                sds_iters=self.sds_end_step - self.sds_start_step,
                total_iters=max_iter,
                device=device,
            )

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if (not self.use_sds) or \
                (self.use_sds and step >= self.nerf_start_step and step < self.nerf_end_step):
            ray_bundle, batch = self.datamanager.next_train(step)
            model_outputs = self.model(ray_bundle, step)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

            if self.config.datamanager.camera_optimizer is not None:
                camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
                if camera_opt_param_group in self.datamanager.get_param_groups():
                    # Report the camera optimization metrics
                    metrics_dict["camera_opt_translation"] = (
                        self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                    )
                    metrics_dict["camera_opt_rotation"] = (
                        self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                    )

            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        else:
            loss_dict = {}
            model_outputs = {}
            metrics_dict = {}
        
        # loss_dict: dict_keys(['rgb_loss', 'interlevel_loss', 'distortion_loss', 'depth_loss'])
        # model_outputs: dict_keys(['rgb', 'accumulation', 'depth', 'weights_list', 'ray_samples_list', 'prop_depth_0', 'prop_depth_1', 'directions_norm'])
        # metrics_dict: dict_keys(['psnr', 'distortion', 'depth_loss', 'camera_opt_translation', 'camera_opt_rotation'])

        if self.use_sds and step >= self.sds_start_step and step < self.sds_end_step:
            data = next(self.refine_dataloader) # H, W, rays_o, rays_d, dir, mvp, polar, azimuth, radius
            H, W = data['H'], data['W']
            sds_render = self.model(data['ray_bundle'], step)
            guidance_input = {
                'image': sds_render['rgb'].reshape(H, W, 3),
                'depth': sds_render['depth'].reshape(H, W),
                'weights_sum': sds_render['accumulation'].reshape(H, W),
                'weights': sds_render['weights_list'][-1].reshape(H, W, -1),
                'normal_image': sds_render['pred_normals'].reshape(H, W, 3) if 'pred_normals' in sds_render else None,
            }
            data_meta = {'H': H, 'W': W, 'azimuth': data['azimuth'], 'polar': data['polar'], 'radius': data['radius'], 'c2w': data['c2w']}
            sds_loss_dict, sds_outputs, sds_metrics = self.refine_trainer.train_step(step, guidance_input, data_meta)
        
            loss_dict.update(sds_loss_dict)
            model_outputs.update(sds_outputs)
            metrics_dict.update(sds_metrics)
        
        assert loss_dict != {}
        return model_outputs, loss_dict, metrics_dict

    def inpaint(self, step: int, num_inpaint_cameras: int):
        """Inpaint the cameras that need inpainting"""
        self.eval()
        self.camera_generator.generate_inpaint_cameras(
            step, 
            num_inpaint_cameras,
            self.datamanager,
            self.model,
            self.inpainter,
        )
        self.train()
    
    def save_guidance_images(self, step: int):
        self.refine_trainer.save_guidance_images(step)

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        self._model.update_to_step(step)
        self.load_state_dict(state, strict=True)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        refine_params = self.refine_trainer.get_param_groups() if self.use_sds else {}
        # TODO(ethan): assert that key names don't overlap
        return {**datamanager_params, **model_params, **refine_params}

    def get_state_dict(self):
        """Get the state dict of the pipeline.

        Returns:
            A dictionary containing the pipeline's state dict.
        """
        state_dict_all = self.state_dict()
        state_dict = {}
        for key in state_dict_all.keys():
            if "refine_trainer" in key:  # remove refine_trainer state_dict
                continue
            state_dict[key] = state_dict_all[key].clone()
        del state_dict_all
        return state_dict
