"""
Code for inpainting with diffusion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union


import torch
from torchtyping import TensorType

from nerfstudio.initializer.base_initializer import Initializer, InitializerConfig

from nerfstudio.inpainter.base_inpainter import Inpainter
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DiffusionPipeline

from nerfstudio.configs.base_config import InstantiateConfig

from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle

import kornia
from kornia.geometry.depth import depth_to_3d, project_points, DepthWarper
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.geometry.linalg import compose_transformations, \
        convert_points_to_homogeneous, inverse_transformation, transform_points

from PIL import Image
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from datetime import datetime
import math

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline

import nerfstudio.initializer.utils.trajectory_util as trajectory_util
import nerfstudio.initializer.utils.mesh_util as mesh_util
import nerfstudio.initializer.utils.image_util as image_util

from nerfstudio.initializer.text2room.model.iron_depth.predict_depth import \
    load_iron_depth_model, predict_iron_depth
from nerfstudio.initializer.text2room.model.depth_alignment import depth_alignment
from nerfstudio.initializer.text2room.model.trajectories.convert_to_nerf_convention import \
    convert_pose_to_nerf_convention, convert_pose_from_nerf_convention
from nerfstudio.initializer.text2room.model.mesh_fusion.render import \
    features_to_world_space_mesh, clean_mesh, render_mesh, save_mesh, load_mesh, edge_threshold_filter
from nerfstudio.initializer.text2room.model.utils.utils import \
    visualize_depth_numpy, save_image, pil_to_torch, save_rgbd, load_sd_inpaint, save_settings, save_animation

@dataclass
class Text2RoomInitializerConfig(InitializerConfig):
    """Text2Room dataset config"""

    _target: Type = field(default_factory=lambda: Text2RoomInitializer)
    """target class to instantiate"""
    n_images: int = 19
    """number of images to generate per trajectory"""
    trajectory_file: Path = Path("nerfstudio/initializer/trajectories/living_room_1.json")
    """path to trajectory file"""
    initial_prompt: str = "living room with a lit furnace, couch and cozy curtains, bright lamps that make the room look well-lit"
    """initial prompt for generating the first image"""
    initial_neg_prompt: str = "blurry, bad art, blurred, text, watermark, plant, nature"
    """initial negative prompt for generating the first image"""


class Text2RoomInitializer(Initializer):
    """Model class for inpainting.

    Args:
        config: configuration for instantiating.
    """

    def __init__(
        self,
        config: Text2RoomInitializerConfig, 
        timestamp: str,
        initialize_save_dir: str = None, 
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device
        self.timestamp = timestamp
        self.initialize_save_dir = initialize_save_dir


    def initialize_scene(self, **kwargs) -> Dict[str, Any]:
        """
        Calling this before nerf training. Construct a initialized set of images and depths for the scene, 
        but different with original text2room, we try not to overengineer the scene by using a lot of iterations to fill 
        in the holes in the mesh. Instead, we will use nerf-guided view selection to refine the scene.
        """

        # self.mask_path = None
        # self.rgb_path = None
        # self.rendered_path = None
        self.bbox = [torch.ones(3) * -1.0, torch.ones(3) * 1.0]  # initilize bounding box of meshs as [-1.0, -1.0, -1.0] -> [1.0, 1.0, 1.0]

        self.out_path = 'text2room_vis/'

        # timestamp = datetime.now().strftime('%m%d-%H%M%S')
        self.out_path = os.path.join(self.out_path, self.timestamp)
        os.makedirs(self.out_path, exist_ok=True)

        self.rgb_path = os.path.join(self.out_path, "rgb")
        self.rgbd_path = os.path.join(self.out_path, "rgbd")
        self.rendered_path = os.path.join(self.out_path, "rendered")
        self.depth_path = os.path.join(self.out_path, "depth")
        self.fused_mesh_path = os.path.join(self.out_path, "fused_mesh")
        self.mask_path = os.path.join(self.out_path, "mask")
        self.output_rendering_path = os.path.join(self.out_path, "output_rendering")
        self.output_depth_path = os.path.join(self.out_path, "output_depth")

        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.rgbd_path, exist_ok=True)
        os.makedirs(self.rendered_path, exist_ok=True)
        os.makedirs(self.depth_path, exist_ok=True)
        os.makedirs(self.fused_mesh_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.output_rendering_path, exist_ok=True)
        os.makedirs(self.output_depth_path, exist_ok=True)

        if "prompt" in kwargs:
            prompt = kwargs["prompt"]
        else:
            prompt = self.config.initial_prompt

        self.orig_n_images = self.config.n_images
        self.orig_prompt = prompt
        self.orig_negative_prompt = self.config.initial_neg_prompt
        self.orig_surface_normal_threshold = 0.1
        
        self.n_images = self.orig_n_images
        self.prompt = self.orig_prompt
        self.trajectory_fn = None
        self.negative_prompt = self.orig_negative_prompt
        self.surface_normal_threshold = self.orig_surface_normal_threshold

        self.clean_mesh_every_nth = 20

        self.vertices = None
        self.faces = None
        self.colors = None
        self.H = 512
        self.W = 512
        self.fov = 55.0

        self.world_to_cam = None
        self.blur_radius = 0.0
        self.faces_per_pixel = 8

        self.rendered_depth = torch.zeros((self.H, self.W), device=self.device)  # depth rendered from point cloud
        self.inpaint_mask = torch.ones((self.H, self.W), device=self.device, dtype=torch.bool)  # 1: no projected points (need to be inpainted) | 0: have projected points
        self.vertices = torch.empty((3, 0), device=self.device)
        self.colors = torch.empty((3, 0), device=self.device)
        self.faces = torch.empty((3, 0), device=self.device, dtype=torch.long)
        self.pix_to_face = None

        self.trajectory_fn = trajectory_util.forward()
        self.trajectory_dict = {}
        self.world_to_cam = torch.eye(4, dtype=torch.float32, 
            device=self.device) # if start_pose is None else start_pose.to(self.device)
        self.K = mesh_util.get_pinhole_intrinsics_from_fov(H=self.H, W=self.W, 
            fov_in_degrees=self.fov).to(self.world_to_cam)

        self.seen_poses = []
        offset = 0

        traj_file = self.config.trajectory_file
        trajectories = json.load(open(traj_file, "r"))

        self.json_file = self.build_nerf_transforms_header()
        # self.json_file = {}
        # self.json_file['fl_x'] = 1.0 * self.W / (2.0 * np.tan(self.fov * np.pi / 360.0))
        # self.json_file['fl_y'] = 1.0 * self.H / (2.0 * np.tan(self.fov * np.pi / 360.0))
        # self.json_file['cx'] = self.W / 2.0
        # self.json_file['cy'] = self.H / 2.0
        # self.json_file['w'] = self.W
        # self.json_file['h'] = self.H
        # self.json_file['frames'] = []

        # ---------- MAIN LOOP -----------
        self.setup_models()
        # offset = self.setup_start_image(offset)

        for t in trajectories:
            self.set_trajectory(t)
            offset = self.generate_images(offset=offset)
        # ---------- MAIN LOOP -----------
        
        json_save_path = os.path.join(self.initialize_save_dir, 'transforms.json')
        with open(json_save_path, 'w') as f:
            json.dump(self.json_file, f)
        
        self.post_initialization()

    def post_initialization(self):
        """
        post initialization
        """
        del self.inpaint_pipe
        del self.iron_depth_n_net
        del self.iron_depth_model

    def set_trajectory(self, trajectory_dict: Any):
        """
        parse the trajectory config
        """
        self.trajectory_dict = trajectory_dict
        fn = getattr(trajectory_util, trajectory_dict["fn_name"])
        self.trajectory_fn = fn(**trajectory_dict["fn_args"])
        self.n_images = trajectory_dict.get("n_images", self.orig_n_images)
        self.prompt = trajectory_dict.get("prompt", self.orig_prompt)
        self.negative_prompt = trajectory_dict.get("negative_prompt", self.orig_negative_prompt)
        self.surface_normal_threshold = trajectory_dict.get("surface_normal_threshold", self.orig_surface_normal_threshold)

    def generate_images(self, offset: int = 0):
        """
        extract pose, call project_and_inpaint
        """
        pbar = tqdm(range(self.n_images))
        for pos in pbar:
            pbar.set_description(f"Image [{pos}/{self.n_images - 1}]")
            self.world_to_cam = self.trajectory_fn(pos, self.n_images).to(self.device)
            self.seen_poses.append(self.world_to_cam.clone())
            # render --> inpaint --> add to 3D structure
            if pos == 0 and offset == 0:
                _ = self.setup_start_image(offset)
            else:
                self.project_and_inpaint(pos, offset, save_files=True)
            if self.clean_mesh_every_nth > 0 and (pos + offset) % self.clean_mesh_every_nth == 0:
                self.vertices, self.faces, self.colors = clean_mesh(
                    vertices=self.vertices,
                    faces=self.faces,
                    colors=self.colors,
                    edge_threshold=0.1,
                    min_triangles_connected=15000,
                    fill_holes=True
                )

        # reset gpu memory
        torch.cuda.empty_cache()
        return offset + self.n_images

    def project_and_inpaint(self, pos=0, offset=0, save_files=True, file_suffix=""):
        """
        project the current mesh. Then call the inpainter.
        """
        # project to next pose
        _, rendered_image_pil, inpaint_mask_pil = self.project()

        if "adaptive" in self.trajectory_dict:
            def update_pose(reverse=False):
                # update the args in trajectory dict
                for d in self.trajectory_dict["adaptive"]:
                    arg = d["arg"]
                    delta = d["delta"] if not reverse else -d["delta"]
                    self.trajectory_dict["fn_args"][arg] += delta

                    if "min" in d:
                        self.trajectory_dict["fn_args"][arg] = max(d["min"], self.trajectory_dict["fn_args"][arg])
                    if "max" in d:
                        self.trajectory_dict["fn_args"][arg] = min(d["max"], self.trajectory_dict["fn_args"][arg])

                # update pose
                self.set_trajectory(self.trajectory_dict)
                self.world_to_cam = self.trajectory_fn(pos, self.n_images).to(self.device)
                self.seen_poses[-1] = self.world_to_cam

                # render new images
                return self.project()

            for i in range(10):
                # increase args as long as depth does not get smaller again
                # can extend this to allow multiple comparisons: e.g., add "as long as mean depth not smaller than X"
                old_std_depth, old_mean_depth = torch.std_mean(self.rendered_depth)
                _, rendered_image_pil, inpaint_mask_pil = update_pose()
                current_std_depth, current_mean_depth = torch.std_mean(self.rendered_depth)

                if current_mean_depth <= old_mean_depth:
                    # go back one step and end search
                    _, rendered_image_pil, inpaint_mask_pil = update_pose(reverse=True)
                    break

        # inpaint projection result
        inpainted_image_pil, eroded_dilated_inpaint_mask_pil = self.inpaint(rendered_image_pil, inpaint_mask_pil)
        if save_files:
            image_util.save_image(eroded_dilated_inpaint_mask_pil, 
            f"mask_eroded_dilated{file_suffix}", 
            offset + pos, self.mask_path)
        # else:
        #     self.eroded_dilated_inpaint_mask = torch.zeros_like(self.inpaint_mask)

        # update images
        self.current_image_pil = inpainted_image_pil
        self.current_image = image_util.pil_to_torch(inpainted_image_pil, self.device)

        if save_files:
            image_util.save_image(rendered_image_pil, f"rendered{file_suffix}", offset + pos, self.rendered_path)
            image_util.save_image(inpaint_mask_pil, f"mask{file_suffix}", offset + pos, self.mask_path)
            image_util.save_image(self.current_image_pil, f"rgb{file_suffix}", offset + pos, self.rgb_path)

        # predict depth, add to 3D structure, save images
        self.add_next_image(pos, offset, save_files, file_suffix)
        # update dataset
        self.append_nerf_extrinsic(offset + pos, self.current_image_pil, self.predicted_depth, self.current_depth_pil)

        # update bounding box
        min_bound = torch.amin(self.vertices, dim=-1)
        max_bound = torch.amax(self.vertices, dim=-1)
        self.bbox = [min_bound, max_bound]
    
    def project(self):
        """
        project the current mesh under the current pose
        """
        rendered_image_tensor, self.rendered_depth, self.inpaint_mask, self.pix_to_face, self.z_buf = render_mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_features=self.colors,
            H=self.H,
            W=self.W,
            fov_in_degrees=self.fov,
            RT=self.world_to_cam,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel
        )

        # mask rendered_image_tensor
        rendered_image_tensor = rendered_image_tensor * ~self.inpaint_mask

        # stable diffusion models want the mask and image as PIL images
        rendered_image_pil = Image.fromarray((rendered_image_tensor.permute(1, 2, 0).detach().cpu().numpy()[..., :3] * 255).astype(np.uint8))
        inpaint_mask_pil = Image.fromarray(self.inpaint_mask.detach().cpu().squeeze().float().numpy() * 255).convert("RGB")

        return rendered_image_tensor, rendered_image_pil, inpaint_mask_pil

    def inpaint(self, rendered_image_pil, inpaint_mask_pil):
        """
        inpaint the mesh-rendered(projected) image using stable diffusion
        """
        m = np.asarray(inpaint_mask_pil)[..., 0].astype(np.uint8)

        # inpaint with classical method to fill small gaps
        rendered_image_numpy = np.asarray(rendered_image_pil)
        rendered_image_pil = Image.fromarray(cv2.inpaint(rendered_image_numpy, m, 3, cv2.INPAINT_TELEA))

        # remove small seams from mask
        kernel = np.ones((7, 7), np.uint8)
        m2 = m
        erode_iters = 1
        dilate_iters = 2
        if erode_iters > 0:
            m2 = cv2.erode(m, kernel, iterations=erode_iters)
        if dilate_iters > 0:
            m2 = cv2.dilate(m2, kernel, iterations=dilate_iters)

        # do not allow mask to extend to boundaries
        boundary_thresh = 10
        if boundary_thresh > 0:
            t = boundary_thresh
            h, w = m2.shape
            m2[:t] = m[:t]  # top
            m2[h - t:] = m[h - t:]  # bottom
            m2[:, :t] = m[:, :t]  # left
            m2[:, w - t:] = m[:, w - t:]  # right

        # close inner holes in dilated masks -- find out-most contours and fill everything inside
        contours, hierarchy = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(m2, contours, -1, 255, thickness=cv2.FILLED)

        # convert back to pil & save updated mask
        inpaint_mask_pil = Image.fromarray(m2).convert("RGB")
        self.eroded_dilated_inpaint_mask = torch.from_numpy(m2).to(self.inpaint_mask)

        # update inpaint mask to contain all updates
        update_mask_after_improvement = False
        if update_mask_after_improvement:
            self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask

        # inpaint large missing areas with stable-diffusion model
        guidance_scale = 7.5
        num_inference_steps = 50
        inpainted_image_pil = self.inpaint_pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=1,
            image=rendered_image_pil,
            mask_image=inpaint_mask_pil,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        return inpainted_image_pil, inpaint_mask_pil

    def setup_models(self):
        """
        setup stable diffusion model and depth model
        """
        # construct inpainting stable diffusion pipeline
        # model_path = os.path.join(args.models_path, "stable-diffusion-2-inpainting")
        # if not os.path.exists(model_path):
        # model_path = "stabilityai/stable-diffusion-2-inpainting"
        self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", 
            torch_dtype=torch.float16).to(self.device)

        self.inpaint_pipe.set_progress_bar_config(**{
            "leave": False,
            "desc": "Generating Next Image"
        })

        self.inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.inpaint_pipe.scheduler.config)

        # construct depth model
        self.iron_depth_type = 'scannet'
        self.iron_depth_iters = 20
        self.models_path = 'checkpoints/'
        self.iron_depth_n_net, self.iron_depth_model = load_iron_depth_model(
            self.iron_depth_type, 
            self.iron_depth_iters, 
            self.models_path, 
            self.device
        )

    def setup_start_image(self, offset, save_file=False, file_suffix="_start"):
        """
        setup the start image with torch.eye pose
        """
        model_path = "stabilityai/stable-diffusion-2-1"
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(self.device)

        pipe.set_progress_bar_config(**{
            "leave": False,
            "desc": "Generating Start Image"
        })

        init_image = pipe(self.prompt).images[0]
        # save & convert first_image
        self.current_image_pil = init_image
        self.current_image_pil = self.current_image_pil.resize((self.W, self.H))

        # save to text2room visualization folder
        self.current_image = pil_to_torch(self.current_image_pil, self.device)
        save_image(self.current_image_pil, "rgb", offset, self.rgb_path)

        # predict depth, add 3D structure, save image
        save_first_frame = True
        if save_first_frame:
            self.add_next_image(pos=0, offset=offset)
            # add to seen poses
            self.seen_poses.append(self.world_to_cam.clone())
            # update dataset
            self.append_nerf_extrinsic(offset, self.current_image_pil, self.predicted_depth, self.current_depth_pil)
            offset += 1
        # save image
        if save_file:
            image_util.save_image(self.current_image_pil, f"rgb{file_suffix}", 0, self.rgb_path)

        del pipe
        return offset

    def add_next_image(self, pos, offset, save_files=True, file_suffix=""):
        """
        generic funciton for adding the next image (stored in self.current_image_pil) to the 3D structure
        """
        # predict & align depth of current image
        predicted_depth, _ = predict_iron_depth(
            image=self.current_image_pil,
            K=self.K,
            device=self.device,
            model=self.iron_depth_model,
            n_net=self.iron_depth_n_net,
            input_depth=self.rendered_depth,
            input_mask=self.inpaint_mask,
            fix_input_depth=True
        )
        predicted_depth = depth_alignment.scale_shift_linear(
            rendered_depth=self.rendered_depth,
            predicted_depth=predicted_depth,
            mask=~self.inpaint_mask,
            fuse=True
        )
        predicted_depth = self.apply_depth_smoothing(predicted_depth, self.inpaint_mask)
        self.predicted_depth = predicted_depth

        rendered_depth_pil = Image.fromarray(visualize_depth_numpy(self.rendered_depth.cpu().numpy())[0].astype(np.uint8))
        depth_pil = Image.fromarray(visualize_depth_numpy(predicted_depth.cpu().numpy())[0].astype(np.uint8))
        self.current_depth_pil = depth_pil
        if save_files:
            save_image(rendered_depth_pil, f"rendered_depth{file_suffix}", offset + pos, self.depth_path)
            save_image(depth_pil, f"depth{file_suffix}", offset + pos, self.depth_path)
            save_rgbd(self.current_image_pil, depth_pil, f'rgbd{file_suffix}', offset + pos, self.rgbd_path)

        # # remove masked-out faces. If we use erosion in the mask it means those points will be removed.
        # self.replace_over_inpainted = False
        # if self.replace_over_inpainted:
        #     # only now update mask: predicted depth will still take old positions as anchors (they are still somewhat correct)
        #     # otherwise if we erode/dilate too much we could get depth estimates that are way off
        #     if not self.args.update_mask_after_improvement:
        #         self.inpaint_mask = self.inpaint_mask + self.eroded_dilated_inpaint_mask
        #     self.remove_masked_out_faces()

        # add new points (novel content)
        self.add_vertices_and_faces(predicted_depth)

        # save current meshes
        self.save_scene_every_nth = 10
        if save_files and self.save_scene_every_nth > 0 and (offset + pos) % self.save_scene_every_nth == 0:
            self.save_mesh(f"fused_until_frame{file_suffix}_{offset + pos:04}.ply")
    
    def add_vertices_and_faces(self, predicted_depth):
        """
        update the mesh using the predicted depth
        """
        if self.inpaint_mask.sum() == 0:
            # when no pixels were masked out, we do not need to add anything, so skip this call
            return

        vertices, faces, colors = features_to_world_space_mesh(
            colors=self.current_image,
            depth=predicted_depth,
            fov_in_degrees=self.fov,
            world_to_cam=self.world_to_cam,
            mask=self.inpaint_mask,
            edge_threshold=0.1,
            surface_normal_threshold=0.1,
            pix_to_face=self.pix_to_face,
            faces=self.faces,
            vertices=self.vertices
        )

        faces += self.vertices.shape[1]  # add face offset

        self.vertices = torch.cat([self.vertices, vertices], dim=1)
        self.colors = torch.cat([self.colors, colors], dim=1)
        self.faces = torch.cat([self.faces, faces], dim=1)

    def apply_depth_smoothing(self, image, mask):
        """
        utility function for smoothing the depth (why is this not in utils?)
        """
        def dilate(x, k=3):
            x = torch.nn.functional.conv2d(
                x.float()[None, None, ...],
                torch.ones(1, 1, k, k).to(x.device),
                padding="same"
            )
            return x.squeeze() > 0

        def sobel(x):
            flipped_sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32).to(x.device)
            flipped_sobel_x = torch.stack([flipped_sobel_x, flipped_sobel_x.t()]).unsqueeze(1)

            x_pad = torch.nn.functional.pad(x.float()[None, None, ...], (1, 1, 1, 1), mode="replicate")

            x = torch.nn.functional.conv2d(
                x_pad,
                flipped_sobel_x,
                padding="valid"
            )
            dx, dy = x.unbind(dim=-3)
            # return torch.sqrt(dx**2 + dy**2).squeeze()
            # new content is created mostly in x direction, sharp edges in y direction are wanted (e.g. table --> wall)
            return dx.squeeze()

        edges = sobel(mask)
        dilated_edges = dilate(edges, k=21)

        img_numpy = image.float().cpu().numpy()
        blur_bilateral = cv2.bilateralFilter(img_numpy, 5, 140, 140)
        blur_gaussian = cv2.GaussianBlur(blur_bilateral, (5, 5), 0)
        blur_gaussian = torch.from_numpy(blur_gaussian).to(image)

        image_smooth = torch.where(dilated_edges, blur_gaussian, image)
        return image_smooth

    def build_nerf_transforms_header(self):
        """
        initialize the transform.json for nerfstudio
        """
        return {
            "fl_x": self.K[0, 0].cpu().numpy().item(),
            "fl_y": self.K[1, 1].cpu().numpy().item(),
            "cx": self.K[0, 2].cpu().numpy().item(),
            "cy": self.K[1, 2].cpu().numpy().item(),
            "w": self.W,
            "h": self.H,
            "camera_angle_x": self.fov * math.pi / 180.0,
            "aabb_scale": 4,
            "integer_depth_scale": 10000,
            "frames": []
        }

    def append_nerf_extrinsic(self, image_id, image, depth=None, depth_vis=None):  # PIL image
        """
        given an image, save the extrinsic to construct the transform.json for nerfstudio
        """
        p = convert_pose_to_nerf_convention(self.world_to_cam)
        save_path = f'{image_id:05d}.png'
        image.save(os.path.join(self.initialize_save_dir, save_path))
        if depth is None:
            self.json_file["frames"].append({
                "transform_matrix": p.cpu().numpy().tolist(),
                "file_path": save_path,
            })
        else:
            depth_save_path = f'{image_id:05d}_depth.npy'
            if depth_vis is not None:
                depth_vis.save(os.path.join(self.initialize_save_dir, f'{image_id:05d}_depth.png'))
            np.save(os.path.join(self.initialize_save_dir, depth_save_path), depth.cpu().numpy())
            self.json_file["frames"].append({
                "transform_matrix": p.cpu().numpy().tolist(),
                "file_path": save_path,
                "depth_file_path": depth_save_path
            })

    def save_mesh(self, name='fused_final.ply'):
        """
        saving the mesh for visualization
        """
        target_path = os.path.join(self.fused_mesh_path, name)

        save_mesh(
            vertices=self.vertices,
            faces=self.faces,
            colors=self.colors,
            target_path=target_path
        )

        return target_path
