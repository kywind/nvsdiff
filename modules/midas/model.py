import os
import glob
import torch
import torch.nn as nn
import cv2
import argparse

import numpy as np
from tqdm import tqdm

from torchvision.transforms import Compose
from .utils import read_image, write_depth
from .dpt_depth import DPTDepthModel
from .midas_net import MidasNet
from .midas_net_custom import MidasNet_small
from .transforms import Resize, NormalizeImage, PrepareForNet


class MiDaS(nn.Module):

    def __init__(
            self, 
            model_path='models/midas/dpt_large-midas-2f21e586.pt', 
            model_type="dpt_large", 
            device=None, 
            optimize=False
        ):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.optimize = optimize

        print("initialize")

        # select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("device: %s" % self.device)

        # load network
        if model_type == "dpt_large": # DPT-Large
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode = "minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "dpt_hybrid": #DPT-Hybrid
            self.model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
            )
            net_w, net_h = 384, 384
            resize_mode="minimal"
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif model_type == "midas_v21":
            self.model = MidasNet(model_path, non_negative=True)
            net_w, net_h = 384, 384
            resize_mode="upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        elif model_type == "midas_v21_small":
            self.model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
            resize_mode="upper_bound"
            normalization = NormalizeImage(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            print(f"model_type '{model_type}' not implemented, use: --model_type large")
            assert False

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
        
        self.model.eval()

        if optimize == True:
            # rand_example = torch.rand(1, 3, net_h, net_w)
            # model(rand_example)
            # traced_script_module = torch.jit.trace(model, rand_example)
            # model = traced_script_module

            if device != torch.device("cpu"):
                self.model = self.model.to(memory_format=torch.channels_last)  
                self.model = self.model.half()

        self.model.to(device)


    def run_offline(self, input_path, output_path):
        """Run MonoDepthNN to compute depth maps.

        Args:
            input_path (str): path to input folder
            output_path (str): path to output folder
            model_path (str): path to saved model
        """
        
        # get input
        img_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(img_names)

        os.makedirs(output_path, exist_ok=True)
        
        # batch
        batch_size = 64
        print('number of images: ', num_images)
        print('batch size: ', batch_size)

        print("start processing")

        img_name_chunks = [img_names[i:i+batch_size] for i in range(0, len(img_names), batch_size)]

        for ind, img_name_chunk in tqdm(enumerate(img_name_chunks)):
            # input
            img_input_list = []
            for batch_index_count in range(len(img_name_chunk)):
                img_name = img_name_chunk[batch_index_count]
                img = read_image(img_name)
                img_input = self.transform({"image": img})["image"][None]
                img_input_list.append(img_input)

            # input
            img_input = np.concatenate(img_input_list, axis=0)   

            # compute
            with torch.no_grad():
                prediction = self.forward(img_input, return_original=True)

            for batch_index_count in range(len(img_name_chunk)):
                # output
                img_name = img_name_chunk[batch_index_count]
                filename = os.path.join(
                    output_path, 
                    os.path.splitext(os.path.basename(img_name))[0]
                )
                write_depth(filename, prediction[batch_index_count], bits=2)

        print("finished")
    
    def forward_PIL(self, img, return_original=False):
        img = np.array(img) / 255.
        images = self.transform({"image": img})["image"][None]
        with torch.no_grad():
            prediction = self.forward(images, return_original=return_original)
        return prediction

    def forward(self, images, return_original=False):
        b, _, h, w = images.shape
        sample = torch.from_numpy(images).to(self.device)
        if self.optimize == True and self.device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)  
            sample = sample.half()
        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        if return_original:
            return prediction  # b, h, w
        else:
            return prediction / 1000.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default=None,
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='dpt_large',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )

    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    default_models = {
        "midas_v21_small": "models/midas/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "models/midas/midas_v21-f6b98070.pt",
        "dpt_large": "models/midas/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "models/midas/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    model = MiDaS(args.model_weights, args.model_type, optimize=args.optimize)
    model.run_offline(args.input_path, args.output_path)
