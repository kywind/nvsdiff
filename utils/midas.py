import argparse
import os
import torch
from ldm.modules.depth.model import MiDaS


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
    parser.set_defaults(optimize=False)

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
    model = MiDaS(args.model_weights, args.model_type, device="cuda:7", optimize=args.optimize)
    model.run_offline(args.input_path, args.output_path)
