import PIL
import requests
import torch
from io import BytesIO
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from ldm.data.re10k import RE10KValidation

def get_image_and_mask(batch, device):
    image = batch["image"].to(device)
    ## inpaint
    # make a simple center square
    b, h, w = image.shape[0], image.shape[1], image.shape[2]
    assert b == 1
    image = image[0].permute(2,0,1)
    image = (image + 1.) * 0.5
    mask = torch.zeros(h, w).to(device)
    # zeros will be filled in
    alpha = (0.1,0.9,0.1,0.9)
    mask[int(h * alpha[0]):int(h * alpha[1]), int(w * alpha[2]):int(w * alpha[3])] = 1.
    mask = mask[None].repeat(3,1,1)
    # mask = 1. - mask

    # masked_image = (1 - mask) * image
    image = transforms.ToPILImage()(image).resize((512, 512))
    mask = transforms.ToPILImage()(mask).resize((512, 512))
    return image, mask

def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

dataset = RE10KValidation(size=256, low=1, high=30, interval=10)
dl = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

for idx, batch in enumerate(dl):
    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    # init_image = download_image(img_url).resize((512, 512))
    # mask_image = download_image(mask_url).resize((512, 512))
    init_image, mask_image = get_image_and_mask(batch, "cuda")

    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "real estate photo"
    init_image.save('init_image.png')
    mask_image.save('mask_image.png')
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    path = 'vis/0220-inpaint'
    os.makedirs(path, exist_ok=True)
    image.save(os.path.join(path, f'{idx}.png'))
    # import ipdb; ipdb.set_trace()
