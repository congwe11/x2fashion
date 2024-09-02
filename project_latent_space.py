import torchvision.transforms as transforms
import os.path as osp
import cv2
import torch
import os, argparse
import tqdm
from PIL import Image
from diffusers import AutoencoderKL
import random
import numpy as np
device = torch.device("cuda")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# vae = AutoencoderKL.from_pretrained(
#             "CompVis/stable-diffusion-v1-4",
#             subfolder="vae",
#             revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
#         ).to(device)

vae = AutoencoderKL.from_pretrained("pretrained_model/sd-vae-ft-mse").to(device)
vae.requires_grad_(False)

def get_transform():
    image_transforms = transforms.Compose(
        [
        # transforms.Resize((640, 512), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Resize((512, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    return image_transforms


@torch.no_grad()
def VAE_decode(video):
    decoded_video = None
    for i in range(video.shape[1]):
        image = video[:, i, :, :, :]
        image = 1 / 0.18215 * image
        if i == 0:
            image = vae.decode(image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            decoded_video = image.unsqueeze(1)
        else:
            image = vae.decode(image).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            decoded_video = torch.cat([decoded_video, image.unsqueeze(1)], 1)
    return decoded_video

@torch.no_grad()
def VAE_encode(video):
    for i in range(video.shape[0]):
        image = video[i, :, :, :]
        image = image.unsqueeze(0)
        if i == 0:
            init_latent_dist = vae.encode(image).latent_dist.sample()
            init_latent_dist *= 0.18215
            encoded_video = (init_latent_dist).unsqueeze(1)
        else:
            init_latent_dist = vae.encode(image).latent_dist.sample()
            init_latent_dist *= 0.18215
            encoded_video = torch.cat([encoded_video, (init_latent_dist).unsqueeze(1)], 1)
    return encoded_video

import cv2
import torchvision.transforms.functional as TVF
def save_video_frames_as_mp4(frames, fps, save_path):
    frame_h, frame_w = frames[0].shape[2:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = frames[0]
    for frame in frames:
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()


import os, argparse
parser = argparse.ArgumentParser(description="Configuration of the tensor projection.")
parser.add_argument('--dataset', default="fashion_dataset/", help="Path to the dataset")
parser.add_argument('--output_dir', default="fashion_dataset_tensor", help="Path to save the tensors")
args = parser.parse_args()

path = osp.join(args.dataset, "fashion_crop")
video_names = os.listdir(path)
transform = get_transform()

output_dir = osp.join(args.output_dir, "fashion_crop")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for video_name in tqdm.tqdm(video_names):
    video = []
    frames = os.listdir(osp.join(path, video_name))
    frames.sort()

    for f in frames:
        frame = cv2.imread(osp.join(path, video_name, f))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frame = frame.unsqueeze(0)

        video.append(frame)

    video = torch.cat(video, 0)
    video = video.to(device=device)
    latent_video = VAE_encode(video)
    vae_video = latent_video.detach().cpu()[0]
    torch.save(vae_video, output_dir + "/" + video_name + ".pt")

    # vae_video = vae_video.unsqueeze(0)
    gen_video = VAE_decode(latent_video)
    save_video_frames_as_mp4(gen_video, 24, output_dir + "/" + video_name + ".mp4")
