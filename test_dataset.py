from dataset import FashionTestDataset
import torch
from PIL import Image

import cv2
import numpy as np
import torchvision.transforms.functional as TVF
import os


def save_video_frames_as_mp4(frames, fps, save_path):
    frame_h, frame_w = frames[0].shape[2:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = frames[0]
    for frame in frames:
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

def create_video_from_images(image_dir, start, end, video_name, frame_size, fps=24):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_name, fourcc, fps, frame_size)
    
    for i in range(start, start + end):
        img_name = os.path.join(image_dir, f'{i:03d}.png')
        assert os.path.isfile(img_name), "path is not exists!!!"

        img = cv2.imread(img_name)
        video.write(img)

    video.release()

dataset = FashionTestDataset(
        video_data_root="fashion_dataset/fashion_crop",
        video_tensor_root="fashion_dataset_tensor/fashion_crop",
        poses_data_root="fashion_dataset/fashion_poses",
        motion_text_root="fashion_dataset/test",
        frameLimit=24
    )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1,)


video_data_root="fashion_dataset/fashion_crop"
gt_video="metrics/gt_videos"

for i, batch in enumerate(dataloader):
    name = batch['name'][0]

    start = batch['start'].item()
    # end = batch['end'].item()
    end = 24

    image_dir = os.path.join(video_data_root, name)

    first_image_path = os.path.join(image_dir, f'{start:03d}.png')
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f'First image {first_image_path} not found.')
        continue
    
    frame_size = (first_image.shape[1], first_image.shape[0])

    video_name = os.path.join(gt_video, f'{name}_{start}_{start+end}.mp4')

    create_video_from_images(image_dir, start, end, video_name, frame_size, fps=12)

    print(f'{i}: Video created successfully.')





