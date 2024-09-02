import os
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
from transformers import CLIPVisionModel, CLIPProcessor
import torch.nn as nn
import torch
import torch.utils.data as data
import os.path as osp
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import torchvision.transforms.functional as TVF
import torch.nn.functional as F
from models.unet_dual_encoder import Embedding_Adapter
from distributed import (get_rank, synchronize)
from diffusers import AutoencoderKL
from einops import rearrange
from datetime import datetime


# from models.diffusion_model import SpaceTimeUnet
from dataset import FashionDataset
from models.unet import SpaceTimeUnet
from models.condition_app import AppConditioningEmbedding, App2DConditioningEmbedding
from models.condition_pose import PoseConditioningEmbedding, PoseConditioningEmbedding_
from pipeline import MotionControlPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from PIL import Image

parser = argparse.ArgumentParser(description="Configuration of the training script.")
parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
parser.add_argument('--video_data_root', default="fashion_dataset/fashion_crop", help="Path to the dataset")
parser.add_argument('--video_tensor_root', default="fashion_dataset_tensor/fashion_crop", help="Path to the dataset")
parser.add_argument('--poses_data_root', default="fashion_dataset/fashion_poses", help="Path to the dataset")
parser.add_argument('--motion_text_root', default="fashion_dataset/train", help="Path to the tensors of latent space")
parser.add_argument('--output_dir', default="checkpoint", help="Path to save the checkpoints")
parser.add_argument('--batchsize', type=int, default=8, help="batchsize")
# parser.add_argument("--pretrained_model", default="pretrained_model/FashionFlow_checkpoint.pth", help="Path to a pretrained model")
parser.add_argument("--pretrained_model", default="pretrained_model/...", help="Path to a pretrained model")
parser.add_argument("--project", type=str, default="Textfineturning")
args = parser.parse_args()

args = parser.parse_args()

torch.distributed.init_process_group(backend="nccl", init_method="env://")
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)
synchronize()

frameLimit = 24
# pose lamda
# lamda_pose = 0.0001
lamda_pose = 0.
project_name = args.project
if get_rank() == 0:
    time_stamp = "{0:%m-%dT%H:%M:%S}".format(datetime.now())
    writer = SummaryWriter(logdir='video_progress/'+ time_stamp + project_name, comment='fashion')

def cosine_beta_schedule(timesteps, start=0.0001, end=0.02):
    betas = []
    for i in reversed(range(timesteps)):
        T = timesteps - 1
        beta = start + 0.5 * (end - start) * (1 + np.cos((i / T) * np.pi))
        betas.append(beta)
    return torch.Tensor(betas)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t): 
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

T = 1000
betas = cosine_beta_schedule(timesteps=T)
# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# vae = AutoencoderKL.from_pretrained(
#             "CompVis/stable-diffusion-v1-4",
#             subfolder="vae",
#             revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
#         ).to(device)

vae = AutoencoderKL.from_pretrained("pretrained_model/sd-vae-ft-mse").to(device)
vae.requires_grad_(False)

text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
text_encoder.requires_grad_(False)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

@torch.no_grad()
def VAE_encode(image):
    init_latent_dist = vae.encode(image).latent_dist.sample()
    init_latent_dist *= 0.18215
    encoded_image = (init_latent_dist).unsqueeze(1)
    return encoded_image

# unet
unet = SpaceTimeUnet(
    dim = 64,
    channels = 4,
    dim_mult = (1, 2, 4, 8),
    temporal_compression = (False, False, False, True),
    self_attns = (False, False, False, True),
    condition_on_timestep = True,
)

# app_cond
app_net = App2DConditioningEmbedding(
                conditioning_channels=4,
                conditioning_embedding_channels=512,
                )

# pose_cond
pose_net = PoseConditioningEmbedding_(
                conditioning_embedding_channels=64, 
                )

# pipeline
# pipe = MotionControlPipeline(text_encoder,
#                                  tokenizer,
#                                  vae=vae,
#                                  app_net=app_net,
#                                  pose_net=pose_net,
#                                  unet=unet)


unet.to(device)
app_net.to(device)
pose_net.to(device)
# pipe.to(device)

parameters = list(unet.parameters()) + list(app_net.parameters()) + list(pose_net.parameters())
# parameters = list(unet.parameters()) + list(pose_net.parameters())
# parameters = list(unet.parameters())

optimizerG = optim.AdamW(parameters, lr=0.0001, weight_decay=0.01)

if args.pretrained_model is not None:
    checkpoint = torch.load(args.pretrained_model)
    
    unet.load_state_dict(checkpoint['unet'])
    pose_net.load_state_dict(checkpoint['pose_net'])
    app_net.load_state_dict(checkpoint['app_net'])
    optimizerG.load_state_dict(checkpoint['opt'])
    start_epoch = checkpoint['epoch']
    # start_epoch = 300
    print('load success:', start_epoch)
    del checkpoint


# if args.pretrained_model is not None:
#     checkpoint = torch.load(args.pretrained_model)
#     unet.load_state_dict(checkpoint['net'])
    

unet = nn.parallel.DistributedDataParallel(
        unet,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True)

app_net = nn.parallel.DistributedDataParallel(
        app_net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True)

pose_net = nn.parallel.DistributedDataParallel(
        pose_net,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True)
# find_unused_parameters=True


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)
    

train_dataset = FashionDataset(
        video_data_root=args.video_data_root,
        video_tensor_root=args.video_tensor_root,
        poses_data_root=args.poses_data_root,
        motion_text_root=args.motion_text_root,
        frameLimit=frameLimit
)
sampler = data_sampler(train_dataset, shuffle=True, distributed=True)
batch = args.batchsize
train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch,
            sampler=sampler,
            num_workers=1,
            drop_last=True)

def save_video_frames_as_mp4(frames, fps, save_path):
    frame_h, frame_w = frames[0].shape[2:]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))
    frames = frames[0]
    for frame in frames:
        frame = np.array(TVF.to_pil_image(frame))
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

mseloss = torch.nn.MSELoss(reduction="mean")

def get_loss(input_image, latent_video, pose, prompt):
    timesteps = torch.randint(0, T, (batch,), device=device)
    timesteps = timesteps.long()

    initial_frame_latent_video = latent_video[:, 0:1].clone().detach() # [b, f, c, h, w]

    x_noisy, noise = forward_diffusion_sample(latent_video, timesteps)

    x_noisy[:, 0:1] = initial_frame_latent_video
    noise[:, 0:1] = torch.zeros(initial_frame_latent_video.shape)

    # x_noisy = x_noisy.permute(0, 2, 1, 3, 4)
    x_noisy = rearrange(x_noisy, 'b f c h w -> b c f h w')
    pose = rearrange(pose, 'b f c h w -> b c f h w')
    
    # app_resiual
    input_image = vae.encode(image).latent_dist.sample() * 0.18215
    # input_image = input_image.unsqueeze(2)
    # input_image = input_image.repeat(1, 1, x_noisy.shape[2], 1, 1)
    down_res, mid_res = app_net(input_image, x_noisy)
    # pose
    pose_embed = pose_net(pose)

    # noise priori
    # x_noisy += lamda_pose * pose_embed

    # text_embeding
    with torch.no_grad():
        prompt_ids = tokenizer(
            prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        text_embeding = text_encoder(prompt_ids)[0]

    noise_pred = unet(x_noisy, text_embeding, pose_embed, down_res, mid_res, timestep=timesteps.float())

    noise_pred = noise_pred.permute(0, 2, 1, 3, 4)
    loss = 0.0
    # 单独计算每一帧
    for i in range(noise_pred.shape[1]):
        loss += mseloss(noise_pred[:, i, :, :, :], noise[:, i, :, :, :])
    
    # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
    return loss

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
def sample_timestep(x_noisy, pose, image, prompt, t):
    betas_t = get_index_from_list(betas, t, x_noisy.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x_noisy.shape)

    # x_noisy：b f c h w
    # x_noisy = rearrange(x_noisy, 'b f c h w -> b c f h w')
    # Call model (current image - noise prediction)
    with torch.cuda.amp.autocast():
        # sample_output = pipe(x_noisy=x.permute(0, 2, 1, 3, 4), 
        #                     prompt=prompt,
        #                     image=image,
        #                     pose=pose.permute(0, 2, 1, 3, 4),
        #                     timestep=t.float(),
        #                     lamda_pose=lamda_pose)
        # text_embeding
        with torch.no_grad():
            prompt_ids = tokenizer(
                prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(device)
            text_embeding = text_encoder(prompt_ids)[0]
        
        # app_resiual
        input_image = vae.encode(image).latent_dist.sample() * 0.18215
        # input_image = input_image.unsqueeze(1)
        # input_image = input_image.repeat(1, x_noisy.shape[1], 1, 1, 1)
        # down_res, mid_res = app_net(input_image.permute(0, 2, 1, 3, 4), x_noisy.permute(0, 2, 1, 3, 4))
        down_res, mid_res = app_net(input_image, x_noisy.permute(0, 2, 1, 3, 4))
        # pose
        pose_embed = pose_net(pose.permute(0, 2, 1, 3, 4))

        # noise priori
        # x_noisy += lamda_pose * pose_embed.permute(0, 2, 1, 3, 4)

        sample_output = unet(x_noisy.permute(0, 2, 1, 3, 4), 
                             text_embeding,
                             pose_embed,
                             down_res, mid_res, timestep=t.float())

        sample_output = sample_output.permute(0, 2, 1, 3, 4)

        # 输出和输入都是 f 在 c 后面
        # 而计算时 f 在 c 前面
        # x_noisy = x_noisy.permute(0, 2, 1, 3, 4)
    
    model_mean = sqrt_recip_alphas_t * (
            x_noisy - betas_t * sample_output / sqrt_one_minus_alphas_cumprod_t
    )
    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_noisy)
        posterior_variance_t = get_index_from_list(posterior_variance, t, x_noisy.shape)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


training_sample = "training_sample" + project_name
if not os.path.exists(training_sample):
    os.makedirs(training_sample)

# start_epoch = 101
start_epoch += 1
end_epoch = 2500

step = 0
for epoch in range(start_epoch, end_epoch):
    app_net.train()
    pose_net.train()
    unet.train()
    for data in train_dataloader:
        step += 1
        vae_video = data['video'].to(device=device) # [b, f, c, h, w]
        image = data['image'].to(device=device)
        pose = data['pose'].to(device=device) # b, f, c, h w
        prompt = data['mt_txt']

        # for i in range(len(prompt)):
        #     prompt[i] = ""

        # for i in range(pose.shape[0]):
        pose = pose[0].unsqueeze(0)
        # print(pose.shape)
        pose = torch.cat([pose] * batch, dim=0)
        # print(pose.shape)
            
        
        loss = get_loss(input_image=image, latent_video=vae_video, pose=pose, prompt=prompt)
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()

    if get_rank() == 0:
        print(f"# epoch : {epoch} / 2500; loss: {loss}")

    if get_rank() == 0 and epoch % 100 == 0:
        writer.add_scalar('loss', loss, epoch)
        # print(f"# epoch : {epoch} / 2500; step ：{step} ")

    if get_rank() == 0 and epoch % 50 == 0:
        torch.save(
            {
                'app_net': app_net.module.state_dict(),
                'pose_net': pose_net.module.state_dict(),
                'unet': unet.module.state_dict(),
                'opt': optimizerG.state_dict(),
                'epoch': epoch
            # }, args.output_dir + "/model_" + str(epoch) + "_" + str(step) + ".pth")
            }, args.output_dir + "/model_text_fineturning" + str(epoch) + "_" + str(step) + ".pth")
    if get_rank() == 0 and epoch % 100 == 0:
        noise_video = torch.randn([1, frameLimit, 4, 64, 32]).to(device)
        prompt = prompt[0]
        pose = pose[0:1, :, :, :, :].to(device)
        image = image[0:1, :, :, :].to(device)
        # encoder_hidden_states = get_image_embedding(input_image=image[0].unsqueeze(0))

        encoded_image = image[0].unsqueeze(0)
        encoded_image = VAE_encode(image[0].unsqueeze(0))
        noise_video[:, 0:1] = encoded_image
        with torch.no_grad():
            for i in range(0, T)[::-1]:
                t = torch.full((1,), i, device=device).long()
                noise_video = sample_timestep(x_noisy=noise_video, pose=pose, image=image, prompt=prompt, t=t)
                noise_video[:, 0:1] = encoded_image
            final_video = VAE_decode(noise_video)
        writer.add_image('input image', image[0], epoch)

        # final_video = torch.cat([final_video, pose], 0)
        writer.add_video('gen_video', final_video, epoch)
        writer.add_video('pose_video', pose, epoch)
        writer.add_text('motion_text', prompt, epoch)
        save_video_frames_as_mp4(final_video, 16, training_sample+"/video"+str(epoch)+".mp4")
if get_rank() == 0:
    torch.save(
            {
                'app_net': app_net.module.state_dict(),
                'pose_net': pose_net.module.state_dict(),
                'unet': unet.module.state_dict(),
                'opt': optimizerG.state_dict(),
            }, args.output_dir + "/motion_control_e100.pth")
    

