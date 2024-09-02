
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


from diffusers import AutoencoderKL
    

vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
        )
vae.requires_grad_(False)


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

class AppConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 512,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (64, 128, 256, 512, 512),
    ):
        super().__init__()

        self.zero_conv = zero_module(
            InflatedConv3d(conditioning_channels, conditioning_channels, kernel_size=1)
        )
        self.conv_in = InflatedConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([])

        # self.zero_blocks = nn.ModuleList([])
        # zero_conv_in = zero_module(InflatedConv3d(block_out_channels[0], block_out_channels[0], kernel_size=1))
        # self.zero_blocks.append(zero_conv_in)

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

            # self.zero_blocks.append(zero_module(InflatedConv3d(channel_in, channel_in, kernel_size=1)))
            # self.zero_blocks.append(zero_module(InflatedConv3d(channel_out, channel_out, kernel_size=1)))

        
        # self.conv_out = zero_module(
        #     nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        # )

        self.conv_out = InflatedConv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        # self.zero_out = zero_module(
        #     InflatedConv3d(conditioning_embedding_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        # )

    def forward(self, conditioning, zt):

        
        embedding = self.zero_conv(conditioning)
        
        embedding += zt

        embedding = self.conv_in(conditioning)

        # down_block_additional_residual = []
        down_block_additional_residual = (embedding,)
        for block in self.blocks:
            
            embedding = block(embedding)
            embedding = F.silu(embedding)

            down_block_additional_residual += (embedding,)

        embedding = self.conv_out(embedding)
        embedding = F.silu(embedding)

        # new_down_block_additional_residual = ()
        # for down, zero_block in zip(down_block_additional_residual, self.zero_blocks):
        #     down = zero_block(down)
        #     new_down_block_additional_residual += (down,)

        # down_block_additional_residual = new_down_block_additional_residual
        down_block_additional_residual = down_block_additional_residual[:len(down_block_additional_residual) - 1]


        # mid_block_additional_residual = self.zero_out(embedding)

        mid_block_additional_residual = embedding

        return down_block_additional_residual, mid_block_additional_residual
    

class App2DConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """


    def __init__(
        self,
        conditioning_embedding_channels: int = 512,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (64, 128, 256, 512, 512),
    ):
        super().__init__()

        self.zero_conv = zero_module(
            nn.Conv2d(conditioning_channels, conditioning_channels, kernel_size=1)
        )
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList([])

        self.zero_blocks = nn.ModuleList([])
        zero_conv_in = zero_module(nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=1))
        self.zero_blocks.append(zero_conv_in)

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=4, padding=1, stride=2))

            self.zero_blocks.append(zero_module(nn.Conv2d(channel_in, channel_in, kernel_size=1)))
            self.zero_blocks.append(zero_module(nn.Conv2d(channel_out, channel_out, kernel_size=1)))

        
        # self.conv_out = zero_module(
        #     nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        # )

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        self.zero_out = zero_module(
            nn.Conv2d(conditioning_embedding_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning, zt):

        
        # embedding = self.zero_conv(conditioning)
        
        # embedding += zt[:, :, 0, :, :]

        embedding = self.conv_in(conditioning)

        # down_block_additional_residual = []
        down_block_additional_residual = (embedding,)
        for block in self.blocks:
            
            embedding = block(embedding)
            embedding = F.silu(embedding)

            down_block_additional_residual += (embedding,)

        embedding = self.conv_out(embedding)
        embedding = F.silu(embedding)

        new_down_block_additional_residual = ()
        for down, zero_block in zip(down_block_additional_residual, self.zero_blocks):
            down = zero_block(down)
            down = down.unsqueeze(2)
            down = down.repeat(1,1,zt.shape[2],1,1)
            new_down_block_additional_residual += (down,)

        down_block_additional_residual = new_down_block_additional_residual
        down_block_additional_residual = down_block_additional_residual[:len(down_block_additional_residual) - 1]


        mid_block_additional_residual = self.zero_out(embedding)
        mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
        mid_block_additional_residual = mid_block_additional_residual.repeat(1,1,zt.shape[2],1,1)

        return down_block_additional_residual, mid_block_additional_residual



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module



if __name__ == "__main__":
    block_out_channels = (64, 128, 256, 512, 512)
    # net = AppConditioningEmbedding(
    #     conditioning_channels=4,
    #     conditioning_embedding_channels=512, 
    #     block_out_channels=block_out_channels)
    
    # zt = torch.rand([1, 4, 16, 64, 32])
    # x = torch.randn([1, 3, 512, 256])
    # x = vae.encode(x).latent_dist.sample()
    # x = x.unsqueeze(2)
    # x = x.repeat(1, 1, zt.shape[2], 1, 1)
    

    net = App2DConditioningEmbedding(
        conditioning_channels=4,
        conditioning_embedding_channels=512, 
        block_out_channels=block_out_channels)
    
    zt = torch.rand([1, 4, 16, 64, 32])
    x = torch.randn([1, 3, 512, 256])
    x = vae.encode(x).latent_dist.sample()
    # x = x.unsqueeze(2)
    # x = x.repeat(1, 1, zt.shape[2], 1, 1)

    down, mid = net(x, zt)

    print(x.shape)