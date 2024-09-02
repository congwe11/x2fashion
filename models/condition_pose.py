
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from einops import rearrange

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x
    

class PoseConditioningEmbedding_(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # InflatedConv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            # nn.SiLU(),
            InflatedConv3d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            # InflatedConv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.SiLU(),
            InflatedConv3d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            # InflatedConv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.SiLU(),
            InflatedConv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            # InflatedConv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.SiLU(),
            InflatedConv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

        self.final_proj = InflatedConv3d(in_channels=128, out_channels=conditioning_embedding_channels, kernel_size=1)

        # Initialize layers
        self._initialize_weights()

        self.scale = nn.Parameter(torch.ones(1) * 2)


    def _initialize_weights(self):
        # Initialize weights with He initialization and zero out the biases
        for m in self.conv_layers:
            if isinstance(m, InflatedConv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias)

        # For the final projection layer, initialize weights to zero (or you may choose to use He initialization here as well)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_proj(x)

        return x * self.scale

class PoseConditioningEmbedding(nn.Module):
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
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()

        self.conv_in = InflatedConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # self.blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=4, padding=2))
            self.blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=4, padding=1, stride=2))

        self.conv_out = zero_module(
            InflatedConv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )
        # self.conv_out = InflatedConv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module



if __name__ == "__main__":
    block_out_channels = (16, 32, 64, 128)
    # net = PoseConditioningEmbedding(
    #     conditioning_embedding_channels=4, 
    #     block_out_channels=block_out_channels)
    
    net = PoseConditioningEmbedding_(
        conditioning_embedding_channels=64, 
        block_out_channels=block_out_channels)
    
    x = torch.randn([2, 3, 16, 512, 256])

    x = net(x)
    print(x.shape)