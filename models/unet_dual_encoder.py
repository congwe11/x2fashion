# Load pretrained 2D UNet and modify with temporal attention
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange

from diffusers.models import UNet2DConditionModel

def get_unet(pretrained_model_name_or_path, revision, resolution=256, n_poses=5):
    # Load pretrained UNet layers
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c",
        cache_dir="checkpoints/unet")

    # Modify input layer to have 1 additional input channels (pose)
    weights = unet.conv_in.weight.clone()
    unet.conv_in = nn.Conv2d(4 + 2*n_poses, weights.shape[0], kernel_size=3, padding=(1, 1)) # input noise + n poses
    with torch.no_grad():
        unet.conv_in.weight[:, :4] = weights # original weights
        unet.conv_in.weight[:, 3:] = torch.zeros(unet.conv_in.weight[:, 3:].shape) # new weights initialized to zero

    return unet

'''
    This module takes in CLIP + VAE embeddings and outputs CLIP-compatible embeddings.
'''
class Embedding_Adapter(nn.Module):
    def __init__(self, input_nc=38, output_nc=4, norm_layer=nn.InstanceNorm2d, chkpt=None):
        super(Embedding_Adapter, self).__init__()

        self.save_method_name = "adapter"

        self.pool =  nn.MaxPool2d(2)
        # self.vae2clip = nn.Linear(1280, 768)
        self.vae2clip = nn.Linear(512, 768)


        self.linear1 = nn.Linear(54, 50) # 50 x 54 shape

        # initialize weights
        with torch.no_grad():
            self.linear1.weight = nn.Parameter(torch.eye(50, 54))

        if chkpt is not None:
            pass

    def forward(self, clip, vae):
        
        vae = self.pool(vae) # 1 4 80 64 --> 1 4 40 32
        vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 20 16 --> 1 4 1280

        vae = self.vae2clip(vae) # 1 4 768

        # Concatenate
        concat = torch.cat((clip, vae), 1)

        # Encode

        concat = rearrange(concat, 'b c d -> b d c')
        concat = self.linear1(concat)
        concat = rearrange(concat, 'b d c -> b c d')

        return concat
    

class Embedding_Adapter_text(nn.Module):
    def __init__(self, input_nc=38, output_nc=4, norm_layer=nn.InstanceNorm2d, chkpt=None):
        super(Embedding_Adapter_text, self).__init__()

        self.save_method_name = "adapter"

        self.linear1 = nn.Linear(127, 77) # 127 x 77 shape

        # initialize weights
        with torch.no_grad():
            self.linear1.weight = nn.Parameter(torch.eye(77, 127))

        if chkpt is not None:
            pass

    def forward(self, emb, text_emb):

        # Concatenate
        concat = torch.cat((emb, text_emb), 1)

        # Encode
        concat = rearrange(concat, 'b c d -> b d c')
        concat = self.linear1(concat)
        concat = rearrange(concat, 'b d c -> b c d')

        return concat



if __name__ == '__main__':
    Net = Embedding_Adapter()
    Net2 = Embedding_Adapter_text()

    x = torch.randn([1,4,64,32])
    emb = torch.randn([1,50,768])
    sample_output = Net(emb, x)

    emb_text = torch.randn([1, 77, 768])
    out = Net2(sample_output, emb_text)