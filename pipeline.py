import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, pack, unpack
from transformers import CLIPTextModel, CLIPTokenizer

from models.condition_app import AppConditioningEmbedding
from models.condition_pose import PoseConditioningEmbedding
from models.unet import SpaceTimeUnet
from diffusers import AutoencoderKL


class MotionControlPipeline(nn.Module):
    def __init__(
            self,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            vae: AutoencoderKL,
            unet: SpaceTimeUnet,
            app_net: AppConditioningEmbedding,
            pose_net: PoseConditioningEmbedding,
            ) -> None:
        super().__init__()


        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.app_net = app_net
        self.pose_net = pose_net
        self.vae = vae
        # self.register_module(text_encoder=text_encoder,
        #                      tokenizer=tokenizer)

    
    def encoder_prompt(self, prompt, device):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        return prompt_embeds

    def forward(
            self,
            x_noisy,
            prompt,
            image,
            pose,
            timestep,
            lamda_pose: float = 0.03
    ):
        
        # text_embeding
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(x_noisy.device)
            text_embeding = self.text_encoder(prompt_ids)[0]
        # text_embeding = self.encoder_prompt(prompt, x_noisy.device)

        # apperence
        image = self.vae.encode(image).latent_dist.sample() * 0.18215
        image = image.unsqueeze(2)
        image = image.repeat(1, 1, x_noisy.shape[2], 1, 1)
        down_res, mid_res = self.app_net(image, x_noisy)
        # input_image = vae.encode(image).latent_dist.sample() * 0.18215
        # input_image = input_image.unsqueeze(2)
        # input_image = input_image.repeat(1, 1, x_noisy.shape[2], 1, 1)
        # down_res, mid_res = app_net(input_image, x_noisy)
        
        # pose
        pose_emb = self.pose_net(pose)

        x_noisy += lamda_pose * pose_emb

        pred = self.unet(x_noisy, text_embeding, down_res, mid_res, timestep=timestep)

        return pred
    

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available else "cpu"

    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    text_encoder.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
        ).to(device)
    vae.requires_grad_(False)

    app_net = AppConditioningEmbedding(
                conditioning_channels=4,
                conditioning_embedding_channels=512,
                ).to(device)
    
    pose_net = PoseConditioningEmbedding(
        conditioning_embedding_channels=4, 
        ).to(device)
    
    unet = SpaceTimeUnet(
        dim = 64,
        channels = 4,
        dim_mult = (1, 2, 4, 8),
        temporal_compression = (False, False, False, True),
        self_attns = (False, False, False, True),
        condition_on_timestep = True,
    ).to(device)

    pipe = MotionControlPipeline(text_encoder,
                                 tokenizer,
                                 vae=vae,
                                 app_net=app_net,
                                 pose_net=pose_net,
                                 unet=unet)


    
    batch = 1
    T = 1000
    image = torch.randn([1, 3, 512, 256]).to(device)
    # image = vae.encode(image).x_noisy_dist.sample() * 0.18215

    x_noisy = torch.randn([1, 16, 4, 64, 32]).to(device)
    pose = torch.randn([1, 16, 3, 512, 256]).to(device)
    timesteps = torch.randint(0, T, (batch,), device=device)
    timesteps = timesteps.long()

    prompts = ["a", "b"]
    prompt = prompts[0]
    x_noisy = x_noisy.permute(0, 2, 1, 3, 4)  # b c f h w
    pose = pose.permute(0, 2, 1, 3, 4)
    noised = pipe(x_noisy, prompt, image, pose, timesteps.float())

    print(noised)


    






    