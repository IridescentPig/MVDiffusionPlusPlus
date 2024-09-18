import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from diffusers import UNet2DConditionModel

UNET_CONFIG = {
    "act_fn": "silu",
    "attention_head_dim": [
        5,
        10,
        20,
        20
    ],
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": 1024,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 9,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ],
    "use_linear_projection": True
}

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.to_out.weight.data.fill_(0)
        self.to_out.bias.data.fill_(0)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        q = self.to_q(x)

        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return self.drop_out(out)

class MultiViewUNet(nn.Module):
    def __init__(self, unet = None):
        super().__init__()

        if unet is None:
            self.unet = UNet2DConditionModel.from_config(UNET_CONFIG)
        else:
            self.unet = unet
        self.Vs = torch.nn.Parameter(torch.zeros(42, 320))
        self.s = torch.nn.Parameter(torch.zeros(1))
        # self.conv = nn.Conv2d(9, 4, 1)
        self.global_self_attn_downblocks = nn.ModuleList()
        for i in range(len(self.unet.down_blocks)):
            dim = self.unet.down_blocks[i].resnets[-1].out_channels
            # TODO: read from config
            num_heads = self.unet.down_blocks[i].attentions[-1].num_attention_heads
            attention_head_dim = self.unet.down_blocks[i].attentions[-1].attention_head_dim
            self.global_self_attn_downblocks.append(
                SelfAttention(
                    dim=dim,
                    heads=num_heads,
                    dim_head=attention_head_dim,
                )
            )
        
        self.global_self_attn_midblock = \
            SelfAttention(
                dim=self.unet.mid_block.resnets[-1].out_channels,
                heads=self.unet.mid_block.attentions[-1].num_attention_heads,
                dim_head=self.unet.mid_block.attentions[-1].attention_head_dim,
            )
        
        self.global_self_attn_upblocks = nn.ModuleList()
        for i in range(len(self.unet.up_blocks)):
            # dim = self.unet.up_blocks[i].resnets[-1].out_channels
            # num_heads = dim // 64
            num_heads = self.unet.up_blocks[i].attentions[-1].num_attention_heads
            attention_head_dim = self.unet.up_blocks[i].attentions[-1].attention_head_dim
            self.global_self_attn_upblocks.append(
                SelfAttention(
                    dim=dim,
                    heads=num_heads,
                    dim_head=attention_head_dim,
                )
            )

        self.trainable_parameters = \
            [(list(self.unet.parameters()) + \
              list(self.global_self_attn_downblocks.parameters()) + \
              list(self.global_self_attn_midblock.parameters()) + \
              list(self.global_self_attn_upblocks.parameters()) + \
              [self.Vs] + [self.s], 1.0)]

        self.trainable_parameters += [(list(self.unet.parameters()), 0.01)]

    
    def forward(self, latents, timestep, prompt_embd, idxs):
        b, m, c, h, w = latents.shape

        # bs*m, 9, 64, 64
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps

        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs*m, 320)
        emb = self.unet.time_embedding(t_emb)  # (bs*m, 1280)
        idxs = idxs.reshape(-1) # (bs*m)
        img_pos_emb = self.Vs[idxs] # (bs*m, 320)
        img_pos_emb = self.unet.time_embedding(img_pos_emb) # (bs*m, 1280)
        emb = emb + self.s * img_pos_emb


        # hidden_states = self.conv(hidden_states) # (bs*m, 4, 64, 64)
        hidden_states = self.unet.conv_in(hidden_states) # (bs*m, 320, 64, 64)

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states, emb)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
            if m > 1:
                _, _, h, w = hidden_states.shape
                hidden_states = rearrange(hidden_states, '(b m) c h w -> b (m h w) c', m=m)
                hidden_states = self.global_self_attn_downblocks[i](hidden_states)
                hidden_states = rearrange(hidden_states, 'b (m h w) c -> (b m) c h w', m=m, h=h, w=w)

            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid

        hidden_states = self.unet.mid_block.resnets[0](
            hidden_states, emb)

        if m > 1:
            _, _, h, w = hidden_states.shape
            hidden_states = rearrange(hidden_states, '(b m) c h w -> b (m h w) c', m=m)
            hidden_states = self.global_self_attn_midblock(hidden_states)
            hidden_states = rearrange(hidden_states, 'b (m h w) c -> (b m) c h w', m=m, h=h, w=w)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)

        h, w = hidden_states.shape[-2:]

        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = resnet(hidden_states, emb)
            if m > 1:
                _, _, h, w = hidden_states.shape
                hidden_states = rearrange(hidden_states, '(b m) c h w -> b (m h w) c', m=m)
                hidden_states = self.global_self_attn_upblocks[i](hidden_states)
                hidden_states = rearrange(hidden_states, 'b (m h w) c -> (b m) c h w', m=m, h=h, w=w)

            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
