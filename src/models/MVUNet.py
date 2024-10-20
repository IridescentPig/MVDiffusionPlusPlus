import torch
import torch.backends
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import TimestepEmbedding

UNET_CONFIG = {
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": 768,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
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
    ]
}

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.to_out.weight.data.fill_(0)
        self.to_out.bias.data.fill_(0)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        h = self.heads

        q = self.to_q(x) # b n (h d)
        k = self.to_k(x) # b n (h d)
        v = self.to_v(x) # b n (h d)

        q = q.reshape(b, n, h, -1).transpose(1, 2) # b h n d
        k = k.reshape(b, n, h, -1).transpose(1, 2) # b h n d
        v = v.reshape(b, n, h, -1).transpose(1, 2) # b h n d

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, 
            enable_math=False, 
            enable_mem_efficient=False
        ):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_out.p) # b h n d
        
        # out = rearrange(out, 'b h n d -> b n (h d)') # b n c
        out = out.transpose(1, 2).reshape(b, n, -1) # b n c
        out = self.to_out(out)
        return self.drop_out(out)

class MultiViewUNet(nn.Module):
    def __init__(self, unet = None):
        super().__init__()

        if unet is None:
            self.unet = UNet2DConditionModel.from_config(UNET_CONFIG)
        else:
            self.unet = unet

        self.index_embed_dim = 320
        self.Vs = torch.nn.Embedding(9, self.index_embed_dim)
        self.index_embedding = TimestepEmbedding(self.index_embed_dim, 1280)
        # self.index_proj = torch.nn.Linear(50, 320)
        self.s = torch.nn.Parameter(torch.zeros(1))
        self.global_self_attn_downblocks = nn.ModuleList()
        if isinstance(self.unet.config['attention_head_dim'], list):
            num_attention_heads = self.unet.config['attention_head_dim']
        else:
            num_attention_heads = [self.unet.config['attention_head_dim']] * len(self.unet.down_blocks)
        for i in range(len(self.unet.down_blocks)):
            dim = self.unet.down_blocks[i].resnets[0].in_channels
            # TODO: read from config
            num_heads = num_attention_heads[i]
            attention_head_dim = dim // num_heads
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
                heads=num_attention_heads[-1],
                dim_head=dim // num_attention_heads[-1],
            )
        
        self.global_self_attn_upblocks = nn.ModuleList()
        num_attention_heads = num_attention_heads[::-1]
        reversed_output_channels = self.unet.config['block_out_channels'][::-1]
        output_channels = reversed_output_channels[0]
        for i in range(len(self.unet.up_blocks)):
            previous_output_channels = output_channels
            output_channels = reversed_output_channels[i]
            dim = previous_output_channels
            # num_heads = dim // 64
            num_heads = num_attention_heads[i]
            attention_head_dim = dim // num_heads
            self.global_self_attn_upblocks.append(
                SelfAttention(
                    dim=dim,
                    heads=num_heads,
                    dim_head=attention_head_dim,
                )
            )

        self.trainable_parameters = [(list(self.global_self_attn_downblocks.parameters()), 1.0)]
        self.trainable_parameters += [(list(self.global_self_attn_midblock.parameters()), 1.0)]
        self.trainable_parameters += [(list(self.global_self_attn_upblocks.parameters()), 1.0)]
        # self.trainable_parameters += [(list(self.index_proj.parameters()), 1.0)]
        self.trainable_parameters += [(list(self.index_embedding.parameters()), 1.0)]
        self.trainable_parameters += [(list(self.Vs.parameters()), 1.0)]
        self.trainable_parameters += [(self.s, 1.0)]
        self.trainable_parameters += [(list(self.unet.parameters()), 1.0)]

    def forward(self, latents, timestep, prompt_embd, idxs):
        b, m, c, h, w = latents.shape
        _, _, l, c_prompt = prompt_embd.shape

        # (b, m, c, h, w) -> (b*m, c, h, w), c=9, h=w=64
        # hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        hidden_states = latents.reshape(-1, c, h, w)
        # (b, m, l, c) -> (b*m, l, c)
        # prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')
        prompt_embd = prompt_embd.reshape(-1, l, c_prompt)

        # 1. process timesteps and index embeddings
        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep)  # (bs*m, 320)
        t_emb = self.unet.time_embedding(t_emb)  # (bs*m, 1280)
        idxs = idxs.reshape(-1) # (bs*m)
        img_idx_emb = self.Vs(idxs) # (bs*m, 320)
        # img_idx_emb = self.index_proj(img_idx_emb) # (bs*m, 320)
        img_idx_emb = self.index_embedding(img_idx_emb)
        emb = t_emb + self.s * img_idx_emb

        # (b*m, c, h, w), c=320, h=w=64
        hidden_states = self.unet.conv_in(hidden_states)
        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if m > 1:
                _, c, h, w = hidden_states.shape
                # (b*m, c, h, w) -> (b, m, c, h, w) -> (b, m, h, w, c) -> (b, m*h*w, c)
                # hidden_states = rearrange(hidden_states, '(b m) c h w -> b (m h w) c', m=m)
                hidden_states = hidden_states.reshape(b, m, c, h, w).permute(0, 1, 3, 4, 2).reshape(b, m*h*w, c)
                hidden_states = self.global_self_attn_downblocks[i](hidden_states)
                # (b, m*h*w, c) -> (b, m, h, w, c) -> (b, m, c, h, w) -> (b*m, c, h, w)
                # hidden_states = rearrange(hidden_states, 'b (m h w) c -> (b m) c h w', m=m, h=h, w=w)
                hidden_states = hidden_states.reshape(b, m, h, w, c).permute(0, 1, 4, 2, 3).reshape(b*m, c, h, w)

            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for resnet, attn in zip(downsample_block.resnets, downsample_block.attentions):
                    hidden_states = resnet(hidden_states.contiguous(), emb)

                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for resnet in downsample_block.resnets:
                    hidden_states = resnet(hidden_states, emb)
                    down_block_res_samples += (hidden_states,)
        
            if downsample_block.downsamplers is not None:
                for downsample in downsample_block.downsamplers:
                    hidden_states = downsample(hidden_states)
                down_block_res_samples += (hidden_states,)

        # b. mid
        hidden_states = self.unet.mid_block.resnets[0](hidden_states, emb)
        if m > 1:
            _, c, h, w = hidden_states.shape
            # (b*m, c, h, w) -> (b, m, c, h, w) -> (b, m, h, w, c) -> (b, m*h*w, c)
            # hidden_states = rearrange(hidden_states, '(b m) c h w -> b (m h w) c', m=m)
            hidden_states = hidden_states.reshape(b, m, c, h, w).permute(0, 1, 3, 4, 2).reshape(b, m*h*w, c)
            hidden_states = self.global_self_attn_midblock(hidden_states)
            # (b, m*h*w, c) -> (b, m, h, w, c) -> (b, m, c, h, w) -> (b*m, c, h, w)
            # hidden_states = rearrange(hidden_states, 'b (m h w) c -> (b m) c h w', m=m, h=h, w=w)
            hidden_states = hidden_states.reshape(b, m, h, w, c).permute(0, 1, 4, 2, 3).reshape(b*m, c, h, w)

        for attn, resnet in zip(self.unet.mid_block.attentions, self.unet.mid_block.resnets[1:]):
            hidden_states = attn(
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = resnet(hidden_states, emb)

        h, w = hidden_states.shape[-2:]
        # c. upsample
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            
            if m > 1:
                _, c, h, w = hidden_states.shape
                # (b*m, c, h, w) -> (b, m, c, h, w) -> (b, m, h, w, c) -> (b, m*h*w, c)
                # hidden_states = rearrange(hidden_states, '(b m) c h w -> b (m h w) c', m=m)
                hidden_states = hidden_states.reshape(b, m, c, h, w).permute(0, 1, 3, 4, 2).reshape(b, m*h*w, c)
                hidden_states = self.global_self_attn_upblocks[i](hidden_states)
                # (b, m*h*w, c) -> (b, m, h, w, c) -> (b, m, c, h, w) -> (b*m, c, h, w)
                # hidden_states = rearrange(hidden_states, 'b (m h w) c -> (b m) c h w', m=m, h=h, w=w)
                hidden_states = hidden_states.reshape(b, m, h, w, c).permute(0, 1, 4, 2, 3).reshape(b*m, c, h, w)

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for resnet, attn in zip(upsample_block.resnets, upsample_block.attentions):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1).contiguous()
                    hidden_states = resnet(hidden_states, emb)
                    hidden_states = attn(
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for resnet in upsample_block.resnets:
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1).contiguous()
                    hidden_states = resnet(hidden_states, emb)
            
            if upsample_block.upsamplers is not None:
                for upsample in upsample_block.upsamplers:
                    hidden_states = upsample(hidden_states)

        # 4.post-process
        sample = self.unet.conv_norm_out(hidden_states)
        sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        _, c, h, w = sample.shape
        # (b*m, c, h, w) -> (b, m, c, h, w)
        # sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        sample = sample.reshape(b, m, c, h, w)
        return sample
