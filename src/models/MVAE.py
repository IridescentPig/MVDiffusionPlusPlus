from diffusers import AutoencoderKL
import torch
import pytorch_lightning as pl

MVAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [
        128,
        256,
        512,
        512
    ],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D"
    ],
    "in_channels": 4,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 512,
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ]
}

def load_vae(model: AutoencoderKL, path: str):
    state_dict = torch.load(path)
    state_dict = {k: v for k, v in state_dict.items() if k not in ['encoder.conv_in', 'decoder.conv_out']}
    model.load_state_dict(state_dict, strict=False)

# def MVAE():
#     model = AutoencoderKL.from_config(MVAE_CONFIG)
#     return model

class MVAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = AutoencoderKL.from_config(config)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        pass
