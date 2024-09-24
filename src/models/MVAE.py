from diffusers import AutoencoderKL
import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import numpy as np
from PIL import Image

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

class MaskVAE(pl.LightningModule):
    def __init__(self, model_config, train_config):
        super().__init__()
        self.model = AutoencoderKL.from_config(model_config)
        self.bce_loss = nn.BCELoss()
        self.learning_rate = train_config.get('learning_rate', 4.5e-6)
        torch.nn.init.xavier_uniform_(self.model.encoder.conv_in.weight)
        torch.nn.init.xavier_uniform_(self.model.decoder.conv_out.weight)

    def load_pretrained_vae(self, path):
        state_dict = torch.load(path, map_location='cpu')
        print(state_dict.keys())
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('encoder.conv_in') and not k.startswith('decoder.conv_out')}
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x, sample_posterior=True):
        posterior = self.model.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.model.decode(z).sample
        return dec, posterior
    
    def loss(self, inputs, posterior, reconstructions):
        # kl_loss = posterior.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        rgb_inputs = inputs[:, :3] # (B, 3, H, W)
        rgb_reconstructions = reconstructions[:, :3] # (B, 3, H, W)
        recon_loss = torch.abs(rgb_inputs.contiguous() - rgb_reconstructions.contiguous())
        recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]
        mask_inputs = inputs[:, 3] # (B, 1, H, W)
        mask_reconstructions = reconstructions[:, 3] # (B, 1, H, W)
        mask_reconstructions = mask_reconstructions.clamp(0, 1) # -> [0,1]
        mask_loss = self.bce_loss(mask_reconstructions, mask_inputs)
        # mask_loss = torch.sum(mask_loss) / mask_loss.shape[0]
        loss = recon_loss + mask_loss
        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('mask_loss', mask_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.model.encoder.parameters())+
                                list(self.model.decoder.parameters())+
                                list(self.model.quant_conv.parameters())+
                                list(self.model.post_quant_conv.parameters()),
                                lr=lr, betas=(0.5, 0.9))
        return {'optimizer': optimizer}
    
    def training_step(self, batch, batch_idx):
        inputs = batch['images'] # (B, 4, H, W)
        reconstructions, posterior = self(inputs)
        loss = self.loss(inputs, posterior, reconstructions)
        self.log('train_loss', loss, prog_bar=True)
        # self.log('train_recon_loss', recon_loss)
        # self.log('train_kl_loss', kl_loss)
        # self.log('train_mask_loss', mask_loss)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images = batch['images']
        reconstructions, posterior = self(images)
        mask = images[:, 3:] # (B, 1, H, W)
        mask = mask * 255. # [0, 1] -> [0, 255]
        images = images[:, :3] # (B, 3, H, W)
        images = ((images / 2+ 0.5) * 255) # (B, 3, H, W), [0, 255]
        images = torch.cat([images, mask], dim=1).permute(0, 2, 3, 1) # (B, H, W, 4)
        images = images.cpu().numpy().astype(np.uint8) # (B, H, W, 4)

        mask_rec = reconstructions[:, 3:] # (B, 1, H, W)
        rgb_rec = reconstructions[:, :3] # (B, 3, H, W)
        mask_rec = mask_rec.clamp(0, 1) * 255. # [0, 1] -> [0, 255]
        rgb_rec = (rgb_rec / 2 + 0.5).clamp(0, 1) * 255. # [0, 1] -> [0, 255]
        images_rec = torch.cat([rgb_rec, mask_rec], dim=1).permute(0, 2, 3, 1) # (B, H, W, 4)
        images_rec = images_rec.cpu().float().numpy().astype(np.uint8) # (B, H, W, 4)
        if self.trainer.global_rank == 0:
            self.save_image(images, images_rec, batch_idx)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images = batch['images']
        reconstructions, posterior = self(images)
        mask = images[:, 3:] # (B, 1, H, W)
        mask = mask * 255. # [0, 1] -> [0, 255]
        images = images[:, :3] # (B, 3, H, W)
        images = ((images / 2+ 0.5) * 255) # (B, 3, H, W), [0, 255]
        images = torch.cat([images, mask], dim=1).permute(0, 2, 3, 1) # (B, H, W, 4)
        images = images.cpu().numpy().astype(np.uint8) # (B, H, W, 4)

        mask_rec = reconstructions[:, 3:] # (B, 1, H, W)
        rgb_rec = reconstructions[:, :3] # (B, 3, H, W)
        mask_rec = mask_rec.clamp(0, 1) * 255. # [0, 1] -> [0, 255]
        rgb_rec = (rgb_rec / 2 + 0.5).clamp(0, 1) * 255. # [0, 1] -> [0, 255]
        images_rec = torch.cat([rgb_rec, mask_rec], dim=1).permute(0, 2, 3, 1) # (B, H, W, 4)
        images_rec = images_rec.cpu().float().numpy().astype(np.uint8) # (B, H, W, 4)
        image_id = batch['image_id'][0]
        
        output_dir = batch['output_dir'][0] if 'output_dir' in batch else os.path.join(self.logger.log_dir, 'images')
        output_dir = os.path.join(output_dir, "{}".format(image_id))
        os.makedirs(output_dir, exist_ok=True)
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i])
            img_rec = Image.fromarray(images_rec[i])
            img.save(os.path.join(output_dir, f'{batch_idx}_{i}_gt.png'))
            img_rec.save(os.path.join(output_dir, f'{batch_idx}_{i}_rec.png'))

    def save_image(self, images, images_rec, batch_idx):
        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        for i in range(images.shape[0]):
            img = Image.fromarray(images[i])
            img_rec = Image.fromarray(images_rec[i])
            img.save(os.path.join(img_dir, f'{self.global_step}_{batch_idx}_{i}_gt.png'))
            img_rec.save(os.path.join(img_dir, f'{self.global_step}_{batch_idx}_{i}_rec.png'))
