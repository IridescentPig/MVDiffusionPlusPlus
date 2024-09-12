import pytorch_lightning as pl
from transformers import CLIPTextModel, CLIPTokenizer
from .MVUNet import MultiViewUNet
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from PIL import Image

class MVDiffuison(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']

        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)
        # TODO: CLIP image encoder

        self.mvae, self.scheduler, unet = self.load_model(
            config['model']['model_id'])
        self.unet = MultiViewUNet(unet)
        self.trainable_params = self.unet.trainable_parameters

        self.save_hyperparameters()

    def load_model(self, model_id):
        mvae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        mvae.eval()
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        return mvae, scheduler, unet

    @torch.no_grad()
    def encode_image(self, x_input, mvae: AutoencoderKL):
        b, m, h, w, c = x_input.shape

        x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, m, 4, 512, 512)
        x_input = x_input.reshape(-1, c, h, w) # (bs*m, 4, 512, 512)
        z = mvae.encode(x_input).latent_dist  # (bs*m, 4, 64, 64)

        z = z.sample()
        _, c, h, w = z.shape
        z = z.reshape(b, -1, c, h, w)  # (bs, m, 4, 64, 64)

        # use the scaling factor from the vae config
        z = z * mvae.config.scaling_factor
        z = z.float()
        return z
    
    @torch.no_grad()
    def decode_latent(self, latents, mvae: AutoencoderKL):
        b, m, c, h, w = latents.shape
        latents = (1 / mvae.config.scaling_factor * latents)
        images = []
        for j in range(m):
            image = mvae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1) # (bs, m, 4, 512, 512)
        # TODO: remove mask channel
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy()
        image = (image * 255).round().astype('uint8')

        return image
    
    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            param_groups.append({"params": params, "lr": self.lr * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):      
        device = batch['images'].device
        prompt_embds = []
        # TODO: change to image encoder
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        
        idxs = batch['idxs']
        latents = self.encode_image(batch['images'], self.mvae)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=1)

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, latents.shape[1])
        denoise = self.unet(noise_z, t, prompt_embds, idxs)
        target = noise

        # eps mode
        loss = torch.nn.functional.mse_loss(denoise, target)
        self.log('train_loss', loss)
        return loss
    
    def gen_cls_free_guide_pair(self, latents, timestep, prompt_embd, idxs):
        latents = torch.cat([latents]*2)
        timestep = torch.cat([timestep]*2)
        idxs = torch.cat([idxs]*2)

        return latents, timestep, prompt_embd, idxs
    
    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, idxs, model):
        latents, _timestep, _prompt_embd, idxs = \
            self.gen_cls_free_guide_pair(latents_high_res, _timestep, prompt_embd, idxs)

        noise_pred = model(latents, _timestep, _prompt_embd, idxs)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * \
            (noise_pred_text - noise_pred_uncond)

        return noise_pred
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred = self.inference(batch)
        images = ((batch['images'] / 2+ 0.5) * 255).cpu().numpy().astype(np.uint8)
      
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch_idx)

    @torch.no_grad()
    def inference(self, batch):
        images = batch['images']
        bs, m, h, w, _ = images.shape
        device = images.device

        latents= torch.randn(bs, m, 4, h // 8, w // 8, device=device)

        # TODO: change to image encoder
        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(self.encode_text(
                prompt, device)[0])
        prompt_embds = torch.stack(prompt_embds, dim=1)

        prompt_null = self.encode_text('', device)[0]
        prompt_embd = torch.cat(
            [prompt_null[:, None].repeat(1, m, 1, 1), prompt_embds])
        
        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps
        idxs = batch['idxs']

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]] * m, dim=1)

            noise_pred = \
                self.forward_cls_free(latents, _timestep, prompt_embd, idxs, self.unet)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        images_pred = self.decode_latent(latents, self.mvae)
       
        return images_pred
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images_pred = self.inference(batch)

        images = ((batch['images'] / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8)

        image_id = batch['image_id'][0]
        
        output_dir = batch['output_dir'][0] if 'output_dir' in batch else os.path.join(self.logger.log_dir, 'images')
        output_dir = os.path.join(output_dir, "{}".format(image_id))
        
        os.makedirs(output_dir, exist_ok = True)
        for i in range(images.shape[1]):
            path = os.path.join(output_dir, f'{i}.png')
            im = Image.fromarray(images_pred[0, i])
            im.save(path)
            im = Image.fromarray(images[0, i])
            path = os.path.join(output_dir, f'{i}_gt.png')
            im.save(path)

    @torch.no_grad()
    def save_image(self, images_pred, images, batch_idx):
        img_dir = os.path.join(self.logger.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        if images_pred is not None:
            for m_i in range(images_pred.shape[1]):
                im = Image.fromarray(images_pred[0, m_i])
                im.save(os.path.join(
                    img_dir, f'{self.global_step}_{batch_idx}_{m_i}_pred.png'))
                if m_i < images.shape[1]:
                    im = Image.fromarray(images[0, m_i])
                    im.save(os.path.join(img_dir, f'{self.global_step}_{batch_idx}_{m_i}_gt.png'))