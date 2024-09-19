import pytorch_lightning as pl
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from .MVUNet import MultiViewUNet
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
from PIL import Image

class MultiViewDiffuison(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.lr = config['train']['lr']
        self.max_epochs = config['train']['max_epochs'] if 'max_epochs' in config['train'] else 0
        self.diff_timestep = config['model']['diff_timestep']
        self.guidance_scale = config['model']['guidance_scale']

        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     config['model']['model_id'], subfolder="tokenizer", torch_dtype=torch.float16)
        # self.text_encoder = CLIPTextModel.from_pretrained(
        #     config['model']['model_id'], subfolder="text_encoder", torch_dtype=torch.float16)

        self.mvae, self.scheduler, unet, self.vision_model, self.visual_projection, self.image_processor = \
            self.load_model(config['model']['model_id'])
        self.unet = MultiViewUNet(unet)
        self.trainable_params = self.unet.trainable_parameters

        self.save_hyperparameters()
        self.m_pos = torch.ones(1, 64, 64)
        self.m_neg = torch.zeros(1, 64, 64)
        self.white_img = \
            torch.cat([torch.ones(3, 512, 512), torch.zeros(1, 512, 512)], dim=0) # (4, 512, 512)
        # epsilon-prediction or velocity-prediction
        scheduler_config = self.scheduler.config
        if config['model']['prediction_type'] is None:
            self.prediction_type = scheduler_config.prediction_type
        else:
            self.prediction_type = config['model']['prediction_type']
        
        scheduler_config['prediction_type'] = self.prediction_type
        self.scheduler = DDIMScheduler.from_config(scheduler_config)


    def load_model(self, model_id):
        mvae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        mvae.eval()
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        image_processor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
        vision_model = safety_checker.vision_model
        visual_projection = safety_checker.visual_projection
        vision_model.eval()
        visual_projection.eval()
        return mvae, scheduler, unet, vision_model, visual_projection, image_processor

    @torch.no_grad()
    def encode_image(self, x_input, mvae: AutoencoderKL):
        b, m, c, h, w = x_input.shape

        # x_input = x_input.permute(0, 1, 4, 2, 3)  # (bs, m, 4, 512, 512)
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
        image = torch.stack(images, dim=1) # (bs, m, 4, 512, 512)
        image = image[:, :, :3, :, :] # (bs, m, 3, 512, 512)
        image = (image / 2 + 0.5).clamp(0, 1) # (bs, m, 3, 512, 512)
        image = image.cpu().permute(0, 1, 3, 4, 2).float().numpy() # (bs, m, 512, 512, 3)
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
        idxs = batch['idxs'] # (bs, m)
        cond_num = batch['cond_num'][0] # int, assume the batch share the same number of condition images

        bs, m = idxs.shape
        # CLIP image encoder for cross-attn embeddings
        for i, idx in enumerate(idxs):
            cond_img = batch['images'][i, 0] # (4, 512, 512)
            cond_img = cond_img[:3, :, :] # remove mask channel # (3, 512, 512)
            cond_img = (cond_img / 2 + 0.5) * 255. # (3, 512, 512)
            inputs = self.image_processor(images=cond_img, return_tensors='pt') # (1, 3, 224, 224)
            img_embeddings = self.vision_model(**inputs).last_hidden_state # (1, l, c_vis)
            img_embeddings = self.visual_projection(img_embeddings) # (1, l, embed_dim)
            prompt_embds.append(img_embeddings.repeat(m, 1, 1)) # (m, l, embed_dim)

        latents = self.encode_image(batch['images'], self.mvae)
        t = torch.randint(0, self.scheduler.num_train_timesteps,
                        (latents.shape[0],), device=latents.device).long()
        prompt_embds = torch.stack(prompt_embds, dim=0) # (bs, m, l, embed_dim)

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t) # (bs, m, 4, 64, 64)
        mask_cond = torch.ones(bs, cond_num, 1, 64, 64, device=device)
        mask_gen = torch.zeros(bs, m - cond_num, 1, 64, 64, device=device)
        mask = torch.cat([mask_cond, mask_gen], dim=1) # (bs, m, 1, 64, 64)

        latents_concat = self.white_img[None, None].repeat(bs, m - cond_num, 1, 1, 1) # (bs, m - cond_num, 4, 512, 512)
        latents_concat = torch.cat([batch['images'][:, cond_num:], latents_concat], dim=1) # (bs, m, 4, 512, 512)
        latents_concat = self.encode_image(latents_concat, self.mvae) # (bs, m, 4, 64, 64)
        noise_z = torch.cat([noise_z, latents_concat, mask], dim=2) # (bs, m, 9, 64, 64)
        t = t[:, None].repeat(1, latents.shape[1])
        denoise = self.unet(noise_z, t, prompt_embds, idxs)
        if self.prediction_type == 'epsilon':
            target = noise
        else:
            target = self.scheduler.get_velocity(latents, noise, t)

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
        images = ((batch['images'] / 2+ 0.5) * 255).cpu().numpy().astype(np.uint8) # (bs, m, 4, 512, 512)
        images = images[:, :, :3, :, :] # (bs, m, 3, 512, 512)
      
        # compute image & save
        if self.trainer.global_rank == 0:
            self.save_image(images_pred, images, batch_idx)

    @torch.no_grad()
    def inference(self, batch):
        images = batch['images']
        bs, m, h, w, _ = images.shape
        device = images.device
        idxs = batch['idxs']
        cond_num = batch['cond_num'][0] # int, assume the batch share the same number of condition images
        latents = torch.randn(bs, m, 4, h // 8, w // 8, device=device)
        encoder_latents = self.encode_image(images, self.mvae)

        # CLIP image encoder for cross-attn embeddings
        prompt_embds = []
        for i, idx in enumerate(idxs):
            cond_img = images[i, 0]
            cond_img = cond_img[:3, :, :] # remove mask channel # (3, 512, 512)
            cond_img = (cond_img / 2 + 0.5) * 255. # (3, 512, 512)
            inputs = self.image_processor(images=cond_img, return_tensors='pt') # (1, 3, 224, 224)
            img_embeddings = self.vision_model(**inputs).last_hidden_state # (1, l, c_vis)
            img_embeddings = self.visual_projection(img_embeddings) # (1, l, embed_dim)
            prompt_embds.append(img_embeddings.repeat(m, 1, 1)) # (m, l, embed_dim)
        
        prompt_embds = torch.stack(prompt_embds, dim=0) # (bs, m, l, embed_dim)

        # prompt_null = self.encode_text('', device)[0]
        null_image = torch.zeros(3, h, w, device=device)
        prompt_null = self.image_processor(images=null_image, return_tensors='pt')
        prompt_null = self.vision_model(**prompt_null).last_hidden_state # (1, l, c_vis)
        prompt_null = self.visual_projection(prompt_null) # (1, l, embed_dim)
        prompt_embd = torch.cat([prompt_null[:, None].repeat(bs, m, 1, 1), prompt_embds]) # (bs*2, m, l, embed_dim)
        
        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps
        
        mask_cond = torch.ones(bs, cond_num, 1, 64, 64, device=device)
        mask_gen = torch.zeros(bs, m - cond_num, 1, 64, 64, device=device)
        mask = torch.cat([mask_cond, mask_gen], dim=1) # (bs, m, 1, 64, 64)
        latents = torch.cat([latents, encoder_latents, mask], dim=2) # (bs, m, 9, 64, 64)
        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]] * m, dim=1)

            noise_pred = \
                self.forward_cls_free(latents, _timestep, prompt_embd, idxs, self.unet)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample # (bs, m, 4, 64, 64)
            latents = torch.cat([latents, encoder_latents, mask], dim=2) # (bs, m, 9, 64, 64)
        
        images_pred = self.decode_latent(latents, self.mvae)
       
        return images_pred
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images_pred = self.inference(batch)

        images = ((batch['images'] / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8)
        images = images[:, :, :3, :, :] # (bs, m, 3, 512, 512)

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