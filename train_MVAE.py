import torch
import argparse
from src.dataset import MVAEDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
from src.models.MVAE import MVAE, MVAE_CONFIG
from torch.utils.data import DataLoader
# from pytorch_lightning.cli import LightningCLI

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.add_argument(
#         'main_cfg_path', type=str, help='main config path')
#         parser.add_argument(
#             '--exp_name', type=str, default='mvae')
#         parser.add_argument(
#             '--batch_size', type=int, default=4, help='batch_size per gpu')
#         parser.add_argument(
#             '--num_workers', type=int, default=1)
#         parser.add_argument(
#             '--ckpt_path', type=str, default=None,
#             help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')

# for older versions of pytorch-lightning
def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='mvae')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=0)
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

# def get_cli():
#     cli = MyLightningCLI(run=False)
#     return cli

if __name__ == "__main__":
    args = parse_args()
    # cli = get_cli()
    main_config_path = args.main_cfg_path
    torch.set_float32_matmul_precision('medium')
    config = yaml.load(open(main_config_path, 'rb'), Loader=yaml.SafeLoader)
    config['train']['max_epochs'] = args.max_epochs
    config['train']['batch_size'] = args.batch_size
    num_workers = args.num_workers

    train_dataset = MVAEDataset(path=config['data']['path'], split='train')
    val_dataset = MVAEDataset(path=config['data']['path'], split='val')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=False
    )

    if 'model_config_path' in config['model']:
        model_config = yaml.load(open(config['model']['model_config_path'], 'rb'), Loader=yaml.SafeLoader)
    else:
        model_config = MVAE_CONFIG

    model = MVAE(model_config, config['train'])
    
    if args.ckpt_path is not None:
        model.load_state_dict(
            torch.load(args.ckpt_path, map_location='cpu')['state_dict'], 
            strict=False
        )
    elif 'pretrained_vae' in config['model']:
        model.load_pretrained_vae(config['model']['pretrained_vae'])
        
    
    checkpoint_callback = \
        ModelCheckpoint(
            save_top_k=2, 
            monitor="train_loss",
            mode="min", 
            save_last=1,
            filename='epoch={epoch}-loss={train_loss:.4f}'
        )

    logger = TensorBoardLogger(
        save_dir='logs/tb_logs', 
        name=args.exp_name, 
        default_hp_metric=False
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)
