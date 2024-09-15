import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os


class MVUNetDataset(torch.utils.data.Dataset):
    """
    base_data_path
    ├── train
    │   ├── xxx(uid)
    │   │   ├── xxx_001.png
    │   │   ├── xxx_002.png
    │   │   ├──...
    │   │   └── xxx_042.png
    │   └── ...
    ├── val
    │   ├── xxx
    │   │   ├── xxx_001.png
    │   │   ├── xxx_002.png
    │   │   ├──...
    │   │   └── xxx_042.png
    │   └── ...
    ├── test
    │   ├── xxx
    │   │   ├── xxx_001.png
    │   │   ├── xxx_002.png
    │   │   ├──...
    │   │   └── xxx_042.png
    │   └── ...
    """
    def __init__(self, path, split, config):
        self.split = split
        self.base_data_path = os.path.join(path, split)
        self.image_dirs = []
        for file in os.listdir(self.base_data_path):
            if os.path.isdir(os.path.join(self.base_data_path, file)):
                self.image_dirs.append(os.path.join(self.base_data_path, file))

        self.to_tensor = ToTensor()
        self.train_stage = config['train_stage'] # 0, 1, 2

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        base_path = self.image_dirs[idx]
        image_paths = []
        for i in range(42):
            image_paths.append(os.path.join(base_path, f'{i:03d}.png'))
        
        images = []
        for path in image_paths:
            image = Image.open(path).convert('RGBA')
            image = self.to_tensor(image)
            rgb = image[:3]
            rgb = (rgb / 255. - 0.5) * 2
            mask = image[3:]
            mask = torch.where(mask > 0., torch.ones_like(mask), torch.zeros_like(mask))
            image = torch.cat([rgb, mask], dim=0) # (4, H, W)
            images.append(image)

        images = torch.stack(images, dim=0) # (42, 4, H, W)
        if self.train_stage == 0 or self.train_stage == 1: # single view
            cond_num = 1
            gen_num = 8 # View dropout training strategy
            cond_idx = torch.randint(0, 10, (cond_num,)) # (cond_num,)
            cond_image = images[0] # (4, H, W)
            gen_idxs = torch.randperm(32)[:gen_num].sort() + 10 # (gen_num,), 10-41
            gen_images = images[gen_idxs]
            idxs = torch.cat([cond_idx, gen_idxs], dim=0) # (cond_num + gen_num,)
            images = torch.cat([cond_image.unsqueeze(0), gen_images], dim=0) # (1 + gen_num, 4, H, W)
        else: # train_stage == 2, sparse views
            # TODO: add sparse view training strategy
            pass

        return {
            'images': images,
            'image_id': self.split,
            'idxs': idxs,
            'cond_num': cond_num,
        }