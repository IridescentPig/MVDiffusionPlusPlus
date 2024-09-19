import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
import random

class MaskedVAEDataset(torch.utils.data.Dataset):
    """
    base_data_path
    ├── train
    │   ├── xxx(uid)
    │   │   ├── 000.png 
    │   │   ├── 001.png
    │   │   └──...
    │   └── ...
    ├── val
    │   ├── xxx
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   └──...
    │   └── ...
    ├── test
    │   ├── xxx
    │   │   ├── 000.png
    │   │   ├── 001.png
    │   │   └──...
    │   └── ...
    """
    def __init__(self, path, split):
        super().__init__()
        self.split = split
        self.base_data_path = os.path.join(path, split)
        self.image_paths = []
        for dir in os.listdir(self.base_data_path):
            for file in os.listdir(os.path.join(self.base_data_path, dir)):
                if file.endswith('.png'):
                    self.image_paths.append(os.path.join(self.base_data_path, dir, file))

        self.to_tensor = ToTensor()
        random.shuffle(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGBA')
        image = self.to_tensor(image) # (4, H, W), [0, 1]
        rgb = image[:3]
        rgb = (rgb - 0.5) * 2. # [-1, 1]
        mask = image[3:]
        mask = torch.where(mask > 0., torch.ones_like(mask), torch.zeros_like(mask)) # binarize mask

        image = torch.cat([rgb, mask], dim=0) # (4, H, W)

        return {
            'images': image,
            'image_id': self.split,
            'image_path': path
        }

