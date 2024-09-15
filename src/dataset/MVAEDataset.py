import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor

class MVAEDataset(torch.utils.data.Dataset):
    """
    base_data_path
    ├── train
    │   ├── xxx_001.png
    │   ├── xxx_002.png
    │   └── ...
    ├── val
    │   ├── xxx_001.png
    │   ├── xxx_002.png
    │   └── ...
    ├── test
    │   ├── xxx_001.png
    │   ├── xxx_002.png
    │   └── ...
    """
    def __init__(self, path, split):
        self.split = split
        self.base_data_path = os.path.join(path, split)
        self.image_paths = []
        for file in os.listdir(self.base_data_path):
            if file.endswith('.png'):
                self.image_paths.append(os.path.join(self.base_data_path, file))

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGBA')
        image = self.to_tensor(image) # (4, H, W)
        rgb = image[:3]
        rgb = (rgb / 255. - 0.5) * 2 # normalize to [-1, 1]
        mask = image[3:]
        mask = torch.where(mask > 0., torch.ones_like(mask), torch.zeros_like(mask)) # binarize mask

        image = torch.cat([rgb, mask], dim=0) # (4, H, W)

        return {
            'images': image,
            'image_id': self.split
        }

