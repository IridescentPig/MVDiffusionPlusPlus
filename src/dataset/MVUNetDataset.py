import torch
from torchvision.transforms import ToTensor
from PIL import Image
import os

WHITE_IMAGE = torch.ones(3, 512, 512)
ZERO_MASK = torch.zeros(1, 512, 512)
ONE_MASK = torch.ones(1, 512, 512)
GEN_IMAGE = torch.cat([WHITE_IMAGE, ONE_MASK], dim=0) # (4, 512, 512)

class MultiViewUNetDataset(torch.utils.data.Dataset):
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
        super().__init__()
        self.split = split
        self.base_data_path = os.path.join(path, split)
        self.image_dirs = []
        self.gen_num = config.get('gen_num', 8)
        for file in os.listdir(self.base_data_path):
            if os.path.isdir(os.path.join(self.base_data_path, file)):
                self.image_dirs.append(os.path.join(self.base_data_path, file))

        self.to_tensor = ToTensor()
        self.train_stage = config.get('train_stage', 0) # 0, 1, 2
        self.white_img = \
            torch.cat([torch.ones(3, 512, 512), torch.ones(1, 512, 512)], dim=0) # (4, 512, 512)

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        base_path = self.image_dirs[idx]
        image_paths = []
        if self.split == 'train':
            for i in range(0, 42):
                image_paths.append(os.path.join(base_path, f'{i:03d}.png'))
        else:
            for i in range(0, 42):
                path = os.path.join(base_path, f'{i:03d}.png')
                if os.path.exists(path):
                    image_paths.append(path)
                else:
                    image_paths.append('')

        images = []
        for path in image_paths:
            if path == '':
                image = self.white_img
            else:
                image = Image.open(path).convert('RGBA')
                image = self.to_tensor(image) # (4, H, W), [0, 1]
            rgb = image[:3]
            rgb = (rgb - 0.5) * 2. # [-1, 1]
            mask = image[3:]
            mask = torch.where(mask > 0., torch.ones_like(mask), torch.zeros_like(mask)) # binarize mask
            image = torch.cat([rgb, mask], dim=0) # (4, H, W)
            images.append(image)

        images = torch.stack(images, dim=0) # (42, 4, H, W)
        
        
        return {
            'images': images,
            'image_id': self.split,
            'train_stage': self.train_stage,
            'split': self.split,
            'gen_num': self.gen_num,
        }

def generate_batched_sorted_unique_random(batch_size, size, min_val=10, max_val=42):
    """
    Generate a batched tensor of sorted unique random integers.
    
    Args:
    batch_size (int): The number of batches to generate.
    size (int): The number of random integers to generate per batch.
    min_val (int): The minimum value of the range (inclusive).
    max_val (int): The maximum value of the range (exclusive).
    
    Returns:
    torch.Tensor: A 2D tensor of shape (batch_size, size) with sorted unique random integers.
    """
    assert size <= max_val - min_val, f"Size {size} is larger than the range of possible values {max_val - min_val}"
    
    # random_perm = torch.randperm(max_val - min_val).unsqueeze(0).expand(batch_size, -1)
    random_perms = []
    for _ in range(batch_size):
        random_perm = torch.randperm(max_val - min_val)
        random_perms.append(random_perm)
    random_perms = torch.stack(random_perms, dim=0)
    selected = random_perms[:, :size]
    sorted_values, _ = torch.sort(selected, dim=1)
    result = sorted_values + min_val
    
    return result

def collate_fn(batch):
    batch_size = len(batch)
    raw_images = [item['images'] for item in batch]
    raw_images = torch.stack(raw_images, dim=0) # (B, 42, 4, H, W)
    image_ids = [item['image_id'] for item in batch]
    train_stages = [item['train_stage'] for item in batch]
    train_stage = train_stages[0]
    splits = [item['split'] for item in batch]
    split = splits[0]
    gen_num = [item['gen_num'] for item in batch][0]
    if split == 'train':
        if train_stage == 0 or train_stage == 1:
            cond_num = 1
            cond_idxs = torch.zeros((batch_size, cond_num)).long() # (B, 1)
            gen_idxs = generate_batched_sorted_unique_random(
                batch_size, gen_num, min_val=10, max_val=42
            ).long() # (B, gen_num)
            cond_images = raw_images[torch.arange(batch_size).unsqueeze(1), cond_idxs] # (B, 1, 4, H, W)
            gen_images = raw_images[torch.arange(batch_size).unsqueeze(1), gen_idxs] # (B, gen_num, 4, H, W)
            images = torch.cat([cond_images, gen_images], dim=1) # (B, 1 + gen_num, 4, H, W)
            # cond_idxs = torch.randint(0, 10, (batch_size, 1)).long() # (B, 1)
            idxs = torch.cat([cond_idxs, gen_idxs], dim=1) # (B, 1 + gen_num)
        else:
            r = torch.rand(1).item()
            if r < 0.5:
                cond_num = 1
                cond_idxs = torch.zeros((batch_size, cond_num)).long() # (B, 1)
                gen_idxs = generate_batched_sorted_unique_random(
                    batch_size, gen_num, min_val=10, max_val=42
                ).long() # (B, gen_num)
                cond_images = raw_images[torch.arange(batch_size).unsqueeze(1), cond_idxs] # (B, 1, 4, H, W)
                gen_images = raw_images[torch.arange(batch_size).unsqueeze(1), gen_idxs] # (B, gen_num, 4, H, W)
                images = torch.cat([cond_images, gen_images], dim=1) # (B, 1 + gen_num, 4, H, W)
                cond_idxs = torch.randint(0, 10, (batch_size, 1)).long() # (B, 1)
                idxs = torch.cat([cond_idxs, gen_idxs], dim=1) # (B, 1 + gen_num)
            else:
                cond_num = torch.randint(2, 11, (1,)).item()
                cond_idxs = generate_batched_sorted_unique_random(
                    batch_size, cond_num - 1, min_val=1, max_val=10
                ).long() # (B, cond_num - 1)
                cond_idxs = torch.cat([torch.zeros((batch_size, 1)).long(), cond_idxs], dim=1) # (B, cond_num)
                gen_idxs = generate_batched_sorted_unique_random(
                    batch_size, gen_num, min_val=10, max_val=42
                ).long()
                cond_images = raw_images[torch.arange(batch_size).unsqueeze(1), cond_idxs] # (B, cond_num, 4, H, W)
                gen_images = raw_images[torch.arange(batch_size).unsqueeze(1), gen_idxs] # (B, gen_num, 4, H, W)
                images = torch.cat([cond_images, gen_images], dim=1)
                cond_idxs = generate_batched_sorted_unique_random(
                    batch_size, cond_num, min_val=0, max_val=10
                ).long()
                idxs = torch.cat([cond_idxs, gen_idxs], dim=1) # (B, cond_num + gen_num)
    else:
        cond_num = 1
        cond_idxs = torch.zeros((batch_size, cond_num)).long() # (B, 1)
        gen_idxs = generate_batched_sorted_unique_random(
            batch_size, gen_num, min_val=10, max_val=42
        ).long() # (B, gen_num)
        cond_images = raw_images[torch.arange(batch_size).unsqueeze(1), cond_idxs] # (B, 1, 4, H, W)
        gen_images = GEN_IMAGE.unsqueeze(0).unsqueeze(0).expand(batch_size, gen_num, -1, -1, -1) # (B, gen_num, 4, H, W)
        gt_gen_images = raw_images[torch.arange(batch_size).unsqueeze(1), gen_idxs] # (B, gen_num, 4, H, W)
        images = torch.cat([cond_images, gen_images], dim=1) # (B, 1 + gen_num, 4, H, W)
        gt_images = torch.cat([cond_images, gt_gen_images], dim=1) # (B, 1 + gen_num, 4, H, W)
        # cond_idxs = torch.randint(0, 10, (batch_size, 1)).long() # (B, 1)
        idxs = torch.cat([cond_idxs, gen_idxs], dim=1) # (B, 1 + gen_num)
    
    if split == 'train':
        return {
            'images': images,
            'idxs': idxs,
            'image_id': image_ids,
            'train_stage': train_stages,
            'split': splits,
            'cond_num': [cond_num] * batch_size,
        }
    else:
        return {
            'images': images,
            'idxs': idxs,
            'image_id': image_ids,
            'train_stage': train_stages,
            'split': splits,
            'cond_num': [cond_num] * batch_size,
            'gt_images': gt_images,
        }