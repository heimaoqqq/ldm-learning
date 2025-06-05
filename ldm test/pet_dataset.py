import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import glob
import numpy as np


class PetDataset(Dataset):
    """Oxford-IIIT Pet Dataset for VAE training"""
    
    def __init__(self, data_root, split='train', size=256, transform=None):
        self.data_root = data_root
        self.size = size
        self.split = split
        
        # 获取所有图像文件
        self.image_paths = glob.glob(os.path.join(data_root, 'images', '*.jpg'))
        
        # 简单的训练/验证分割 (80/20)
        np.random.seed(42)
        indices = np.random.permutation(len(self.image_paths))
        split_idx = int(0.8 * len(self.image_paths))
        
        if split == 'train':
            self.image_paths = [self.image_paths[i] for i in indices[:split_idx]]
        else:
            self.image_paths = [self.image_paths[i] for i in indices[split_idx:]]
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
            
        print(f"PetDataset {split} loaded with {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            return {"image": image}
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回第一张图片作为fallback
            return self.__getitem__(0)


class PetDatasetTrain(PetDataset):
    def __init__(self, data_root, size=256, **kwargs):
        super().__init__(data_root, split='train', size=size, **kwargs)


class PetDatasetValidation(PetDataset):
    def __init__(self, data_root, size=256, **kwargs):
        super().__init__(data_root, split='validation', size=size, **kwargs) 
