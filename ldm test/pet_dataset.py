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
        
        # 适配Oxford-IIIT Pet数据集在Kaggle的结构
        # 尝试各种可能的路径模式
        potential_paths = [
            os.path.join(data_root, '*.jpg'),                      # /path/to/images/*.jpg
            os.path.join(data_root, '**', '*.jpg'),                # 递归搜索所有子目录
            os.path.join(data_root, 'images', '*.jpg'),            # /path/to/images/images/*.jpg
            os.path.join(data_root, 'images', 'images', '*.jpg'),  # 有些数据集有多层images目录
        ]
        
        self.image_paths = []
        for path_pattern in potential_paths:
            files = glob.glob(path_pattern, recursive=True)
            if files:
                # 过滤掉在annotations目录下的文件
                files = [f for f in files if 'annotations' not in f and 'trimaps' not in f]
                if files:
                    self.image_paths = files
                    print(f"找到图像文件: {path_pattern} ({len(files)}张)")
                    break
        
        # 如果还是找不到，显示更多调试信息
        if not self.image_paths:
            print(f"Warning: 在指定目录 {data_root} 中未找到有效的JPG图像文件")
            print(f"尝试列出目录内容:")
            try:
                if os.path.exists(data_root) and os.path.isdir(data_root):
                    all_files = os.listdir(data_root)
                    print(f"目录包含 {len(all_files)} 个文件/文件夹")
                    print(f"前10个文件: {all_files[:10]}")
                else:
                    print(f"指定的数据路径不是一个有效的目录: {data_root}")
            except Exception as e:
                print(f"列出目录内容时出错: {e}")
        
        # 如果图像路径为空，使用备用方法查找
        if not self.image_paths:
            print("尝试备用方法查找图像...")
            for root, dirs, files in os.walk(data_root):
                # 跳过annotations目录
                if 'annotations' in root or 'trimaps' in root:
                    continue
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(root, file))
            
            if self.image_paths:
                print(f"备用方法找到 {len(self.image_paths)} 张图像")
        
        # 简单的训练/验证分割 (80/20)
        if self.image_paths:
            np.random.seed(42)
            indices = np.random.permutation(len(self.image_paths))
            split_idx = int(0.8 * len(self.image_paths))
            
            if split == 'train':
                self.image_paths = [self.image_paths[i] for i in indices[:split_idx]]
            else:
                self.image_paths = [self.image_paths[i] for i in indices[split_idx:]]
            
            # 打印前几张图像的路径以便调试
            print(f"前5张{split}图像路径示例:")
            for path in self.image_paths[:5]:
                print(f"  {path}")
        else:
            print("ERROR: 无法找到任何图像文件！训练将失败。")
            # 创建一个空数组，之后会尝试处理
            self.image_paths = []
        
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
        return max(len(self.image_paths), 1)  # 至少返回1以避免空数据集错误
    
    def __getitem__(self, idx):
        # 检查是否有图像路径
        if not self.image_paths:
            # 返回随机噪声作为图像，以避免训练过程崩溃
            dummy_image = torch.randn(3, self.size, self.size)
            dummy_image = torch.clamp((dummy_image + 1.0) / 2.0, min=0.0, max=1.0)
            dummy_image = dummy_image * 2.0 - 1.0  # 归一化到[-1, 1]
            return {"image": dummy_image}
        
        # 确保idx在范围内
        idx = idx % len(self.image_paths)
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            
            return {"image": image}
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 生成随机噪声作为备用
            dummy_image = torch.randn(3, self.size, self.size)
            dummy_image = torch.clamp((dummy_image + 1.0) / 2.0, min=0.0, max=1.0)
            dummy_image = dummy_image * 2.0 - 1.0  # 归一化到[-1, 1]
            return {"image": dummy_image}


class PetDatasetTrain(PetDataset):
    def __init__(self, data_root, size=256, **kwargs):
        super().__init__(data_root, split='train', size=size, **kwargs)


class PetDatasetValidation(PetDataset):
    def __init__(self, data_root, size=256, **kwargs):
        super().__init__(data_root, split='validation', size=size, **kwargs) 
