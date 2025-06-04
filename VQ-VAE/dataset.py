"""
自定义数据集加载器
用于256*256彩色图像的VQ-VAE训练
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from typing import Optional, Tuple, List

class ColorImageDataset(Dataset):
    """256*256彩色图像数据集"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ):
        """
        初始化数据集
        
        Args:
            data_path: 数据集根目录路径
            split: 数据集分割 ('train', 'val', 'test')
            image_size: 图像尺寸 (默认256)
            augment: 是否使用数据增强
            extensions: 支持的图像格式
        """
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # 获取所有图像文件路径
        self.image_paths = self._get_image_paths(extensions)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在 {data_path} 中未找到任何图像文件")
        
        print(f"加载 {split} 数据集: {len(self.image_paths)} 张图像")
        
        # 定义变换
        self.transform = self._get_transforms()
    
    def _get_image_paths(self, extensions: List[str]) -> List[str]:
        """获取所有图像文件路径"""
        image_paths = []
        
        # 支持多种目录结构
        possible_paths = [
            os.path.join(self.data_path, self.split),  # data/train/, data/val/
            os.path.join(self.data_path, f"{self.split}_data"),  # data/train_data/
            self.data_path  # 直接在data/目录下
        ]
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                for ext in extensions:
                    pattern = os.path.join(base_path, f"**/*{ext}")
                    paths = glob.glob(pattern, recursive=True)
                    image_paths.extend(paths)
                
                # 如果找到图像，就不再搜索其他路径
                if image_paths:
                    break
        
        return sorted(list(set(image_paths)))  # 去重并排序
    
    def _get_transforms(self):
        """定义图像变换"""
        transforms_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化到[-1, 1]
        ]
        
        if self.augment and self.split == "train":
            # 训练时的数据增强
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05)
                )
            ]
            
            # 在Resize和ToTensor之间插入增强
            transforms_list = (
                [transforms.Resize((self.image_size, self.image_size))] +
                augment_transforms +
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            )
        
        return transforms.Compose(transforms_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """获取单个样本"""
        image_path = self.image_paths[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            # 返回随机图像作为备用
            return torch.randn(3, self.image_size, self.image_size)

def create_dataloaders(
    data_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 256,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_path: 数据集路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        train_split: 训练集比例
        val_split: 验证集比例  
        test_split: 测试集比例
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 检查是否已经有分割的目录结构
    if (os.path.exists(os.path.join(data_path, "train")) and
        os.path.exists(os.path.join(data_path, "val"))):
        # 使用已有的目录结构
        train_dataset = ColorImageDataset(data_path, "train", image_size, augment=True)
        val_dataset = ColorImageDataset(data_path, "val", image_size, augment=False)
        
        if os.path.exists(os.path.join(data_path, "test")):
            test_dataset = ColorImageDataset(data_path, "test", image_size, augment=False)
        else:
            test_dataset = val_dataset  # 使用验证集作为测试集
    
    else:
        # 自动分割数据集
        full_dataset = ColorImageDataset(data_path, "train", image_size, augment=False)
        
        # 计算分割大小
        total_size = len(full_dataset)
        train_size = int(total_size * train_split)
        val_size = int(total_size * val_split)
        test_size = total_size - train_size - val_size
        
        # 随机分割
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # 为训练集启用数据增强
        train_dataset.dataset.augment = True
        train_dataset.dataset.transform = train_dataset.dataset._get_transforms()
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"数据集统计:")
    print(f"  训练集: {len(train_dataset)} 张图像")
    print(f"  验证集: {len(val_dataset)} 张图像")
    print(f"  测试集: {len(test_dataset)} 张图像")
    print(f"  批次大小: {batch_size}")
    
    return train_loader, val_loader, test_loader

def visualize_batch(dataloader: DataLoader, num_images: int = 8):
    """可视化数据批次"""
    import matplotlib.pyplot as plt
    
    # 获取一个批次
    batch = next(iter(dataloader))
    
    if isinstance(batch, tuple):
        images = batch[0]
    else:
        images = batch
    
    # 反归一化
    images = (images + 1) / 2  # 从[-1,1]恢复到[0,1]
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Image {i+1}')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"图像形状: {images.shape}")
    print(f"像素值范围: [{images.min():.3f}, {images.max():.3f}]")

if __name__ == "__main__":
    # 测试数据集加载
    data_path = "data"  # 您的数据集路径
    
    if os.path.exists(data_path):
        train_loader, val_loader, test_loader = create_dataloaders(
            data_path=data_path,
            batch_size=8,
            num_workers=2
        )
        
        print("数据加载器创建成功!")
        visualize_batch(train_loader)
    else:
        print(f"数据路径 {data_path} 不存在，请创建并放入您的256*256彩色图像") 
