#!/usr/bin/env python3
"""
测试Oxford-IIIT Pet数据集加载
使用方法: python VAE/test_dataset.py
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 添加当前目录到路径
sys.path.insert(0, '.')

from VAE.pet_dataset import PetDatasetTrain, PetDatasetValidation


def test_dataset():
    """测试数据集加载和可视化"""
    
    print("测试Oxford-IIIT Pet数据集...")
    
    # 创建训练和验证数据集
    train_dataset = PetDatasetTrain(data_root='data', size=256)
    val_dataset = PetDatasetValidation(data_root='data', size=256)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 获取一个批次的数据
    batch = next(iter(train_loader))
    images = batch['image']
    
    print(f"批次形状: {images.shape}")
    print(f"数据类型: {images.dtype}")
    print(f"数值范围: [{images.min():.3f}, {images.max():.3f}]")
    
    # 可视化一些样本
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(4):
        # 反归一化图像 (从[-1,1]到[0,1])
        img = (images[i] + 1.0) / 2.0
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("样本图像保存为: dataset_samples.png")
    
    # 测试数据集的一致性
    print("\n测试数据集一致性...")
    sample_indices = [0, 100, 500, 1000]
    
    for idx in sample_indices:
        if idx < len(train_dataset):
            try:
                sample = train_dataset[idx]
                print(f"样本 {idx}: 形状 {sample['image'].shape}, 范围 [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            except Exception as e:
                print(f"样本 {idx} 加载失败: {e}")
    
    print("\n数据集测试完成!")


if __name__ == "__main__":
    test_dataset() 
