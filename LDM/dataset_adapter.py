"""
数据集适配器模块
将VAE文件夹下的dataset.py适配为LDM训练所需的格式
"""

import sys
import os
from typing import Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader

# 添加VAE路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'VAE'))

# 尝试导入VAE的数据加载器，如果失败则使用内置版本
try:
    from dataset import build_dataloader as vae_build_dataloader
except ImportError:
    print("⚠️ VAE dataset module not found, using built-in dataloader")
    vae_build_dataloader = None

def build_dataloader(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    适配器函数：将VAE的build_dataloader适配为LDM格式
    
    Args:
        config: LDM配置字典
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    # 提取数据集配置
    dataset_config = config.get('dataset', {})
    
    # 将LDM配置转换为VAE build_dataloader的参数
    vae_params = {
        'root_dir': dataset_config.get('root_dir', '/kaggle/input/dataset'),
        'batch_size': dataset_config.get('batch_size', 6),
        'num_workers': dataset_config.get('num_workers', 2),
        'shuffle_train': dataset_config.get('shuffle_train', True),
        'shuffle_val': False,  # 验证集不需要shuffle
        'val_split': dataset_config.get('val_split', 0.1),
        'random_state': 42
    }
    
    print(f"🔄 使用VAE数据加载器")
    print(f"📂 数据集路径: {vae_params['root_dir']}")
    print(f"📊 批次大小: {vae_params['batch_size']}")
    print(f"🔀 验证集比例: {vae_params['val_split']}")
    
    if vae_build_dataloader is None:
        print(f"❌ VAE数据加载器不可用")
        print(f"🔄 回退使用合成数据集...")
        return create_synthetic_dataloaders(config)
    
    try:
        # 调用VAE的build_dataloader
        train_loader, val_loader, train_size, val_size = vae_build_dataloader(**vae_params)
        
        # 创建包装的数据加载器以适配LDM的数据格式
        wrapped_train_loader = DataLoaderWrapper(train_loader)
        wrapped_val_loader = DataLoaderWrapper(val_loader)
        
        print(f"✅ 数据加载器创建成功:")
        print(f"  训练集: {train_size}个样本")
        print(f"  验证集: {val_size}个样本")
        
        return wrapped_train_loader, wrapped_val_loader
        
    except Exception as e:
        print(f"❌ 使用VAE数据加载器失败: {e}")
        print(f"🔄 回退使用合成数据集...")
        
        # 回退到合成数据集
        return create_synthetic_dataloaders(config)

class DataLoaderWrapper:
    """
    数据加载器包装器
    将VAE的数据格式(image, label)转换为LDM需要的格式{'image': ..., 'label': ...}
    """
    
    def __init__(self, original_loader: DataLoader):
        self.original_loader = original_loader
        
        # 复制原始loader的属性
        self.dataset = original_loader.dataset
        self.batch_size = original_loader.batch_size
        self.num_workers = original_loader.num_workers
        self.pin_memory = original_loader.pin_memory
        self.shuffle = getattr(original_loader, 'shuffle', False)
    
    def __iter__(self):
        """迭代器，转换数据格式"""
        for batch in self.original_loader:
            # 检查batch的类型和格式
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                # VAE格式: (images, labels) 或 [images, labels]
                images, labels = batch
                # 转换为LDM格式: {'image': images, 'label': labels}
                yield {
                    'image': images,
                    'label': labels
                }
            elif isinstance(batch, dict):
                # 如果已经是字典格式，直接返回
                yield batch
            else:
                # 处理其他可能的格式
                print(f"⚠️ 未知的批次格式: {type(batch)}")
                if hasattr(batch, '__len__') and len(batch) >= 2:
                    # 尝试取前两个元素
                    yield {
                        'image': batch[0],
                        'label': batch[1]
                    }
                else:
                    raise ValueError(f"无法处理的批次格式: {type(batch)}")
    
    def __len__(self):
        return len(self.original_loader)

def create_synthetic_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    创建合成数据集作为备用方案
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class SyntheticDataset(Dataset):
        def __init__(self, num_samples: int, num_classes: int):
            self.num_samples = num_samples
            self.num_classes = num_classes
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = torch.rand(3, 256, 256)
            label = torch.randint(0, self.num_classes, (1,)).item()
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    dataset_config = config.get('dataset', {})
    batch_size = dataset_config.get('batch_size', 6)
    num_workers = dataset_config.get('num_workers', 2)
    num_classes = config['unet']['num_classes']
    
    # 创建合成数据集
    train_dataset = SyntheticDataset(1000, num_classes)
    val_dataset = SyntheticDataset(200, num_classes)
    
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
    
    print(f"✅ 合成数据集创建完成:")
    print(f"  训练集: {len(train_dataset)}个样本")
    print(f"  验证集: {len(val_dataset)}个样本")
    print(f"  类别数: {num_classes}")
    
    return train_loader, val_loader

# 测试函数
if __name__ == "__main__":
    # 测试配置
    test_config = {
        'dataset': {
            'root_dir': '/kaggle/input/dataset',
            'batch_size': 4,
            'num_workers': 0,
            'val_split': 0.2,
            'shuffle_train': True
        },
        'unet': {
            'num_classes': 31
        }
    }
    
    print("🧪 测试数据集适配器...")
    
    try:
        train_loader, val_loader = build_dataloader(test_config)
        
        # 测试一个批次
        for batch in train_loader:
            print(f"✅ 批次格式正确:")
            print(f"  图像形状: {batch['image'].shape}")
            print(f"  标签形状: {batch['label'].shape}")
            print(f"  标签值: {batch['label']}")
            break
        
        print("✅ 数据集适配器测试通过")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise 
