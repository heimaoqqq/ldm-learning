#!/usr/bin/env python3
"""
快速测试脚本 - 验证VQ-VAE环境设置
"""

import os
import sys
import torch
from pathlib import Path

def test_environment():
    """测试基本环境"""
    print("🧪 环境测试")
    print("=" * 40)
    
    # Python和PyTorch版本
    print(f"✅ Python: {sys.version.split()[0]}")
    print(f"✅ PyTorch: {torch.__version__}")
    
    # CUDA检查
    if torch.cuda.is_available():
        print(f"✅ CUDA: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("❌ CUDA不可用")
    
    print()

def test_taming_import():
    """测试taming-transformers导入"""
    print("📦 taming-transformers导入测试")
    print("=" * 40)
    
    # 添加路径
    taming_dir = Path("../taming-transformers-master").resolve()
    if not taming_dir.exists():
        print(f"❌ 未找到taming-transformers目录: {taming_dir}")
        return False
    
    sys.path.insert(0, str(taming_dir))
    
    try:
        from taming.models.vqgan import VQModel
        print("✅ taming.models.vqgan.VQModel 导入成功")
        
        from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
        print("✅ VQLPIPSWithDiscriminator 导入成功")
        
        from taming.modules.vqvae.quantize import VectorQuantizer2
        print("✅ VectorQuantizer2 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🔧 模型创建测试")
    print("=" * 40)
    
    try:
        from taming.models.vqgan import VQModel
        
        # 最小配置
        ddconfig = {
            "double_z": False,
            "z_channels": 256,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 64,  # 减少通道数用于测试
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0
        }
        
        lossconfig = {
            "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "disc_conditional": False,
                "disc_in_channels": 3,
                "disc_start": 10000,
                "disc_weight": 0.8,
                "codebook_weight": 1.0
            }
        }
        
        model = VQModel(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=1024,  # 小码本用于测试
            embed_dim=256
        )
        
        print("✅ VQModel 创建成功")
        
        # 计算参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ 模型参数: {total_params/1e6:.2f}M")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n🚀 前向传播测试")
    print("=" * 40)
    
    try:
        from taming.models.vqgan import VQModel
        
        # 创建测试模型 (更小的配置)
        ddconfig = {
            "double_z": False,
            "z_channels": 64,
            "resolution": 64,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 32,
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0
        }
        
        lossconfig = {
            "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "disc_conditional": False,
                "disc_in_channels": 3,
                "disc_start": 10000,
                "disc_weight": 0.8,
                "codebook_weight": 1.0
            }
        }
        
        model = VQModel(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=512,
            embed_dim=64
        )
        
        # 创建测试输入
        x = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            # 前向传播
            dec, diff = model(x)
            print(f"✅ 输入形状: {x.shape}")
            print(f"✅ 输出形状: {dec.shape}")
            print(f"✅ 量化损失: {diff.item():.4f}")
            
            # 编码解码测试
            quant, emb_loss, info = model.encode(x)
            rec = model.decode(quant)
            print(f"✅ 编码形状: {quant.shape}")
            print(f"✅ 重建形状: {rec.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation():
    """测试数据集创建"""
    print("\n📊 数据集测试")
    print("=" * 40)
    
    try:
        # 创建测试图像
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        # 生成一些测试图像
        from PIL import Image
        import numpy as np
        
        for i in range(3):
            # 创建随机RGB图像
            img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(test_dir / f"test_image_{i}.jpg")
        
        print(f"✅ 创建了3张测试图像在 {test_dir}")
        
        # 测试数据集类
        dataset_code = '''
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

class TestDataset(Dataset):
    def __init__(self, data_path="test_images", size=256):
        self.data_path = Path(data_path)
        self.size = size
        self.image_paths = list(self.data_path.glob("*.jpg"))
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return {"image": self.transform(image)}

# 测试数据集
dataset = TestDataset()
print(f"数据集大小: {len(dataset)}")

# 测试加载
sample = dataset[0]
print(f"样本形状: {sample['image'].shape}")
print(f"数据范围: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
'''
        
        exec(dataset_code)
        print("✅ 数据集创建和加载成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🎯 VQ-VAE环境完整测试")
    print("=" * 50)
    
    tests = [
        ("环境基础", test_environment),
        ("taming导入", test_taming_import),
        ("模型创建", test_model_creation),
        ("前向传播", test_forward_pass),
        ("数据集创建", test_dataset_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
            results[test_name] = False
        
        print()
    
    # 总结
    print("📋 测试总结")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有测试通过！环境配置正确。")
        print("现在可以运行:")
        print("  python kaggle_vqvae_setup.py")
        print("  python train_kaggle_vqgan.py")
    else:
        print("⚠️ 部分测试失败，请检查环境配置。")
    
    # 清理测试文件
    import shutil
    if Path("test_images").exists():
        shutil.rmtree("test_images")
        print("🧹 已清理测试文件")

if __name__ == "__main__":
    main() 
