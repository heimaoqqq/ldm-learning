"""
🚀 增强版LDM训练脚本
使用包含真正Transformer的简化增强版U-Net
"""

import yaml
import torch
import sys
import os

# 使用最新的训练脚本作为基础
sys.path.append('.')
from train_fid_optimized import main

def train_enhanced_ldm():
    """训练增强版LDM"""
    
    print("🚀 启动增强版LDM训练")
    print("=" * 60)
    print("📊 模型特点:")
    print("  - 简化增强版U-Net (115.8M参数)")
    print("  - 6个注意力层 × 2层Transformer = 12个Transformer块")
    print("  - 多头注意力机制增强空间理解能力")
    print("  - P100显存友好设计")
    print("=" * 60)
    
    # 使用增强版配置
    config_path = "config_enhanced_unet.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 验证配置
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config['unet']['type'] != 'simple_enhanced':
            print("❌ 配置文件不是增强版类型")
            return
            
        print(f"✅ 配置验证成功: {config['unet']['type']}")
        print(f"✅ Transformer深度: {config['unet']['transformer_depth']}")
        
    except Exception as e:
        print(f"❌ 配置文件错误: {e}")
        return
    
    # 调用训练主函数
    try:
        main(config_path)
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        raise

if __name__ == "__main__":
    train_enhanced_ldm() 
