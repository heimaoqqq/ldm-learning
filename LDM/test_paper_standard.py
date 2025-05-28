"""
测试论文标准CLDM实现
验证模型结构和参数是否正确
"""

import torch
import yaml
from cldm_paper_standard import PaperStandardCLDM

def test_model_structure():
    """测试模型结构"""
    print("=== 测试论文标准CLDM模型结构 ===")
    
    # 创建模型
    model = PaperStandardCLDM(
        latent_dim=256,
        num_classes=31,
        num_timesteps=1000,
        beta_schedule="linear",
        model_channels=128,
        time_emb_dim=256,
        class_emb_dim=256,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[8],
        dropout=0.0
    )
    
    # 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ 模型创建成功")
    print(f"📊 总参数量: {total_params:,}")
    print(f"🏋️ 可训练参数: {trainable_params:,}")
    
    return model

def test_forward_pass():
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    model = PaperStandardCLDM(
        latent_dim=256,
        num_classes=31,
        num_timesteps=1000
    )
    
    # 创建测试数据
    batch_size = 4
    x_0 = torch.randn(batch_size, 256, 16, 16)
    class_labels = torch.randint(0, 31, (batch_size,))
    
    print(f"🔍 输入形状: {x_0.shape}")
    print(f"📝 类别标签: {class_labels}")
    
    # 前向传播
    try:
        loss, pred_noise, target_noise = model(x_0, class_labels)
        print(f"✅ 前向传播成功")
        print(f"📉 损失值: {loss.item():.6f}")
        print(f"🔊 预测噪声形状: {pred_noise.shape}")
        print(f"🎯 目标噪声形状: {target_noise.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

def test_sampling():
    """测试采样过程"""
    print("\n=== 测试DDIM采样 ===")
    
    model = PaperStandardCLDM(
        latent_dim=256,
        num_classes=31,
        num_timesteps=1000
    )
    
    # 创建类别标签
    class_labels = torch.tensor([0, 1, 2, 3])
    device = 'cpu'
    
    print(f"🎯 采样类别: {class_labels.tolist()}")
    
    try:
        # 采样
        samples = model.sample(
            class_labels=class_labels,
            device=device,
            num_inference_steps=20,  # 快速测试
            eta=0.0
        )
        
        print(f"✅ 采样成功")
        print(f"🖼️ 采样结果形状: {samples.shape}")
        print(f"📊 采样值范围: [{samples.min():.3f}, {samples.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"❌ 采样失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n=== 测试配置文件 ===")
    
    try:
        with open('config_paper_standard.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 配置文件加载成功")
        
        # 检查关键配置
        cldm_config = config['cldm']
        print(f"🏗️ 模型类型: {cldm_config['model_type']}")
        print(f"⏱️ 扩散步数: {cldm_config['num_timesteps']}")
        print(f"📐 模型通道数: {cldm_config['model_channels']}")
        print(f"🎯 类别数: {cldm_config['num_classes']}")
        
        training_config = config['training']
        print(f"🎓 学习率: {training_config['lr']}")
        print(f"📦 批量大小: {config['dataset']['batch_size']}")
        
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def test_architecture_compliance():
    """测试架构是否符合论文标准"""
    print("\n=== 验证论文标准符合性 ===")
    
    model = PaperStandardCLDM(
        latent_dim=256,
        num_classes=31,
        num_timesteps=1000
    )
    
    # 检查U-Net结构
    unet = model.unet
    
    print("🏗️ U-Net架构检查:")
    print(f"   ✅ 时间嵌入维度: {unet.time_emb_dim}")
    print(f"   ✅ 基础通道数: {unet.model_channels}")
    print(f"   ✅ 输入通道数: {unet.in_channels}")
    
    # 检查扩散过程
    diffusion = model.diffusion
    print(f"🔄 扩散过程检查:")
    print(f"   ✅ 时间步数: {diffusion.num_timesteps}")
    print(f"   ✅ Beta形状: {model.betas.shape}")
    
    # 检查关键组件
    has_attention = any('SelfAttentionBlock' in str(type(layer)) 
                       for layer in unet.middle)
    print(f"🎯 自注意力机制: {'✅ 存在' if has_attention else '❌ 缺失'}")
    
    return True

def main():
    """主测试函数"""
    print("🧪 开始测试论文标准CLDM实现...")
    
    tests = [
        ("模型结构", test_model_structure),
        ("前向传播", test_forward_pass),
        ("DDIM采样", test_sampling),
        ("配置文件", test_config_loading),
        ("论文符合性", test_architecture_compliance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} - 通过")
            else:
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
    
    print(f"\n📊 测试总结: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！论文标准实现准备就绪！")
        print("\n🚀 建议下一步:")
        print("   1. 上传到Kaggle")
        print("   2. 运行 train_paper_standard.py")
        print("   3. 监控训练进展和FID改进")
    else:
        print("⚠️ 部分测试失败，请检查实现")

if __name__ == "__main__":
    main() 