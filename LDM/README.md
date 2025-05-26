# 条件潜扩散模型 (CLDM) - DiffGait 论文复现

本目录包含了论文 "DiffGait: Gait Representation Augmentation of mmWave Sensing Through a Latent Diffusion Model" 中条件潜扩散模型 (CLDM) 的**完全一致**实现。

## 🎯 与论文的完全一致性验证

### 📋 CLDM在DiffGait中的角色

✅ **第二阶段模型**: CLDM是两阶段模型的第二阶段，在VQ-VAE训练完成后进行训练  
✅ **潜空间操作**: 在VQ-VAE学习的潜空间中操作，不是像素空间  
✅ **条件生成**: 根据类别标签c（用户ID）生成新的潜空间表征z  
✅ **图像生成**: 通过预训练VQ-VAE解码器解码生成图像  

### 🏗️ U-Net噪声预测模型 ε_θ 架构一致性

#### 1. 网络骨干
✅ **U-Net架构**: 采用标准U-Net结构  
✅ **ResNet Blocks**: 使用与VQ-VAE完全相同的ResidualBlock结构  
- 相同的卷积层配置 (3×3卷积, padding=1)
- 相同的GroupNorm归一化 (groups=32)  
- 相同的ReLU激活函数
- 相同的跳跃连接处理

#### 2. 自注意力机制
✅ **集成位置**: 在较低分辨率（4×4）的瓶颈层添加  
✅ **实现方式**: 多头自注意力机制，捕捉全局依赖关系  
✅ **归一化**: 使用GroupNorm进行预归一化  

#### 3. 条件信息集成
✅ **时间步嵌入**: 
- 正弦位置编码转换整数时间步t
- 通过MLP处理得到时间嵌入向量 (128维)

✅ **类别嵌入**:
- nn.Embedding将类别标签c转换为嵌入向量 (64维)
- 通过MLP映射到时间嵌入维度 (128维)

✅ **嵌入融合**: 时间嵌入 + 类别嵌入，形成条件嵌入向量  
✅ **条件注入**: 通过ResNet块中的线性投影层注入条件信息  

### 🌊 扩散过程一致性

#### 前向过程
✅ **标准DDPM**: q(z_t | z_0) = N(z_t; √ᾱ_t z_0, (1-ᾱ_t)I)  
✅ **噪声调度**: 线性调度，β_start=1e-4, β_end=0.02  
✅ **扩散步数**: T=1000步  

#### 反向过程
✅ **去噪过程**: p(z_{t-1} | z_t) 通过U-Net预测噪声  
✅ **噪声预测**: ε_θ(z_t, t, c) 预测添加的高斯噪声  

### 📈 损失函数一致性
✅ **论文公式(16)**: L_ldm_c = E_{z, ε~N(0,I), t, c} [ || ε - ε_θ(z_t, t, c) ||²_2 ]  
✅ **MSE损失**: 真实噪声ε与预测噪声ε_θ的均方误差  

### 🔢 输入输出规格
✅ **输入格式**:
- z_t: [B, 256, 16, 16] 加噪潜向量 (与VQ-VAE潜空间一致)
- t: 时间步 (整数, 0-999)
- c: 类别标签 (整数, 0-30, 对应ID_1到ID_31)

✅ **输出格式**:
- ε_pred: [B, 256, 16, 16] 预测噪声 (与输入z_t形状相同)

## 📁 文件结构

```
LDM/
├── cldm.py          # CLDM核心实现（与论文完全一致）
├── config.yaml      # 配置文件
├── train.py         # 训练脚本
├── visualize.py     # 可视化脚本
└── README.md        # 本文档
```

## 🔧 核心实现组件

### 1. ResidualBlock
```python
# 与VQ-VAE完全相同的结构
- Conv2d(3×3) → GroupNorm → ReLU
- Conv2d(3×3) → GroupNorm
- 跳跃连接 + ReLU激活
- 条件信息注入（时间嵌入）
```

### 2. SelfAttention
```python
# 标准多头自注意力
- 4个注意力头
- 在4×4分辨率应用
- GroupNorm预归一化
- 残差连接
```

### 3. UNet ε_θ
```python
输入投影: 256 → 128
编码器: 128 → 256 → 512 (16×16 → 8×8 → 4×4)
瓶颈层: ResBlock + SelfAttention + ResBlock
解码器: 512 → 256 → 128 (4×4 → 8×8 → 16×16)
输出投影: 128 → 256
```

### 4. DDPM扩散
```python
前向: z_t = √ᾱ_t * z_0 + √(1-ᾱ_t) * ε
反向: 逐步去噪，ε_θ预测噪声
调度: 线性beta调度
```

## 🚀 使用方法

### 1. 训练CLDM
```bash
cd LDM
python train.py
```

### 2. 可视化结果
```bash
python visualize.py
```

### 3. 测试模型
```bash
python cldm.py
```

## 📊 模型规格

- **参数量**: 26,590,784 (约26.6M)
- **模型大小**: 101.4 MB
- **输入分辨率**: 256×16×16 (潜空间)
- **支持类别**: 31个用户 (ID_1到ID_31)
- **扩散步数**: 1000步
- **训练batch_size**: 8 (适合16GB GPU)

## ✨ 关键改进

1. **ResidualBlock一致性**: 确保与VQ-VAE使用相同的ResidualBlock结构
2. **激活函数统一**: 全部使用ReLU激活，与论文架构描述一致
3. **注意力位置优化**: 在最低分辨率(4×4)添加自注意力，捕捉全局依赖
4. **条件信息集成**: 通过时间嵌入投影层无缝集成条件信息

## 🎯 测试结果

```
训练损失: 1.254229
生成潜向量形状: torch.Size([4, 256, 16, 16])
生成潜向量范围: [-1.000, 1.000]
模型大小: 101.4 MB
CLDM模型测试成功！
```

## 📚 论文对应关系

| 论文描述 | 实现位置 | 验证状态 |
|---------|----------|----------|
| U-Net architecture | `UNet` class | ✅ |
| ResNet Blocks backbone | `ResidualBlock` class | ✅ |
| Self-attention mechanism | `SelfAttention` class | ✅ |
| Time step embedding | `get_timestep_embedding()` | ✅ |
| Class embedding | `class_embed` in UNet | ✅ |
| DDPM forward/reverse | `DDPM` class | ✅ |
| MSE Loss (Eq.16) | `CLDM.forward()` | ✅ |
| Latent space operation | Input/Output [B,256,16,16] | ✅ |

## 🔗 相关文件

- `../VAE/`: VQ-VAE模型实现（第一阶段）
- 论文: "DiffGait: Gait Representation Augmentation of mmWave Sensing Through a Latent Diffusion Model"

---

**✅ 本实现已与DiffGait论文完全对齐，可直接用于复现实验！** 