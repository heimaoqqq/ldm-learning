"""
快速验证训练 - 云服务器版本
训练5个epoch评估FID趋势，决定是否需要fine-tune VAE
适配Kaggle等云环境
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# 云环境路径设置
print("🌐 云服务器环境初始化...")
print(f"📂 当前工作目录: {os.getcwd()}")

# 检查并安装必要的依赖
def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    try:
        import diffusers
        print("✅ diffusers 已安装")
    except ImportError:
        print("📦 安装 diffusers...")
        os.system("pip install diffusers transformers accelerate -q")
    
    try:
        import torchvision
        print("✅ torchvision 已安装")
    except ImportError:
        print("📦 安装 torchvision...")
        os.system("pip install torchvision -q")

check_and_install_dependencies()

# 添加路径 - 云环境适配
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# 尝试导入所需模块
try:
    from vae_ldm_standard import create_standard_vae_ldm
    print("✅ 成功导入 vae_ldm_standard")
except ImportError as e:
    print(f"❌ 导入 vae_ldm_standard 失败: {e}")
    print("💡 请确保所有必要文件都已上传到云环境")

try:
    from fid_evaluation import FIDEvaluator
    print("✅ 成功导入 fid_evaluation")
except ImportError as e:
    print(f"❌ 导入 fid_evaluation 失败: {e}")
    print("💡 请确保 fid_evaluation.py 文件已上传")

# 云环境数据加载器
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

class CloudGaitDataset(Dataset):
    """云环境适配的数据集类"""
    
    def __init__(self, image_paths, class_ids, transform=None):
        self.image_paths = image_paths
        self.class_ids = class_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_ids[idx]
        
        # 确保标签在有效范围内
        if label < 0 or label > 30:
            label = max(0, min(30, label))
        
        return image, label

def build_cloud_dataloader(root_dir, batch_size=4, num_workers=2, val_split=0.3):
    """云环境数据加载器"""
    print(f"🔍 扫描云数据集目录: {root_dir}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_image_paths = []
    all_class_ids = []
    
    # 检查目录结构
    if not os.path.exists(root_dir):
        raise ValueError(f"❌ 数据集目录不存在: {root_dir}")
    
    # 支持ID_1, ID_2, ... ID_31文件夹结构
    id_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith('ID_'):
            id_folders.append(item)
    
    if id_folders:
        print(f"📁 发现ID文件夹结构: {len(id_folders)} 个文件夹")
        
        for folder_name in sorted(id_folders):
            folder_path = os.path.join(root_dir, folder_name)
            
            try:
                # 提取类别ID
                class_id = int(folder_name.split('_')[1]) - 1  # 转换为0-based
                
                if not (0 <= class_id <= 30):
                    print(f"⚠️  跳过超出范围的类别: {folder_name}")
                    continue
                
                # 收集图片
                folder_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    folder_images.extend(glob.glob(os.path.join(folder_path, ext)))
                
                print(f"  {folder_name}: {len(folder_images)} 张图片")
                
                all_image_paths.extend(folder_images)
                all_class_ids.extend([class_id] * len(folder_images))
                
            except (ValueError, IndexError) as e:
                print(f"⚠️  无法解析文件夹名 {folder_name}: {e}")
                continue
    else:
        # 直接扫描所有图片文件
        print("📁 未发现ID文件夹结构，直接扫描图片...")
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            all_image_paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
        
        # 从文件名推断类别（简化版本）
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            try:
                # 尝试从文件名提取ID
                if 'ID_' in filename:
                    class_id = int(filename.split('ID_')[1].split('_')[0]) - 1
                else:
                    # 默认分配类别0
                    class_id = 0
                
                if 0 <= class_id <= 30:
                    all_class_ids.append(class_id)
                else:
                    all_class_ids.append(0)  # 默认类别
                    
            except (ValueError, IndexError):
                all_class_ids.append(0)  # 默认类别

    if not all_image_paths:
        raise ValueError(f"❌ 在 {root_dir} 中没有找到图片文件")

    print(f"📊 数据集统计:")
    print(f"  总图片数: {len(all_image_paths)}")
    print(f"  类别数: {len(set(all_class_ids))}")

    # 数据集划分
    train_paths, val_paths, train_ids, val_ids = train_test_split(
        all_image_paths, all_class_ids, 
        test_size=val_split, 
        random_state=42,
        stratify=all_class_ids if len(set(all_class_ids)) > 1 else None
    )

    print(f"  训练集: {len(train_paths)} 张")
    print(f"  验证集: {len(val_paths)} 张")

    # 创建数据集和加载器
    train_dataset = CloudGaitDataset(train_paths, train_ids, transform=transform)
    val_dataset = CloudGaitDataset(val_paths, val_ids, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

class CloudQuickValidationTrainer:
    """云环境快速验证训练器"""
    
    def __init__(self, data_dir='/kaggle/input/dataset'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 云服务器设备: {self.device}")
        
        # 云环境配置
        self.config = {
            'batch_size': 8,  # 云环境可以用更大的batch size
            'learning_rate': 0.0005,
            'weight_decay': 0.001,
            'max_epochs': 5,
            'data_dir': data_dir,
            'save_dir': './quick_validation_results',  # 云环境保存路径
            'num_workers': 2  # 云环境worker数量
        }
        
        # 创建保存目录
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # 1. 创建数据加载器
        print("📁 加载云数据集...")
        self.train_loader, self.val_loader, train_size, val_size = build_cloud_dataloader(
            root_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            val_split=0.3
        )
        print(f"   训练集: {len(self.train_loader)} batches ({train_size} samples)")
        print(f"   验证集: {len(self.val_loader)} batches ({val_size} samples)")
        
        # 2. 创建标准VAE-LDM模型
        print("🔧 创建标准VAE-LDM模型...")
        self.model = create_standard_vae_ldm(
            image_size=32,
            num_classes=31,
            diffusion_steps=1000,
            noise_schedule="cosine",
            device=self.device
        )
        
        # 3. 配置优化器
        self.optimizer = self.model.configure_optimizers(
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 4. 初始化FID评估器
        print("📊 初始化FID评估器...")
        self.fid_evaluator = FIDEvaluator(device=self.device)
        self.fid_evaluator.compute_real_features(
            self.val_loader, 
            max_samples=200
        )
        
        # 记录训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'fid_score': []
        }
        
        print("✅ 云环境快速验证训练器初始化完成!")
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            losses = self.model(images, labels, current_epoch=epoch)
            loss = losses['loss'].mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=2.0)
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })
            
            # 云环境限制batch数量以加速验证
            if batch_idx >= 100:  # 云环境可以训练更多batch
                break
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate_fid(self, epoch: int, num_samples: int = 200):
        """FID评估"""
        print(f"📊 Epoch {epoch+1} FID评估...")
        
        self.model.eval()
        
        # 随机选择类别
        classes = torch.randint(0, 31, (num_samples,), device=self.device)
        
        # 生成样本
        generated_images, _ = self.model.sample(
            num_samples=num_samples,
            class_labels=classes,
            use_ddim=True,
            num_inference_steps=50
        )
        
        # 计算FID
        fid_score = self.fid_evaluator.calculate_fid_score(generated_images)
        
        print(f"🎯 Epoch {epoch+1} FID: {fid_score:.2f}")
        
        return fid_score
    
    def save_sample_images(self, epoch: int, num_samples: int = 16):
        """保存样本图像"""
        self.model.eval()
        
        with torch.no_grad():
            # 生成样本
            classes = torch.arange(min(16, 31), device=self.device)[:num_samples]
            generated_images, _ = self.model.sample(
                num_samples=num_samples,
                class_labels=classes,
                use_ddim=True,
                num_inference_steps=50
            )
            
            # 反归一化
            def denormalize(tensor):
                return torch.clamp((tensor + 1) / 2, 0, 1)
            
            images = denormalize(generated_images).cpu()
            
            # 创建网格显示
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            
            for i in range(num_samples):
                img = images[i].permute(1, 2, 0).numpy()
                axes[i].imshow(img)
                axes[i].set_title(f'Class {classes[i].item()}')
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{self.config["save_dir"]}/samples_epoch_{epoch+1}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def train(self):
        """主训练循环"""
        print("🚀 开始云环境快速验证训练...")
        print("=" * 60)
        
        best_fid = float('inf')
        
        for epoch in range(self.config['max_epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['max_epochs']}")
            
            # 训练一个epoch
            avg_loss = self.train_epoch(epoch)
            print(f"📊 训练损失: {avg_loss:.4f}")
            
            # 评估FID
            fid_score = self.evaluate_fid(epoch, num_samples=200)
            
            # 保存样本图像
            if epoch % 2 == 0:
                self.save_sample_images(epoch)
            
            # 记录历史
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(avg_loss)
            self.train_history['fid_score'].append(fid_score)
            
            # 更新最佳FID
            if fid_score < best_fid:
                best_fid = fid_score
                # 保存最佳模型
                self.model.save_checkpoint(
                    f'{self.config["save_dir"]}/best_model.pth',
                    epoch,
                    self.optimizer.state_dict()
                )
                print(f"💾 保存最佳模型 (FID: {best_fid:.2f})")
            
            print(f"🏆 当前最佳FID: {best_fid:.2f}")
        
        # 训练完成总结
        print("\n" + "=" * 60)
        print("🎉 云环境快速验证训练完成!")
        print(f"🏆 最佳FID: {best_fid:.2f}")
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 判断结果
        if best_fid < 50:
            print("✅ 成功！FID < 50，标准AutoencoderKL可以直接使用")
            print("💡 建议：继续用方案A进行完整训练")
            print("🚀 可以开始运行完整的LDM训练脚本")
        elif best_fid < 100:
            print("⚠️  FID在可接受范围，但建议Fine-tune VAE")
            print("💡 建议：转向方案B，先Fine-tune AutoencoderKL")
        else:
            print("❌ FID较高，需要进一步优化")
            print("💡 建议：检查数据质量或使用方案B/C")
        
        return best_fid
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 训练损失
        ax1.plot(self.train_history['epoch'], self.train_history['train_loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # FID分数
        ax2.plot(self.train_history['epoch'], self.train_history['fid_score'], 'r-o')
        ax2.axhline(y=50, color='g', linestyle='--', label='Target FID=50')
        ax2.set_title('FID Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FID')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.config["save_dir"]}/training_curves.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 训练曲线已保存到: {self.config['save_dir']}/training_curves.png")

def main():
    """主函数"""
    print("🌐 启动云环境快速验证训练...")
    
    # 检查数据路径
    data_paths = ['/kaggle/input/dataset', './dataset', '../dataset']
    data_dir = None
    
    for path in data_paths:
        if os.path.exists(path):
            data_dir = path
            print(f"✅ 找到数据集: {data_dir}")
            break
    
    if data_dir is None:
        print("❌ 未找到数据集，请检查路径")
        print("💡 请确保数据集已正确上传到云环境")
        return None
    
    trainer = CloudQuickValidationTrainer(data_dir=data_dir)
    best_fid = trainer.train()
    
    print(f"\n🎯 最终结果: FID = {best_fid:.2f}")
    
    return best_fid

if __name__ == "__main__":
    main() 
