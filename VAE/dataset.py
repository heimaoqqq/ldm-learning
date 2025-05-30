import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class GaitDataset(Dataset):
    def __init__(self, image_paths, class_ids, transform=None):
        self.transform = transform
        self.image_paths = image_paths
        self.class_ids = class_ids

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.class_ids[idx]
        
        # 添加标签范围验证（防御性编程）
        if label < 0 or label > 30:  # 应该在0-30范围内
            print(f"⚠️ 警告: 标签 {label} 超出范围 [0, 30]，来自文件: {img_path}")
            label = max(0, min(30, label))  # 强制限制在有效范围内
        
        return image, label

def build_dataloader(root_dir, batch_size=8, num_workers=2, shuffle_train=True, shuffle_val=False, val_split=0.3, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. 收集所有图片路径和对应的类别ID - 支持文件夹结构
    all_image_paths = []
    all_class_ids = []
    
    print(f"扫描数据集目录: {root_dir}")
    
    # 检查是否为文件夹结构 (ID_1, ID_2, ... ID_31)
    id_folders = []
    if os.path.exists(root_dir):
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and item.startswith('ID_'):
                id_folders.append(item)
    
    if id_folders:
        print(f"发现ID文件夹结构: {sorted(id_folders)}")
        
        # 处理文件夹结构
        for folder_name in sorted(id_folders):
            folder_path = os.path.join(root_dir, folder_name)
            
            # 从文件夹名提取类别ID
            try:
                if folder_name.startswith('ID_'):
                    class_id = int(folder_name.split('_')[1]) - 1  # 转换为0-based索引
                elif folder_name.startswith('ID'):
                    class_id = int(folder_name[2:]) - 1  # ID1, ID01等格式
                else:
                    continue
                
                # 验证class_id范围
                if not (0 <= class_id <= 30):
                    print(f"警告: 类别ID {class_id+1} 超出范围(1-31)，跳过文件夹: {folder_name}")
                    continue
                
                # 收集该文件夹中的所有图片
                folder_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    folder_images.extend(glob.glob(os.path.join(folder_path, ext)))
                
                print(f"  {folder_name}: 发现 {len(folder_images)} 张图片")
                
                # 添加到总列表
                all_image_paths.extend(folder_images)
                all_class_ids.extend([class_id] * len(folder_images))
                
            except (ValueError, IndexError) as e:
                print(f"警告: 无法解析文件夹名 {folder_name}: {e}")
                continue
    else:
        # 如果没有文件夹结构，则使用原来的文件名解析方法
        print("未发现ID文件夹结构，尝试从文件名解析...")
        
        for dir_path, _, _ in os.walk(root_dir):
            # 支持多种图片格式
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                all_image_paths.extend(glob.glob(os.path.join(dir_path, ext)))
        
        all_image_paths.sort()
        skipped_files = 0
        
        for img_path in all_image_paths:
            filename = os.path.basename(img_path)
            # 支持多种命名格式：
            # 1. ID_1, ID_2, ..., ID_31
            # 2. ID01, ID02, ..., ID31
            # 3. ID1, ID2, ..., ID31
            try:
                # 尝试提取ID部分
                if '_' in filename and filename.startswith('ID_'):
                    # 格式: ID_1_...jpg 或 ID_1.jpg
                    class_id_str = filename.split('_')[1]
                    if class_id_str.isdigit():
                        class_id = int(class_id_str) - 1
                    else:
                        # 可能包含其他信息，取第一个数字部分
                        import re
                        match = re.search(r'\d+', class_id_str)
                        if match:
                            class_id = int(match.group()) - 1
                        else:
                            raise ValueError(f"无法解析ID: {class_id_str}")
                elif filename.startswith('ID'):
                    # 格式: IDxx_...jpg 或 IDx_...jpg
                    id_part = filename[2:].split('_')[0]
                    if id_part.isdigit():
                        class_id = int(id_part) - 1
                    else:
                        raise ValueError(f"无法解析ID: {id_part}")
                else:
                    # 尝试从文件名或目录名中提取数字
                    import re
                    # 首先尝试从文件名提取
                    match = re.search(r'(\d+)', filename)
                    if match:
                        class_id = int(match.group()) - 1
                    else:
                        # 尝试从父目录名提取
                        parent_dir = os.path.basename(os.path.dirname(img_path))
                        match = re.search(r'(\d+)', parent_dir)
                        if match:
                            class_id = int(match.group()) - 1
                        else:
                            raise ValueError(f"无法从文件名或目录名提取ID")
                
                # 验证class_id范围
                if 0 <= class_id <= 30:
                    all_class_ids.append(class_id)
                else:
                    print(f"警告: 类别ID {class_id+1} 超出范围(1-31)，跳过文件: {filename}")
                    skipped_files += 1
                    continue
                
            except (ValueError, IndexError) as e:
                print(f"警告: 无法从文件名 {filename} 解析 class_id ({e})。跳过此文件。")
                skipped_files += 1
                continue
            
            # 移除跳过的文件
            if skipped_files > 0:
                valid_indices = []
                for i, img_path in enumerate(all_image_paths):
                    if i < len(all_class_ids):
                        valid_indices.append(i)
                
                all_image_paths = [all_image_paths[i] for i in valid_indices if i < len(all_class_ids)]
                print(f"跳过了 {skipped_files} 个无法解析的文件")

    if not all_image_paths or len(all_image_paths) != len(all_class_ids):
        raise ValueError(f"在目录 {root_dir} 中解析后没有有效的图片文件")

    # 打印数据集统计信息
    unique_classes = set(all_class_ids)
    print(f"\n数据集统计:")
    print(f"  发现的类别数: {len(unique_classes)} (期望31个)")
    print(f"  类别范围: {min(unique_classes)+1} 到 {max(unique_classes)+1}")
    print(f"  总图片数: {len(all_image_paths)}")
    
    # 统计每个类别的样本数
    from collections import Counter
    class_counts = Counter(all_class_ids)
    print("  各类别样本数:")
    for class_id in sorted(unique_classes):
        print(f"    ID_{class_id+1}: {class_counts[class_id]} 张图片")

    # 2. 划分数据集为训练集和验证集
    # 确保每个类别都有足够的样本进行分层采样
    min_samples = min(class_counts.values())
    if min_samples < 2:
        print(f"警告: 某些类别样本数少于2，将使用简单随机划分而非分层划分")
        train_paths, val_paths, train_ids, val_ids = train_test_split(
            all_image_paths, all_class_ids, test_size=val_split, random_state=random_state
        )
    else:
        train_paths, val_paths, train_ids, val_ids = train_test_split(
            all_image_paths, all_class_ids, test_size=val_split, random_state=random_state, stratify=all_class_ids
        )

    print(f"\n数据集划分完成:")
    print(f"  总样本数: {len(all_image_paths)}")
    print(f"  训练集样本数: {len(train_paths)}")
    print(f"  验证集样本数: {len(val_paths)}")
    if len(all_image_paths) > 0:
        print(f"  训练集比例: {len(train_paths)/len(all_image_paths):.2f}")
        print(f"  验证集比例: {len(val_paths)/len(all_image_paths):.2f}")

    # 3. 创建 Dataset 和 DataLoader
    train_dataset = GaitDataset(train_paths, train_ids, transform=transform)
    val_dataset = GaitDataset(val_paths, val_ids, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset), len(val_dataset) 
