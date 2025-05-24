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
        return image, label

def build_dataloader(root_dir, batch_size=8, num_workers=2, shuffle_train=True, shuffle_test=False, test_split=0.3, random_state=42):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. 收集所有图片路径和对应的类别ID
    all_image_paths = []
    for dir_path, _, _ in os.walk(root_dir):
        all_image_paths.extend(glob.glob(os.path.join(dir_path, "*.jpg")))
    all_image_paths.sort()

    if not all_image_paths:
        raise ValueError(f"在目录 {root_dir} 中没有找到 .jpg 文件")

    all_class_ids = []
    for img_path in all_image_paths:
        filename = os.path.basename(img_path)
        # 假设文件名格式为 IDxx_...jpg，例如 ID01_seq01_angle090_001.jpg
        try:
            class_id_str = filename.split('_')[0].replace('ID', '')
            class_id = int(class_id_str) - 1
            all_class_ids.append(class_id)
        except ValueError:
            print(f"警告: 无法从文件名 {filename} 解析 class_id。跳过此文件。")
            continue

    if not all_image_paths or len(all_image_paths) != len(all_class_ids):
        valid_indices = [i for i, path in enumerate(all_image_paths) if os.path.basename(path).split('_')[0].replace('ID', '').isdigit()]
        all_image_paths = [all_image_paths[i] for i in valid_indices]
        all_class_ids = [all_class_ids[i] for i in valid_indices]
        if not all_image_paths:
            raise ValueError(f"在目录 {root_dir} 中解析后没有有效的 .jpg 文件")

    # 2. 划分数据集
    train_paths, test_paths, train_ids, test_ids = train_test_split(
        all_image_paths, all_class_ids, test_size=test_split, random_state=random_state, stratify=all_class_ids
    )

    print(f"数据集划分完成:")
    print(f"  总样本数: {len(all_image_paths)}")
    print(f"  训练集样本数: {len(train_paths)}")
    print(f"  测试集样本数: {len(test_paths)}")
    if len(all_image_paths) > 0:
        print(f"  训练集比例: {len(train_paths)/len(all_image_paths):.2f}")
        print(f"  测试集比例: {len(test_paths)/len(all_image_paths):.2f}")

    # 3. 创建 Dataset 和 DataLoader
    train_dataset = GaitDataset(train_paths, train_ids, transform=transform)
    test_dataset = GaitDataset(test_paths, test_ids, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset) 