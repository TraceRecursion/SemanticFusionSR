import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np
from config import TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR, TRAIN_ANN_FILE, VAL_ANN_FILE
from tqdm import tqdm


class CocoStuffSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ann_file, img_size=(512, 512), sample_fraction=1.0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        # 加载 COCO-Stuff 标注
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # 获取所有图像信息和类别分布
        all_images = self.coco_data['images']
        annotations = {ann['image_id']: ann for ann in self.coco_data['annotations']}

        # 计算每个图像的类别数，用于加权抽样
        image_weights = []
        for img in all_images:
            ann = annotations.get(img['id'])
            if ann and 'category_id' in ann:
                # 假设每个图像至少有一个主要类别
                image_weights.append(1.0)  # 简单起见，这里用均匀权重
            else:
                image_weights.append(0.1)  # 缺少标注的图像降低权重

        # 加权随机抽样
        num_samples = int(len(all_images) * sample_fraction)
        sampled_indices = np.random.choice(
            len(all_images),
            size=num_samples,
            replace=False,
            p=np.array(image_weights) / np.sum(image_weights)
        )
        self.img_ids = [all_images[i]['id'] for i in sampled_indices]
        self.images = [all_images[i] for i in sampled_indices]
        print(f"Loaded {len(self.img_ids)} images from {ann_file} (sampled {sample_fraction * 100:.1f}%)")

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        mask_path = os.path.join(self.mask_dir, img_info['file_name'].replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image)

        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = self.mask_transform(mask)  # 已经移除 * 255

            # 检查掩码值的范围，确保在预期范围内 (0-181)
            mask_min = mask.min()
            mask_max = mask.max()
            if mask_max > 181:
                print(f"警告：掩码值超出预期范围：min={mask_min}, max={mask_max}")

            mask = mask.squeeze(0).long()  # [H, W]
        else:
            print(f"Mask not found for {img_info['file_name']}")
            mask = torch.zeros(self.img_size, dtype=torch.long)

        return img_tensor, mask

    def analyze_class_distribution(self, num_classes=182):
        """分析数据集中各类别的像素数量分布"""
        print(f"分析数据集 '{os.path.basename(self.mask_dir)}' 的类别分布...")
        class_pixels = np.zeros(num_classes, dtype=np.int64)
        class_images = np.zeros(num_classes, dtype=np.int32)  # 记录每个类别出现在多少张图像中

        for i in tqdm(range(len(self)), desc="统计类别分布"):
            _, mask = self[i]
            unique_classes, counts = np.unique(mask.numpy(), return_counts=True)

            # 统计每个类别的像素数
            for cls, count in zip(unique_classes, counts):
                if cls < num_classes:  # 确保类别索引在有效范围内
                    class_pixels[cls] += count
                    class_images[cls] += 1

        # 输出统计结果
        print("\n类别分布统计:")
        print("-" * 60)
        print(f"{'类别ID':<8}{'像素数量':<15}{'图像数量':<10}{'占比(%)':<10}")
        print("-" * 60)

        non_zero_classes = 0
        for cls in range(num_classes):
            if class_pixels[cls] > 0:
                non_zero_classes += 1
                percentage = class_images[cls] / len(self) * 100
                print(f"{cls:<8}{class_pixels[cls]:<15}{class_images[cls]:<10}{percentage:.2f}")

        print("-" * 60)
        print(f"总计: 发现 {non_zero_classes}/{num_classes} 个类别")

        return class_pixels, class_images


# 添加一个辅助函数来检查数据集的掩码值范围
def check_mask_values(dataset, num_samples=10):
    print("检查掩码值范围...")
    for i in range(min(num_samples, len(dataset))):
        _, mask = dataset[i]
        print(f"样本 {i}: 掩码值范围 min={mask.min().item()}, max={mask.max().item()}")


if __name__ == "__main__":
    train_dataset = CocoStuffSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_ANN_FILE, sample_fraction=0.1)
    print(f"Dataset size: {len(train_dataset)}")

    # 检查一些样本的掩码值
    check_mask_values(train_dataset)

    # 分析训练集的类别分布
    train_dataset.analyze_class_distribution()

    # 也可以分析验证集的类别分布
    val_dataset = CocoStuffSegDataset(VAL_IMG_DIR, VAL_MASK_DIR, VAL_ANN_FILE, sample_fraction=0.2)
    val_dataset.analyze_class_distribution()