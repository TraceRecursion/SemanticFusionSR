import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np
from config import TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR, TRAIN_ANN_FILE, VAL_ANN_FILE


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
            mask = self.mask_transform(mask) * 255  # 转换为类别索引 (0-181)
            mask = mask.squeeze(0).long()  # [H, W]
        else:
            print(f"Mask not found for {img_info['file_name']}")
            mask = torch.zeros(self.img_size, dtype=torch.long)

        return img_tensor, mask


if __name__ == "__main__":
    train_dataset = CocoStuffSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_ANN_FILE, sample_fraction=0.1)
    print(f"Dataset size: {len(train_dataset)}")