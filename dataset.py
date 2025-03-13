import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np

class CocoStuffSRDataset(Dataset):
    def __init__(self, img_dir, ann_file, scale_factor=4, img_size=(512, 512)):
        self.img_dir = img_dir
        self.scale_factor = scale_factor
        self.img_size = img_size
        
        # 加载 COCO-Stuff 标注
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        self.img_ids = [img['id'] for img in self.coco_data['images']]
        
        # 图像预处理
        self.hr_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.lr_transform = transforms.Compose([
            transforms.Resize((img_size[0] // scale_factor, img_size[1] // scale_factor)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info = self.coco_data['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # 生成 HR 和 LR 图像
        hr_img = self.hr_transform(image)
        lr_img = self.lr_transform(image)
        
        return lr_img, hr_img

# 示例用法
if __name__ == "__main__":
    train_dataset = CocoStuffSRDataset(
        img_dir="/Users/sydg/Documents/数据集/COCO-Stuff/train2017",
        ann_file="/Users/sydg/Documents/数据集/COCO-Stuff/stuff_trainval2017/stuff_train2017.json"
    )
    print(f"Dataset size: {len(train_dataset)}")