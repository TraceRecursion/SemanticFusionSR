import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
from config import TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR, TRAIN_ANN_FILE, VAL_ANN_FILE


class CocoStuffSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ann_file, img_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        self.img_ids = [img['id'] for img in self.coco_data['images']]
        print(f"Loaded {len(self.img_ids)} images from {ann_file}")

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
        img_info = self.coco_data['images'][idx]
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
    train_dataset = CocoStuffSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_ANN_FILE)
    print(f"Dataset size: {len(train_dataset)}")