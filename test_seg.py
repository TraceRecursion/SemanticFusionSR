import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import os
import json
from config import VAL_IMG_DIR, VAL_MASK_DIR, VAL_ANN_FILE


class CocoStuffSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ann_file, img_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        # 加载 COCO-Stuff 标注
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        self.img_ids = [img['id'] for img in self.coco_data['images']]
        print(f"Loaded {len(self.img_ids)} images from {ann_file}")

        # 图像预处理
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

        # 加载 PNG 格式的分割掩码
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            mask = self.mask_transform(mask) * 255  # 转换为类别索引
            mask = mask.squeeze(0).long()  # [H, W]
        else:
            print(f"Mask not found for {img_info['file_name']}")
            mask = torch.zeros(self.img_size, dtype=torch.long)

        return img_tensor, mask


def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def test_seg(token=None):
    # 数据集
    val_dataset = CocoStuffSegDataset(
        img_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        ann_file=VAL_ANN_FILE
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)  # 先用 batch_size=1 调试

    # 模型
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config = SegformerConfig.from_pretrained("nvidia/segformer-b5", num_labels=182)  # 调整为 182 类
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5",
        config=config,
        token=token
    ).to(device)
    model.eval()

    # 评估指标
    num_classes = 182
    all_ious = [[] for _ in range(num_classes)]
    num_samples = len(val_loader)

    # 进度条
    print(f"Testing SegFormer on {num_samples} samples...")
    start_time = time.time()
    with torch.no_grad():
        for img, mask in tqdm(val_loader, desc="Testing Segmentation", unit="batch"):
            img, mask = img.to(device), mask.to(device)
            outputs = model(img).logits  # [B, 182, H/4, W/4]
            pred = torch.argmax(outputs, dim=1)  # [B, H/4, W/4]
            pred = F.interpolate(pred.float().unsqueeze(0), size=mask.shape[-2:], mode='nearest').squeeze(0).long()
            ious = calculate_iou(pred, mask, num_classes)
            for cls, iou in enumerate(ious):
                if not np.isnan(iou):
                    all_ious[cls].append(iou)

    # 计算每个类的 IoU 和 mIoU
    class_ious = [np.mean(ious) if ious else float('nan') for ious in all_ious]
    miou = np.nanmean(class_ious)
    elapsed_time = time.time() - start_time

    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    print(f"mIoU: {miou:.4f}")
    for cls, iou in enumerate(class_ious):
        if not np.isnan(iou):
            print(f"Class {cls} IoU: {iou:.4f}")


if __name__ == "__main__":
    test_seg()