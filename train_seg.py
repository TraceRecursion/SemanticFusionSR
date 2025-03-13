import os
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import numpy as np
from dataset import CocoStuffSegDataset
from config import TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_MASK_DIR, VAL_MASK_DIR, TRAIN_ANN_FILE, VAL_ANN_FILE


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


def train_seg():
    # 数据集（抽样）
    train_dataset = CocoStuffSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_ANN_FILE, sample_fraction=0.1)
    val_dataset = CocoStuffSegDataset(VAL_IMG_DIR, VAL_MASK_DIR, VAL_ANN_FILE, sample_fraction=0.2)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=12)

    # 模型（SegFormer-B0）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = SegformerConfig.from_pretrained("nvidia/mit-b0", num_labels=182)
    model = SegformerForSemanticSegmentation(config=config).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/segformer_b0_coco_stuff")

    # 训练参数
    num_epochs = 10
    num_classes = 182

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        start_time = time.time()

        # 训练进度条
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", unit="batch") as pbar:
            for img, mask in pbar:
                img, mask = img.to(device), mask.to(device)
                optimizer.zero_grad()

                outputs = model(img).logits
                outputs = nn.functional.interpolate(outputs, size=mask.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, mask)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        all_ious = [[] for _ in range(num_classes)]
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", unit="batch") as pbar:
                for img, mask in pbar:
                    img, mask = img.to(device), mask.to(device)
                    outputs = model(img).logits
                    outputs = nn.functional.interpolate(outputs, size=mask.shape[-2:], mode='bilinear',
                                                        align_corners=False)
                    loss = criterion(outputs, mask)
                    val_loss += loss.item()

                    pred = torch.argmax(outputs, dim=1)
                    ious = calculate_iou(pred, mask, num_classes)
                    for cls, iou in enumerate(ious):
                        if not np.isnan(iou):
                            all_ious[cls].append(iou)

                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        val_loss /= len(val_loader)
        class_ious = [np.mean(ious) if ious else float('nan') for ious in all_ious]
        miou = np.nanmean(class_ious)

        # TensorBoard 记录
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("mIoU/Val", miou, epoch)
        for cls, iou in enumerate(class_ious):
            if not np.isnan(iou):
                writer.add_scalar(f"IoU/Class_{cls}", iou, epoch)

        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {miou:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f"checkpoints/segformer_b0_epoch_{epoch + 1}.pth")

    writer.close()


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_seg()