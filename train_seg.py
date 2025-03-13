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
    train_dataset = CocoStuffSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TRAIN_ANN_FILE, sample_fraction=0.3)
    val_dataset = CocoStuffSegDataset(VAL_IMG_DIR, VAL_MASK_DIR, VAL_ANN_FILE, sample_fraction=0.4)

    # 分析数据集类别分布
    print("分析训练集类别分布...")
    train_class_pixels, train_class_images = train_dataset.analyze_class_distribution()
    print("分析验证集类别分布...")
    val_class_pixels, val_class_images = val_dataset.analyze_class_distribution()

    # 查找验证集中存在的类别
    val_classes = set(np.where(val_class_pixels > 0)[0])
    print(f"验证集中存在 {len(val_classes)} 个类别: {sorted(val_classes)}")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=12)

    # 模型（SegFormer-B0）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用from_pretrained方法加载整个模型
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=182,
        ignore_mismatched_sizes=True  # 允许修改分类器层而不报错
    ).to(device)

    # 差异化学习率：编码器使用较小的学习率，解码器（新初始化的部分）使用较大的学习率
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if "decode_head" in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 0.00002},  # 编码器使用较小学习率
        {'params': decoder_params, 'lr': 0.0001}  # 解码器使用较大学习率
    ], weight_decay=0.01)

    # 使用学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/segformer_b0_coco_stuff")

    # 训练参数
    num_epochs = 50
    num_classes = 182

    # 记录数据集类别分布到TensorBoard
    for cls in range(num_classes):
        if train_class_pixels[cls] > 0:
            writer.add_scalar(f"Dataset/TrainClassPixels_{cls}", train_class_pixels[cls], 0)
        if val_class_pixels[cls] > 0:
            writer.add_scalar(f"Dataset/ValClassPixels_{cls}", val_class_pixels[cls], 0)

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
                outputs = nn.functional.interpolate(outputs, size=mask.shape[-2:], mode='nearest')
                loss = criterion(outputs, mask)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)
        scheduler.step()  # 更新学习率

        # 验证
        model.eval()
        val_loss = 0
        all_ious = [[] for _ in range(num_classes)]
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", unit="batch") as pbar:
                for img, mask in pbar:
                    img, mask = img.to(device), mask.to(device)
                    outputs = model(img).logits
                    outputs = nn.functional.interpolate(outputs, size=mask.shape[-2:], mode='nearest')
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

        # 记录每个类别的IoU (如果该类别在验证集中存在)
        non_zero_ious = 0
        for cls in range(num_classes):
            if not np.isnan(class_ious[cls]) and cls in val_classes:
                writer.add_scalar(f"IoU/Class_{cls}", class_ious[cls], epoch)
                non_zero_ious += 1

        # 记录当前学习率
        for param_group_idx, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LearningRate/group_{param_group_idx}", param_group['lr'], epoch)

        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mIoU: {miou:.4f}")
        print(
            f"Learning rates: Encoder={optimizer.param_groups[0]['lr']:.6f}, Decoder={optimizer.param_groups[1]['lr']:.6f}")

        # 打印前10个有效类别的IoU
        print("\n前10个类别IoU:")
        valid_classes = [(cls, iou) for cls, iou in enumerate(class_ious) if not np.isnan(iou) and cls in val_classes]
        for i, (cls, iou) in enumerate(valid_classes[:10]):
            print(f"Class {cls}: {iou:.4f}", end="\t")
            if (i + 1) % 5 == 0:
                print()  # 每5个类别换行
        print("\n")

        # 保存模型
        torch.save(model.state_dict(), f"checkpoints/segformer_b0_epoch_{epoch + 1}.pth")

    # 最终IoU详细报告
    print("\n=== 最终验证集IoU报告 ===")
    for cls in range(num_classes):
        if not np.isnan(class_ious[cls]) and cls in val_classes:
            print(
                f"类别 {cls}: IoU = {class_ious[cls]:.4f}, 像素数 = {val_class_pixels[cls]}, 图像数 = {val_class_images[cls]}")

    writer.close()


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_seg()