import torch
from torch.utils.data import DataLoader
from dataset import CocoStuffSRDataset
from model import SemanticFusionSR
from utils import calculate_psnr, calculate_ssim
from torch.cuda.amp import GradScaler, autocast
import os

def train():
    # 数据集路径
    train_dataset = CocoStuffSRDataset(
        img_dir="/Users/sydg/Documents/数据集/COCO-Stuff/train2017",
        ann_file="/Users/sydg/Documents/数据集/COCO-Stuff/stuff_trainval2017/stuff_train2017.json"
    )
    val_dataset = CocoStuffSRDataset(
        img_dir="/Users/sydg/Documents/数据集/COCO-Stuff/val2017",
        ann_file="/Users/sydg/Documents/数据集/COCO-Stuff/stuff_trainval2017/stuff_val2017.json"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=13, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=13, shuffle=False, num_workers=4)
    
    # 模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticFusionSR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scaler = GradScaler()
    
    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            optimizer.zero_grad()
            
            with autocast():
                sr_img = model(lr_img)
                loss = torch.abs(sr_img - hr_img).mean()  # L1 损失
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_psnr, val_ssim = 0, 0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                sr_img = model(lr_img)
                val_psnr += calculate_psnr(sr_img, hr_img)
                val_ssim += calculate_ssim(sr_img, hr_img)
        
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)
        
        scheduler.step(train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()