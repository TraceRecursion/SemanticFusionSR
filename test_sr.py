import torch
from torch.utils.data import DataLoader
from dataset import CocoStuffSRDataset
from model import SemanticFusionSR
from utils import calculate_psnr, calculate_ssim
from tqdm import tqdm
import time

def test_sr(model_path):
    # 数据集
    val_dataset = CocoStuffSRDataset(
        img_dir="/Users/sydg/Documents/数据集/COCO-Stuff/val2017",
        ann_file="/Users/sydg/Documents/数据集/COCO-Stuff/stuff_trainval2017/stuff_val2017.json"
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)  # batch_size=1 以精确评估
    
    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SemanticFusionSR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 评估指标
    total_psnr, total_ssim = 0, 0
    num_samples = len(val_loader)
    
    # 进度条
    print(f"Testing SR model on {num_samples} samples...")
    start_time = time.time()
    with torch.no_grad():
        for lr_img, hr_img in tqdm(val_loader, desc="Testing SR", unit="batch"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr_img = model(lr_img)
            total_psnr += calculate_psnr(sr_img, hr_img)
            total_ssim += calculate_ssim(sr_img, hr_img)
    
    # 计算平均值
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    # 指定训练好的模型路径
    model_path = "checkpoints/model_epoch_50.pth"  # 替换为你的模型路径
    test_sr(model_path)