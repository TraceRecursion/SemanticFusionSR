import torch
import torch.nn.functional as F

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    # 简化的 SSIM 计算，需安装额外库（如 pytorch-msssim）以获得更精确结果
    from pytorch_msssim import ssim
    return ssim(img1, img2, data_range=1.0, size_average=True).item()