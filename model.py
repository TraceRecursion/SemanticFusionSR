import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial attention
        sa = self.spatial_attention(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        x = x * sa
        return x

class SemanticFusionSR(nn.Module):
    def __init__(self, num_classes=182, scale_factor=4):  # COCO-Stuff 有 182 类
        super(SemanticFusionSR, self).__init__()
        self.scale_factor = scale_factor
        
        # 语义特征提取：SegFormer-B5
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-coco-stuff-640-1280")
        self.embedding = nn.Conv2d(num_classes, 64, 1)  # 将语义分割结果转为 64 维特征
        
        # 低级特征提取：简化的 ResNet-like 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合模块
        self.fusion_conv = nn.Conv2d(64 + 256, 256, 1)
        self.cbam = CBAM(256)
        
        # 上采样和重建网络
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 64 * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取语义特征
        seg_output = self.segformer(x).logits  # [B, 182, H/4, W/4]
        sem_feature = self.embedding(F.interpolate(seg_output, scale_factor=4, mode='bilinear'))  # [B, 64, H, W]
        
        # 提取低级特征
        low_feature = self.encoder(x)  # [B, 256, H/4, W/4]
        low_feature = F.interpolate(low_feature, size=sem_feature.shape[2:], mode='bilinear')  # 对齐分辨率
        
        # 特征融合
        fused_feature = torch.cat([sem_feature, low_feature], dim=1)  # [B, 320, H, W]
        fused_feature = self.fusion_conv(fused_feature)  # [B, 256, H, W]
        fused_feature = self.cbam(fused_feature)
        
        # 重建 HR 图像
        hr_output = self.decoder(fused_feature)
        return hr_output

# 示例用法
if __name__ == "__main__":
    model = SemanticFusionSR()
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(f"Output shape: {y.shape}")  # 应为 [1, 3, 512, 512]