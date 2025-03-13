```shell
SemanticFusionSR/
├── dataset.py          # 数据集加载和预处理
├── model.py           # 模型定义
├── train.py           # 训练脚本
├── utils.py           # 工具函数（损失、指标等）
└── README.md         # 项目说明
```

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install transformers Pillow tqdm numpy pytorch-msssim tensorboard
```