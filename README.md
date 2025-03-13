

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install transformers Pillow tqdm numpy pytorch-msssim tensorboard
```

```shell
tensorboard --logdir runs/segformer_b0_coco_stuff --host=100.74.5.7 --port=6006
```