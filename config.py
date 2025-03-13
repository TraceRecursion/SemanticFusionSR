import os

HOME_DIR = os.path.expanduser("~")
DATASET_ROOT = os.path.join(HOME_DIR, "Documents", "数据集", "COCO-Stuff")

TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train2017")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "val2017")
STUFF_ANN_DIR = os.path.join(DATASET_ROOT, "stuff_trainval2017")
STUFF_MASK_DIR = os.path.join(DATASET_ROOT, "stuffthingmaps_trainval2017")

TRAIN_ANN_FILE = os.path.join(STUFF_ANN_DIR, "stuff_train2017.json")
VAL_ANN_FILE = os.path.join(STUFF_ANN_DIR, "stuff_val2017.json")
TRAIN_MASK_DIR = os.path.join(STUFF_MASK_DIR, "train2017")
VAL_MASK_DIR = os.path.join(STUFF_MASK_DIR, "val2017")