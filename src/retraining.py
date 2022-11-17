import os
import gc
import json
import random
import numpy as np
import pandas as pd
import tifffile as tiff
import cv2 as cv
import albumentations as albu
from os import path

import pathlib, sys, os, random, time
import numba, gc, cv2
import rasterio
from rasterio.windows import Window

from tqdm.notebook import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data as D
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from albumentations import (
    Compose, OneOf, Normalize, HorizontalFlip, VerticalFlip, RandomRotate90, Transpose, ShiftScaleRotate, IAAAdditiveGaussianNoise, IAAPerspective,
    CLAHE, RandomBrightness, RandomGamma, IAASharpen, Blur, MotionBlur, RandomContrast, HueSaturationValue, VerticalFlip,
    RandomRotate90, OneOf, Resize, Rotate, RandomBrightnessContrast, Lambda
    )
from albumentations.pytorch import ToTensorV2, ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


TRAIN_DATA_DIR = path.realpath(path.curdir)+'/Data/256x256/'
TEST_DATA_DIR = path.realpath(path.curdir)+'/Data/'
MODEL_SAVE_DIR = path.realpath(path.curdir)+"/models/"
TILE_SIZE = 256
REDUCE_RATE = 4
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


train_img_paths, train_mask_paths = [], []
for dt in ['train/', 'masks/']:
    for item in os.listdir(TRAIN_DATA_DIR+dt):
        if dt=='train/':
            train_img_paths.append(TRAIN_DATA_DIR+'train/'+item)
        elif dt=='masks/':
            train_mask_paths.append(TRAIN_DATA_DIR+'masks/'+item)  
            
paths = list(zip(train_img_paths, train_mask_paths))


# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = cv.imread(self.paths[i][0])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(self.paths[i][1], 0)  # 0=cv2.IMREAD_GRAYSCALE, np.array of 0 and 1. (256,256)
        if self.transform:
            augmented = self.transform(image=image, mask=mask) # image np(256,256,3) -> tensor(3,256,256)
            image, mask = augmented['image'], augmented['mask']
            mask = mask.unsqueeze(0) # (256,256) -> (1, 256, 256)
        return image, mask
    
class TestDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv.imread(self.paths[i][0])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    
    if data == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Transpose(p=0.5),

            ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.2, p=0.2, border_mode=0),

            IAAAdditiveGaussianNoise(p=0.2),
            IAAPerspective(p=0.5),

            OneOf([
                    CLAHE(p=1),
                    RandomBrightness(p=1),
                    RandomGamma(p=1),
                ], p=0.9,
            ),

            OneOf([
                IAASharpen(p=1),
                Blur(blur_limit=3, p=1),
                MotionBlur(blur_limit=3, p=1),
            ], p=0.9,
            ),

            OneOf([   
                RandomContrast(p=1),
                HueSaturationValue(p=1),],  p=0.9,
            ),
            Compose([
                VerticalFlip(p=0.5),              
                RandomRotate90(p=0.5)]
            ),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),  #ToTensor(num_classes=2),
        ])    
    elif data == 'valid':
        return Compose([
            Resize(256,256),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


train_paths, val_paths = train_test_split(paths, test_size=0.2, random_state=SEED, shuffle=True)
train_dataset = TrainDataset(train_paths, transform=get_transforms(data='train'),)
valid_dataset = TrainDataset(val_paths, transform=get_transforms(data='valid'),)

train_loader = D.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = D.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ====================================================
# Modeling
# ====================================================
model = smp.Unet(encoder_name='se_resnext50_32x4d', encoder_weights='imagenet', activation='sigmoid')
# model = smp.FPN(encoder,encoder_weights=encoder_weights)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, amsgrad=False)


train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=device,
    verbose=True,
)

n_epochs = 10
best_loss = 1.0
train_losses, val_losses = [], []
train_scores, val_scores = [], []

for i in range(0, n_epochs):
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    train_losses.append(train_logs['dice_loss'])
    val_losses.append(valid_logs['dice_loss'])
    train_scores.append(train_logs['iou_score'])
    val_scores.append(valid_logs['iou_score'])

    if best_loss > valid_logs['dice_loss']:
        best_loss = valid_logs['dice_loss']
        torch.save(model, os.path.join(MODEL_SAVE_DIR, 'model_retrained.pth'))
        print('Model saved!')






