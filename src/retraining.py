import os
import random
from os import path

import cv2 as cv
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.utils.data as D
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

TRAIN_DATA_DIR = path.realpath(path.curdir) + '/Data/256x256/'
TEST_DATA_DIR = path.realpath(path.curdir) + '/Data/'
MODEL_SAVE_DIR = path.realpath(path.curdir) + "/models/"
TILE_SIZE = 256
REDUCE_RATE = 4
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 20


train_img_paths, train_mask_paths = [], []
for dt in ['train/', 'masks/']:
    for item in os.listdir(TRAIN_DATA_DIR + dt):
        if dt == 'train/':
            train_img_paths.append(TRAIN_DATA_DIR + 'train/' + item)
        elif dt == 'masks/':
            train_mask_paths.append(TRAIN_DATA_DIR + 'masks/' + item)

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
            augmented = self.transform(image=image, mask=mask)  # image np(256,256,3) -> tensor(3,256,256)
            image, mask = augmented['image'], augmented['mask']
            mask = mask.unsqueeze(0)  # (256,256) -> (1, 256, 256)
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
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),

            A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.2, p=0.2, border_mode=0),

            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),

            A.OneOf([
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ], p=0.9,
            ),

            A.OneOf([
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ], p=0.9,
            ),

            A.OneOf([
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1), ], p=0.9,
            ),
            A.Compose([
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)]
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.ToTensorV2(),  # ToTensor(num_classes=2),
        ])
    elif data == 'valid':
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.ToTensorV2(),
        ])


train_paths, val_paths = train_test_split(paths, test_size=0.2, random_state=SEED, shuffle=True)
train_dataset = TrainDataset(train_paths, transform=get_transforms(data='train'), )
valid_dataset = TrainDataset(val_paths, transform=get_transforms(data='valid'), )

train_loader = D.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = D.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ====================================================
# Modeling
# ====================================================
model = smp.DeepLabV3Plus(encoder_name='se_resnext50_32x4d', encoder_weights='imagenet', activation='sigmoid')

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
