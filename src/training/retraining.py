import os

import segmentation_models_pytorch as smp
import torch
import torch.utils.data as D
from sklearn.model_selection import train_test_split

from src.data_scripts.dataset import TrainDataset
from src.utills.config import CFG
from src.utills.helpers import get_image_masks_path
from src.utills.inference.transforms import get_transforms

paths = get_image_masks_path(CFG)
train_paths, val_paths = train_test_split(
    paths, test_size=0.2, random_state=CFG.SEED, shuffle=True
)

train_dataset = TrainDataset(
    train_paths,
    transform=get_transforms(data="train"),
)
valid_dataset = TrainDataset(
    val_paths,
    transform=get_transforms(data="valid"),
)

train_loader = D.DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
valid_loader = D.DataLoader(
    valid_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
)


model = smp.DeepLabV3Plus(
    encoder_name="se_resnext50_32x4d", encoder_weights="imagenet", activation="sigmoid"
)
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, amsgrad=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

best_loss = 1.0
train_losses, val_losses = [], []
train_scores, val_scores = [], []


for i in range(0, CFG.NUM_EPOCHS):
    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    train_losses.append(train_logs["dice_loss"])
    val_losses.append(valid_logs["dice_loss"])
    train_scores.append(train_logs["iou_score"])
    val_scores.append(valid_logs["iou_score"])

    if best_loss > valid_logs["dice_loss"]:
        best_loss = valid_logs["dice_loss"]
        torch.save(model, os.path.join(CFG.MODEL_SAVE_DIR, "model_retrained.pth"))
        print("Model saved!")
