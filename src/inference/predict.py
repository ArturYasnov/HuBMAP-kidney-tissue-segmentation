import gc
import os
import pathlib
from os import path

import cv2
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch

from src.utills.config import CFG
from src.utills.helpers import make_grid, rle_numba_encode
from src.utills.inference.transforms import get_val_transforms

model = smp.DeepLabV3Plus(
    encoder_name="se_resnext50_32x4d", encoder_weights="imagenet", activation="sigmoid"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(
    os.path.join(path.realpath(path.curdir) + CFG.MODEL_SAVE_DIR, "model.pth"),
    map_location=torch.device(device),
)
model.to(device)
model.eval()


identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
transforms = get_val_transforms()
p = pathlib.Path(CFG.IMAGE_DIR)

subm = {}

for i, filename in enumerate(p.glob("*.tiff")):

    dataset = rasterio.open(filename.as_posix(), transform=identity)
    slices = make_grid(dataset.shape, window=CFG.WINDOW, min_overlap=CFG.MIN_OVERLAP)
    preds = np.zeros(dataset.shape, dtype=np.uint8)

    for (x1, x2, y1, y2) in slices:
        image = dataset.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
        image = np.moveaxis(image, 0, -1)
        image = transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image)

        with torch.no_grad():
            image = image.float().to(device)[None]
            score = model(image)
            score = score.cpu().numpy()[0][0]
            score = cv2.resize(score, (CFG.WINDOW, CFG.WINDOW))

        preds[x1:x2, y1:y2] = (score > 0.5).astype(np.uint8)
    subm[i] = {"id": filename.stem, "predicted": rle_numba_encode(preds)}
    del preds
    gc.collect()

output = pd.DataFrame.from_dict(subm, orient="index")
output.to_csv(path.realpath(path.curdir) + "/Data/Output/masks.csv", index=False)
