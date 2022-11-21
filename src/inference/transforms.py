import albumentations as A


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # ToTensorV2(),
        ]
    )
