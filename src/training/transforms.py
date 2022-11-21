import albumentations as A


def get_transforms(*, data):
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(
                    scale_limit=0.2,
                    rotate_limit=0,
                    shift_limit=0.2,
                    p=0.2,
                    border_mode=0,
                ),
                A.IAAAdditiveGaussianNoise(p=0.2),
                A.IAAPerspective(p=0.5),
                A.OneOf(
                    [
                        A.CLAHE(p=1),
                        A.RandomBrightness(p=1),
                        A.RandomGamma(p=1),
                    ],
                    p=0.9,
                ),
                A.OneOf(
                    [
                        A.IAASharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.9,
                ),
                A.OneOf(
                    [
                        A.RandomContrast(p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.9,
                ),
                A.Compose([A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.ToTensorV2(),
            ]
        )
