import cv2 as cv
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = cv.imread(self.paths[i][0])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mask = cv.imread(
            self.paths[i][1], 0
        )  # 0=cv2.IMREAD_GRAYSCALE, np.array of 0 and 1. (256,256)
        if self.transform:
            augmented = self.transform(
                image=image, mask=mask
            )  # image np(256,256,3) -> tensor(3,256,256)
            image, mask = augmented["image"], augmented["mask"]
            mask = mask.unsqueeze(0)  # (256,256) -> (1, 256, 256)
        return image, mask


class TestDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv.imread(self.paths[idx][0])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image
