from torch.utils.data.dataset import Dataset
from torchvision import datasets
import numpy as np


class XRayDataset(Dataset):
    def __init__(self, data_dir, transform=None, augmentations=None):
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)
        self.augmentations = augmentations

    def __getitem__(self, index):
        if self.augmentations is not None:
            data = self.augmentations(image=np.asarray(self.dataset.__getitem__(index)[0]))
            return data['image'], self.dataset.__getitem__(index)[1]
        else:
            return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()
