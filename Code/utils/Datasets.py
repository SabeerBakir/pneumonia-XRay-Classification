from torch.utils.data.dataset import Dataset
from torchvision import datasets


class XRayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return self.dataset.__len__()
