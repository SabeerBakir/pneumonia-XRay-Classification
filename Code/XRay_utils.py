from __future__ import print_function, division
import os
import sys
import collections
import numbers

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
from torchvision.transforms import functional as transf_f
import torch.nn.functional as F

import pandas as pd

from PIL import Image
from PIL import ImageOps

import numpy as np
import matplotlib.pyplot as plt

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class XRayDataset(Dataset):
    """Train Normal Dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        
        names = [item for item in os.listdir(root_dir) if not item.startswith('.')]
        
        self.class_names = []
        self.image_names = []
        for class_name in names:
            self.image_names += [item for item in os.listdir(os.path.join(root_dir, class_name)) if not item.startswith('.')]
            for i in range(len(self.image_names)):
                self.class_names += [class_name]

        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.class_names[idx] ,self.image_names[idx])
        image = Image.open(img_name)
        image = ImageOps.grayscale(image)
        # image = np.asarray(image)
        sample = {'image': image, 'class': self.class_names[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
        

# From PyTorch Documentation, edited for this project
class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        padding (tuple): Padding on each border. Pad image to a specified size specified
        by a tuple.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        max_image_size = np.array(self.padding)
        image = img['image']
        # Check final image size is equal or larger than input image
        assert max_image_size[0] >= image.size[0]
        assert max_image_size[1] >= image.size[1]

        padding = max_image_size - np.array(image.size)
        padding = tuple(padding // 2)
        new_image = transf_f.pad(image, padding, self.fill, 'constant')
        if new_image.size is not self.padding:
            # check for odd height
            if (new_image.size[0] % 2 != 0) and (new_image.size[1] % 2 == 0):
                padding = (padding[0], padding[1], padding[0] + 1, padding[1])
                new_image = transf_f.pad(image, padding, self.fill, 'constant')
            # check for odd width
            elif (new_image.size[0] % 2 == 0) and (new_image.size[1] % 2 != 0):
                padding = (padding[0], padding[1], padding[0], padding[1] + 1)
                new_image = transf_f.pad(image, padding, self.fill, 'constant')
            # check for both odd height and width
            elif (new_image.size[0] % 2 != 0) and (new_image.size[1] % 2 != 0):
                padding = (padding[0], padding[1], padding[0] + 1, padding[1] + 1)
                new_image = transf_f.pad(image, padding, self.fill, 'constant')

        return {'image': new_image, 'class': img['class']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        return {'image': torch.from_numpy(np.asarray(image)), 'class': sample['class']}


class Rescale(object):
    """Rescale the image in a sample by a given factor.

    Args:
        scaler (float or int): A Scaler value to enlarge or reduce the size of an image.
    """

    def __init__(self, scaler):
        assert isinstance(scaler, (int, float))
        self.scaler = scaler

    def __call__(self, sample):
        img = sample['image']

        h, w = img.size

        new_h = h * self.scaler
        new_w = w * self.scaler

        new_h, new_w = int(new_h), int(new_w)

        return {'image': img.resize((new_h, new_w)), 'class': sample['class']}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x.float())), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x.float())), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def value_counts(data_folder):
    """Outputs value counts

    :param data_folder: Folder containing data.
    """
    classes = os.listdir(data_folder)

    for label in classes:
        print(label + ': ' + str(len([i for i in os.listdir(data_folder + os.sep + label) if i.endswith('.jpeg')])))


def label_to_num(labels):
    """
    Function to return numeric labels

    :param labels: an array of labels
    :return: an array of numbers instead of labels
    """

    return np.array(list(map(lambda x: 0 if x == 'NORMAL' else 1, labels)))
