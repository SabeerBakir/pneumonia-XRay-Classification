from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from PIL import Image
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
		image = np.asarray(image)
		sample = {'image': image, 'class': self.class_names[idx]}

		if self.transform:
			sample = self.transform(sample)

		return sample
		

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image = sample['image']


		return {'image': torch.from_numpy(image)}