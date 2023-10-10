"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision
import random

from ..consts import *   # import all mean/std constants

import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import os
import glob

from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg
from torch.utils.data.dataset import Dataset
from torchvision import datasets
import numpy as np


# Block ImageNet corrupt EXIF warnings
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def construct_datasets(dataset, data_path, normalize=True):
    """Construct datasets with appropriate transforms."""
    # Compute mean, std:
    if dataset == 'CIFAR100':
        trainset = CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
    elif dataset == 'CIFAR10':
        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    else:
        raise ValueError(f'Invalid dataset {dataset} given.')

    if normalize:
        print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        trainset.data_mean = data_mean
        trainset.data_std = data_std
    else:
        print('Normalization disabled.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        trainset.data_std = (1.0, 1.0, 1.0)

    # Setup data
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    trainset.transform = transform_train

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    if dataset == 'CIFAR100':
        validset = CIFAR100(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10':
        validset = CIFAR10(root=data_path, train=False, download=True, transform=transform_valid)

    if normalize:
        validset.data_mean = data_mean
        validset.data_std = data_std
    else:
        validset.data_mean = (0.0, 0.0, 0.0)
        validset.data_std = (1.0, 1.0, 1.0)


    return trainset, validset


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class CIFAR100(torchvision.datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index