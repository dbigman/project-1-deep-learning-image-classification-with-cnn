# src/data_preprocessing.py

import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class DataPreprocessor:
    def __init__(self, dataset_name, data_dir, batch_size=64, validation_split=0.15, test_split=0.15, random_seed=42):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.transform = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self):
        if self.dataset_name.lower() == 'cifar10':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        elif self.dataset_name.lower() == 'animal':
            # Implement loading for the animal dataset
            # For example, using ImageFolder
            self.transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            dataset = datasets.ImageFolder(root=os.path.join(self.data_dir, 'animal_dataset'), transform=self.transform)
        else:
            raise ValueError("Unsupported dataset. Choose 'CIFAR10' or 'animal'.")

        return dataset

    def split_data(self, dataset):
        total_size = len(dataset)
        val_size = int(self.validation_split * total_size)
        test_size = int(self.test_split * total_size)
        train_size = total_size - val_size - test_size

        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )
        return train_set, val_set, test_set

    def create_data_loaders(self):
        dataset = self.load_data()
        train_set, val_set, test_set = self.split_data(dataset)

        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=4)

        return self.train_loader, self.val_loader, self.test_loader

    def get_data_loaders(self):
        if not self.train_loader:
            self.create_data_loaders()
        return self.train_loader, self.val_loader, self.test_loader
