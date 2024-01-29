import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, labels_path: str, samples_path: str, transform=None, target_transform=None):
        self.labels_path = labels_path
        self.samples_path = samples_path
        self.samples_list = os.listdir(self.samples_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        sample_name = self.samples_list[idx]
        sample_path = os.path.join(self.samples_path, sample_name)
        sample = np.array(Image.open(sample_path).convert('RGB'))

        label_path = os.path.join(self.labels_path, sample_name)
        label = np.array(Image.open(label_path).convert('RGB'))

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)

        return sample, label
