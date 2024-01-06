import os
import re
from torchvision.io import read_image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, labels_path, samples_path, pattern=r'-\d+\.png'):
        self.labels_path = labels_path
        self.samples_path = samples_path
        self.samples_list = os.listdir(self.samples_path)
        self.pattern = pattern

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        sample_name = self.samples_list[idx]
        sample_path = os.path.join(self.samples_path, sample_name)
        sample = read_image(sample_path)

        label_name = ''.join(re.split(self.pattern, sample_name)) + '.jpg'
        label_path = os.path.join(self.labels_path, label_name)
        label = read_image(label_path)

        return sample, label
