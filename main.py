import os
from torch.utils.data import DataLoader, random_split
from dataset import LoadData

import matplotlib.pyplot as plt


train_dataloader, test_dataloader = LoadData().get_dataloader(train_split=0.7,
                                                              test_split=0.3,
                                                              batch_size=64)



_, axs = plt.subplots(1, 2)
for X, y in train_dataloader:
    axs[0].imshow(X[0].T)
    axs[1].imshow(y[0].T)
    plt.show()
