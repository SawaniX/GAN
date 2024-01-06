import os
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset

import matplotlib.pyplot as plt
import time


labels_path = 'dataset/archive/256x256/photo/tx_000000000000/airplane/' #os.environ.get('LABELS_PATH')
samples_path = 'dataset/archive/256x256/sketch/tx_000000000000/airplane/' #os.environ.get('SAMPLES_PATH')
dataset = CustomDataset(labels_path=labels_path,
                        samples_path=samples_path)

train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
print(len(train_dataset))

batch_size = 64
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size)
test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size)



# _, axs = plt.subplots(1, 2)
# for X, y in train_dataloader:
#     axs[0].imshow(X[0].T)
#     axs[1].imshow(y[0].T)
#     plt.show()
