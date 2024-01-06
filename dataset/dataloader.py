from torch.utils.data import DataLoader, random_split
from .dataset import CustomDataset


class LoadData:
    def __init__(self):
        labels_path = 'dataset/archive/256x256/photo/tx_000000000000/airplane/' #os.environ.get('LABELS_PATH')
        samples_path = 'dataset/archive/256x256/sketch/tx_000000000000/airplane/' #os.environ.get('SAMPLES_PATH')
        self.dataset = CustomDataset(labels_path=labels_path,
                        samples_path=samples_path)

    def get_dataloader(self, train_split: float, test_split: float, batch_size: int) -> DataLoader:
        train_dataset, test_dataset = random_split(self.dataset, 
                                                   [train_split, test_split])

        train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=batch_size)
        return train_dataloader, test_dataloader
    