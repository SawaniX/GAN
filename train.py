import torch
import os
from models import Trainer, config
from models.config import get_transform_input, get_transform_target
from torch.utils.data import DataLoader
from dataset import CustomDataset

torch.backends.cudnn.benchmark = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    DATASET_PATH = 'dataset/datasets/ecommerce/data/shoes_men/processed/'
    SAVE_PATH = 'dataset/datasets/ecommerce/data/shoes_men/processed/'
    
    train_dataset = CustomDataset(labels_path=os.path.join(DATASET_PATH, 'train/labels/'),
                            samples_path=os.path.join(DATASET_PATH, 'train/sketches/'),
                            transform=get_transform_input(),
                            target_transform=get_transform_target())
    train_dataloader = DataLoader(train_dataset,
                              batch_size=config.get('BATCH_SIZE'),
                              shuffle=True)
    
    test_dataset = CustomDataset(labels_path=os.path.join(DATASET_PATH, 'test/labels/'),
                            samples_path=os.path.join(DATASET_PATH, 'test/sketches/'),
                            transform=get_transform_target(),
                            target_transform=get_transform_target())
    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.get('BATCH_SIZE'),
                                shuffle=False)
    

    neural_networks = Trainer(device, SAVE_PATH)

    neural_networks.train(train_dataloader, test_dataloader)

    
if __name__=='__main__':
    main()
