import torch
from models import Trainer, config
from models.config import get_transform_input, get_transform_target
from torch.utils.data import DataLoader
from dataset import CustomDataset

torch.backends.cudnn.benchmark = True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    TRAIN_LABELS_PATH = 'dataset/datasets/clothes/processed/train/labels/'
    TRAIN_SAMPLES_PATH = 'dataset/datasets/clothes/processed/train/sketches/'
    TEST_LABELS_PATH = 'dataset/datasets/clothes/processed/test/labels/'
    TEST_SAMPLES_PATH = 'dataset/datasets/clothes/processed/test/sketches/'

    
    train_dataset = CustomDataset(labels_path=TRAIN_LABELS_PATH,
                            samples_path=TRAIN_SAMPLES_PATH,
                            transform=get_transform_input(),
                            target_transform=get_transform_target())
    train_dataloader = DataLoader(train_dataset,
                              batch_size=config.get('BATCH_SIZE'),
                              shuffle=True)
    
    test_dataset = CustomDataset(labels_path=TEST_LABELS_PATH,
                            samples_path=TEST_SAMPLES_PATH,
                            transform=get_transform_target(),
                            target_transform=get_transform_target())
    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.get('BATCH_SIZE'),
                                shuffle=False)
    

    neural_networks = Trainer(device)

    neural_networks.train(train_dataloader, test_dataloader)

    
if __name__=='__main__':
    main()
