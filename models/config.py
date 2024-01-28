CONFIG = {
    'BATCH_SIZE': 1,
    'LEARNING_RATE': 0.0002,
    'BETAS': (0.5, 0.999),
    'NUM_EPOCHS': 1000,
    'SAVE_MODEL': True,
    'LAMBDA': 100,
    'RESIZE': (286, 286),
    'CROP_SIZE': (256, 256)
}

from torchvision.transforms import transforms

def get_transform_input():
    return transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(size=CONFIG.get('RESIZE'), antialias=True),
                            transforms.RandomCrop(size=CONFIG.get('CROP_SIZE')),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

def get_transform_target():
    return transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(size=CONFIG.get('CROP_SIZE'), antialias=True),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
