import torch
import torchvision
from models import Discriminator, Generator, config
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn, Tensor
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataset import CustomDataset

torch.backends.cudnn.benchmark = True

class InitNeuralNetworks:
    def __init__(self, device: str, in_channels: int = 3) -> None:
        self.SAVE_PATH: str = 'models/checkpoints/checkpoint.path.tar'
        self.device: str = device
        self.generator: Generator = Generator(in_channels).to(device)
        self.discriminator: Discriminator = Discriminator(in_channels).to(device)
        self.generator_optimizer: Adam = Adam(self.generator.parameters(), 
                                              lr=config.get('LEARNING_RATE'), 
                                              betas=config.get('BETAS'))
        self.discriminator_optimizer: Adam = Adam(self.discriminator.parameters(), 
                                                  lr=config.get('LEARNING_RATE'), 
                                                  betas=config.get('BETAS'))
        
    def train(self, train_dataloader: DataLoader, test_dataloader: DataLoader):
        BCE = nn.BCEWithLogitsLoss()
        L1 = nn.L1Loss()
        g_scaler = torch.cuda.amp.grad_scaler.GradScaler()
        d_scaler = torch.cuda.amp.grad_scaler.GradScaler()

        x, y = next(iter(test_dataloader))
        x, y = x.to(self.device), y.to(self.device)

        for epoch in range(config.get('NUM_EPOCHS')):
            loop = tqdm(train_dataloader, leave=True)

            for idx, (x, y) in enumerate(loop):
                x, y = x.to(self.device), y.to(self.device)

                with torch.cuda.amp.autocast_mode.autocast():
                    y_fake = self.generator(x)
                    D_real = self.discriminator(x, y)
                    D_fake = self.discriminator(x, y_fake.detach())
                    D_real_loss = BCE(D_real, torch.ones_like(D_real))
                    D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                    D_loss = (D_real_loss + D_fake_loss) / 2

                self.discriminator.zero_grad()
                d_scaler.scale(D_loss).backward()
                d_scaler.step(self.discriminator_optimizer)
                d_scaler.update()

                with torch.cuda.amp.autocast_mode.autocast():
                    D_fake = self.discriminator(x, y_fake)
                    G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                    l1 = L1(y_fake, y) * config.get('LAMBDA')
                    G_loss = G_fake_loss + l1

                self.generator_optimizer.zero_grad()
                g_scaler.scale(G_loss).backward()
                g_scaler.step(self.generator_optimizer)
                g_scaler.update()

                if idx % 10 == 0:
                    loop.set_postfix(
                        D_real=torch.sigmoid(D_real).mean().item(),
                        D_fake=torch.sigmoid(D_fake).mean().item(),
                    )

            if config.get('SAVE_MODEL') and epoch % 5 == 0:
                self._save_checkpoint()

            self._save_example(x, y, epoch)

    def _save_example(self, x: Tensor, y: Tensor, epoch: int) -> None:
        folder = 'evaluation'
        self.generator.eval()
        with torch.no_grad():
            y_fake = self.generator(x)
            save_image(y_fake, folder + f"/y_gen_{epoch}.png")
            save_image(x, folder + f"/input_{epoch}.png")
            if epoch == 1:
                save_image(y, folder + f"/label_{epoch}.png")
        self.generator.train()
        
    def _save_checkpoint(self) -> None:
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            }, self.SAVE_PATH)
        
    def load_checkpoint(self, eval: bool = True, lr: float = None) -> None:
        checkpoint = torch.load(self.SAVE_PATH)
        self.generator.load_state_dict(checkpoint.get('generator_state_dict'))
        self.discriminator.load_state_dict(checkpoint.get('generator_optimizer_state_dict'))
        self.generator_optimizer.load_state_dict(checkpoint.get('generator_optimizer_state_dict'))
        self.discriminator_optimizer.load_state_dict(checkpoint.get('discriminator_optimizer_state_dict'))
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        if lr:
            for gen_param_group, disc_param_group in zip(self.generator_optimizer.param_groups, self.discriminator_optimizer.param_groups):
                gen_param_group['lr'] = lr
                disc_param_group['lr'] = lr
        if eval:
            self.generator.eval()
            self.discriminator.eval()
        else:
            self.generator.train()
            self.discriminator.train()



def plot_samples(train_dataloader):
    writer = SummaryWriter()
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    img = torch.cat((images, labels))
    img_grid = torchvision.utils.make_grid(img)
    writer.add_image('dataset_samples', img_grid)
    writer.flush()
    writer.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    LABELS_PATH = 'dataset/archive/256x256/photo/tx_000000000000/airplane/' #os.environ.get('LABELS_PATH')
    SAMPLES_PATH = 'dataset/archive/256x256/sketch/tx_000000000000/airplane/' #os.environ.get('SAMPLES_PATH')
    train_split, test_split = 0.7, 0.3

    transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.ToTensor()])
    dataset = CustomDataset(labels_path=LABELS_PATH,
                            samples_path=SAMPLES_PATH,
                            transform=transform,
                            target_transform=target_transform)
    train_dataset, test_dataset = random_split(dataset, [train_split, test_split])
    train_dataloader = DataLoader(train_dataset,
                              batch_size=config.get('BATCH_SIZE'),
                              shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.get('BATCH_SIZE'),
                                shuffle=False)

    neural_networks = InitNeuralNetworks(device)

    # plot_samples(train_dataloader)

    print('============= TRAINING STARTED =============')
    neural_networks.train(train_dataloader, test_dataloader)

    
if __name__=='__main__':
    main()
