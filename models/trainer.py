import torch
from models import Discriminator, Generator, config
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torchvision.utils import save_image


class Trainer:
    SAVE_PATH: str = 'models/checkpoints/checkpoint.path.tar'

    def __init__(self, device: str, in_channels: int = 3) -> None:
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

        print('============= TRAINING STARTED =============')
        for epoch in range(config.get('NUM_EPOCHS')):
            loop = tqdm(train_dataloader, leave=True)

            for x, y in loop:
                x, y = x.to(self.device), y.to(self.device)

                y_fake = self.generator(x)
                D_real = self.discriminator(x, y)
                D_fake = self.discriminator(x, y_fake.detach())
                D_real_loss = BCE(D_real, torch.ones_like(D_real))
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = D_real_loss + D_fake_loss

                self.discriminator.zero_grad()
                D_loss.backward()
                self.discriminator_optimizer.step()

                D_fake = self.discriminator(x, y_fake)
                G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                l1 = L1(y_fake, y) * config.get('LAMBDA')
                G_loss = G_fake_loss + l1

                self.generator_optimizer.zero_grad()
                G_loss.backward()
                self.generator_optimizer.step()

            if config.get('SAVE_MODEL') and epoch % 5 == 0:
                self._save_checkpoint()

            self._save_example(test_dataloader, epoch)

    def _save_example(self, test_dataloader: DataLoader, epoch: int) -> None:
        folder = 'evaluation'
        x, y = next(iter(test_dataloader))
        x, y = x.to(self.device), y.to(self.device)
        self.generator.eval()
        with torch.no_grad():
            y_fake = self.generator(x)
            y_fake = y_fake * 0.5 + 0.5
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
