import torch
from dataset import LoadData
from models import Discriminator, Generator, config
from torchvision.transforms import ToTensor
from torch import nn, optim
from tqdm import tqdm
from urils import save_checkpoint, load_checkpoint, save_some_examples
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    discriminator = Discriminator(in_channels=3).to(device)
    generator = Generator(in_channels=3).to(device)
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.get('LEARNING_RATE'), betas=config.get('BETAS'))
    opt_gen = optim.Adam(generator.parameters(), lr=config.get('LEARNING_RATE'), betas=config.get('BETAS'))
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()

    train_dataloader, test_dataloader = LoadData(transform=ToTensor(),
                                                target_transform=ToTensor()
                                                ).get_dataloader(train_split=0.7,
                                                                test_split=0.3,
                                                                batch_size=1)
    g_scaler = torch.cuda.amp.grad_scaler.GradScaler()
    d_scaler = torch.cuda.amp.grad_scaler.GradScaler()

    print('============= TRAINING STARTED =============')
    for epoch in range(config.get('NUM_EPOCHS')):
        loop = tqdm(train_dataloader, leave=True)

        for idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast_mode.autocast(device_type=device):
                y_fake = generator(x)
                D_real = discriminator(x, y)
                D_fake = discriminator(x, y_fake.detach())
                D_real_loss = BCE(D_real, torch.ones_like(D_real))
                D_fake_loss = BCE(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

            discriminator.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            with torch.cuda.amp.autocast_mode.autocast(device_type=device):
                D_fake = discriminator(x, y_fake)
                G_fake_loss = BCE(D_fake, torch.ones_like(D_fake))
                l1 = L1(y_fake, y) * config.get('LAMBDA')
                G_loss = G_fake_loss + l1

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 10 == 0:
                loop.set_postfix(
                    D_real=torch.sigmoid(D_real).mean().item(),
                    D_fake=torch.sigmoid(D_fake).mean().item(),
                )

        if config.get('SAVE_MODEL') and epoch % 5 == 0:
            save_checkpoint(generator, opt_gen, filename='gen.pth.tar')
            save_checkpoint(discriminator, opt_disc, filename='disc.pth.tar')

        save_some_examples(generator, test_dataloader, epoch, folder="evaluation", device=device)


    # _, axs = plt.subplots(1, 2)
    # for X, y in train_dataloader:
    #     axs[0].imshow(torch.permute(X[0], (1, 2, 0)))
    #     axs[1].imshow(torch.permute(y[0], (1, 2, 0)))
    #     plt.show()

    
if __name__=='__main__':
    main()
