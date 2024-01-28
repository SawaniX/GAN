from PIL import Image
from torch import nn, cat, randn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act='relu', use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect') if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        self.down1 = Block(features, features*2, down=True, act='leaky', use_dropout=False)
        self.down2 = Block(features*2, features*4, down=True, act='leaky', use_dropout=False)
        self.down3 = Block(features*4, features*8, down=True, act='leaky', use_dropout=False)
        self.down4 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        self.down5 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        self.down6 = Block(features*8, features*8, down=True, act='leaky', use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode='reflect'),
            nn.ReLU()
        )
        self.up1 = Block(features*8, features*8, down=False, act='relu', use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False, act='relu', use_dropout=False)
        self.up5 = Block(features*8*2, features*4, down=False, act='relu', use_dropout=False)
        self.up6 = Block(features*4*2, features*2, down=False, act='relu', use_dropout=False)
        self.up7 = Block(features*2*2, features, down=False, act='relu', use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(cat([up1, d7], 1))
        up3 = self.up3(cat([up2, d6], 1))
        up4 = self.up4(cat([up3, d5], 1))
        up5 = self.up5(cat([up4, d4], 1))
        up6 = self.up6(cat([up5, d3], 1))
        up7 = self.up7(cat([up6, d2], 1))
        return self.final_up(cat([up7, d1], 1))
    
model = Generator()

import torch
checkpoint = torch.load('checkpoint.path.tar')

model.load_state_dict(checkpoint['generator_state_dict'])
model.eval()

import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
transform_label = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image_filenames = [x for x in os.listdir('maps/val')]

for img_name in image_filenames:
    image = np.array(Image.open(f'maps/val/{img_name}'))
    img = image[:, :600, :]
    label = image[:, 600:, :]
    label = transform_label(label)
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    if torch.cuda.is_available():
        model.cuda()
    out = model(img)
    out.detach().squeeze(0).cpu()

    from torchvision.utils import save_image
    save_image(label,  f"wyniki/label{img_name}.png")
    save_image(out,  f"wyniki/wynik{img_name}.png")