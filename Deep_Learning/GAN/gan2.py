import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

"""https://www.kaggle.com/datasets/frabbisw/facial-age?resource=download"""

#-------------------------------DISCIMINATOR----------------------------------#

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=100,age_dim = 6):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, ), #32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  #16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),    #8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),    #4x4
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.fc = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Generator(nn.Module):
    def __init__(self, latent_dim=100, age_dim=6):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim+age_dim, 512 * 4 * 4)  #condition the input with the age dimention

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),   #8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),#16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),#32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),#64x64
            nn.Tanh()
        )

    def forward(self, z, age_label):

        z = z.view(z.size(0), -1)

        age_label = F.one_hot(age_label,num_classes = 6).float()

        x = torch.cat((z, age_label), dim=1)  # Concatenate latent vector with age label

        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        img = self.deconv(x)
        return img



class Discriminator(nn.Module):
    def __init__(self, age_dim=10):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), #32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),#16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),#8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),#4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )

        self.fc_real = nn.Linear(512 * 4 * 4, 1)  # Real vs Fake classification
        self.fc_age = nn.Linear(512 * 4 * 4, age_dim)  # Age classification

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        real_or_fake = torch.sigmoid(self.fc_real(x))
        age_pred = self.fc_age(x)
        return real_or_fake, age_pred

#-------------NETWORK-------------#


#-------------DATASET-------------#


class CustomDataset(Dataset):
    def __init__(self,csv,transform = None):
        self.csv = pd.read_csv(csv)
        self.transform = transform
        
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):

        image = Image.open(self.csv.iloc[index].iloc[0])
        label = self.csv.iloc[index].iloc[1]
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label)
        
        return image,label


#-------------DATASET-------------#
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
IMG_SIZE = 64
IMG_CHANNELS = 3
Z_DIM = 100
EPOCHS = 3
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_CLASSES = 6
GEN_EMBEDDING = 100
TRAIN = True

class_map = {0:19,
             1:29,
             2:39,
             3:49,
             4:59,
             5:69}

transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
    ])    #normalize to [-1,1]

dataset = CustomDataset('image_labels2.csv',transform = transform)

loader = DataLoader(dataset,batch_size= BATCH_SIZE, shuffle=True)
gen = Generator()
disc = Discriminator()
encoder = Encoder()

optimizer = optim.Adam(gen.parameters(),lr = LEARNING_RATE,betas = (0.5,0.999))


recon_loss = nn.L1Loss()  # Used for Encoder-Generator phase
adv_loss = nn.BCELoss()  # Binary cross-entropy for real/fake classification

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_recon = SummaryWriter(f"logs/recons")


step = 0

gen.train()
disc.train()
encoder.train()
if TRAIN:
    for epoch in range(EPOCHS):
        for batch , (real,label) in enumerate(loader):

            z = encoder(real)

            recon = gen(z,label)
            
            loss = recon_loss(recon,real)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 40 == 0:
                print(
                    f"Epoch: [{epoch}/{EPOCHS}] Batch {batch}/{len(loader)}\
                        Loss : {loss:.4f}")
                with torch.no_grad():
                    z = encoder(real)
                    fake = gen(z,label)
                    
                    img_grid_real = torchvision.utils.make_grid(real[:32],normalize = True)
                    
                    img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize = True)
                    
                    
                    writer_real.add_image("Real",img_grid_real,global_step=step)
                    writer_recon.add_image("recons",img_grid_fake,global_step=step)
                    
                step += 1
    torch.save(disc.state_dict, 'disc.pth')    
    torch.save(gen.state_dict, 'gen.pth')   
else:
    gen.load_state_dict(torch.load('gen.pth',weight_only = False))    
    disc.load_state_dict(torch.load('disc.pth',weight_only = False))
    
    
            





















        