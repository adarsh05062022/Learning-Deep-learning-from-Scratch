
import torch



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get all image file paths from the directory
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply the transform if provided
        if self.transform:
            image = self.transform(image)

        return image

# defining thw ransformations

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
])

dataset_path = '/data1/adarsh/CELEB_FACE_DATASET/img_align_celeba'
dataset = CelebADataset(root_dir = dataset_path,transform = transform)

dataloader = DataLoader(dataset,batch_size = 128,shuffle = True,pin_memory=True,num_workers=2)

print(f"Total num of imag loaded:{len(dataset)}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGANGenerator(nn.Module):
    def __init__(self,z_dim = 100,img_channels = 3,feature_map = 64):
        super(DCGANGenerator,self).__init__()
        self.net = nn.Sequential(
            # input zdim x 1 X  1
            nn.ConvTranspose2d(z_dim,feature_map*8,kernel_size=4,stride=1,bias=False),
            nn.BatchNorm2d(feature_map*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map*8,feature_map*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(feature_map*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map*4,feature_map*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(feature_map*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map*2,feature_map,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map,img_channels,4,2,1,bias=False),
            nn.Tanh()

        )

    def forward(self,z):
        z = z.view(z.size(0),z.size(1),1,1)
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    def __init__(self,img_channels = 3,feature_map = 64):
        super(DCGANDiscriminator,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels,feature_map,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(feature_map,feature_map*2,4,2,1,bias=False),
            nn.BatchNorm2d(feature_map*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(feature_map*2,feature_map*4,4,2,1,bias=False),
            nn.BatchNorm2d(feature_map*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(feature_map*4,feature_map*8,4,2,1,bias=False),
            nn.BatchNorm2d(feature_map*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(feature_map*8,1,4,1,0,bias=False),
            nn.Sigmoid()



        )

    def forward(self,x):
        return self.net(x).view(-1,1)

# Loss function and optimiser

adversarial_loss = nn.BCELoss()
generator = DCGANGenerator(z_dim = 100)
discriminator = DCGANDiscriminator()


generator.apply(weights_init)
discriminator.apply(weights_init)

optimizer_G = optim.Adam(generator.parameters(),lr = 0.0002,betas = (0.5,0.999))
optimizer_D = optim.Adam(discriminator.parameters(),lr = 0.0002,betas = (0.5,0.999))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = generator.to(device)
discriminator = discriminator.to(device)

def save_generated_images(generator, epoch, device, num_images=16, image_dir="generated_images"):
    os.makedirs(image_dir, exist_ok=True)  # ensure folder exists

    # sample noise
    z = torch.randn(num_images, 100).to(device)
    generated_imgs = generator(z).detach().cpu()

    # save in local directory
    save_path = os.path.join(image_dir, f"epoch_{epoch+1}.png")
    torchvision.utils.save_image(
        generated_imgs,
        save_path,
        nrow=4, normalize=True
    )
    print(f"Saved generated images at: {save_path}")

    # show inline (optional during training)
    grid = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=True)
    plt.figure(figsize=(6, 6))   # better scaling
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(f"Epoch {epoch+1}")
    plt.axis('off')
    plt.show()

import os
import torch

# Directories
checkpoint_dir = "model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, max_keep=5):
    """Save model checkpoints and keep only the last 'max_keep' files."""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")

    # Save both Generator & Discriminator states
    torch.save({
        'epoch': epoch+1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, checkpoint_path)

    # Manage only last max_keep checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    # Sort numerically by epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove oldest if exceeding max_keep
    while len(checkpoints) > max_keep:
        os.remove(os.path.join(checkpoint_dir, checkpoints[0]))
        checkpoints.pop(0)

# Resume training if checkpoint exists
def load_latest_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_dir="model_checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return 0  # start from epoch 0

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if len(checkpoints) == 0:
        return 0  # no checkpoints found

    # Sort numerically by epoch number
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    latest_checkpoint = checkpoints[-1]  # now this is truly the latest epoch
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    return checkpoint['epoch']  # return the last trained epoch

def train(generator,discriminator,dataloader,epochs = 10):
    start_epoch = load_latest_checkpoint(generator, discriminator, optimizer_G, optimizer_D)
    for epoch in range(start_epoch,epochs):
        for i , imgs in enumerate(dataloader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            valid = torch.ones(batch_size,1).to(device)
            fake = torch.zeros(batch_size,1).to(device)

            # train discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs),valid)
            fake_loss = adversarial_loss(discriminator(generator(torch.randn(batch_size,100).to(device)).detach()),fake)
            d_loss = (real_loss + fake_loss)/2
            d_loss.backward()
            optimizer_D.step()

            # Training the generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(generator(torch.randn(batch_size,100).to(device))),valid)
            g_loss.backward()
            optimizer_G.step()

            if i%100 ==0:
                print(f"[Epochs {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] [D_loss : {d_loss.item()}] [G_loss : {g_loss.item()}]")

        save_generated_images(generator, epoch, device)
        # At the end of epoch
        save_checkpoint(generator, discriminator,optimizer_G,optimizer_D, epoch, max_keep=2)

# Start training
train(generator, discriminator, dataloader, epochs=15)