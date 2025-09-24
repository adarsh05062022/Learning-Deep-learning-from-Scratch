import torch
from utils import save_generated_images,save_checkpoint,load_latest_checkpoint
import torch.nn as nn
import torch.optim as optim
import config
from my_dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm


checkpoint_dir = "/data1/adarsh/MODEL_TRAINING/GAN/PIX2PIX/checkpoints"
generated_img_dir = "/data1/adarsh/MODEL_TRAINING/GAN/PIX2PIX/generated_image"
dataset_dir = "/data1/adarsh/maps"


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce,
):
    loop = tqdm(loader, leave=False)
    total_D_loss, total_G_loss = 0.0, 0.0

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # ---- Train Discriminator ----
        y_fake = gen(x)
        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # ---- Train Generator ----
        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        # accumulate losses
        total_D_loss += D_loss.item()
        total_G_loss += G_loss.item()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    avg_D_loss = total_D_loss / len(loader)
    avg_G_loss = total_G_loss / len(loader)
    return avg_D_loss, avg_G_loss


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = 1e-5,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    START_EPOCH =0
    if config.LOAD_MODEL:
        START_EPOCH = load_latest_checkpoint(gen,disc,opt_gen,opt_disc,checkpoint_dir)
    
    train_dataset = MapDataset(f"{dataset_dir}/train")
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)


    val_dataset = MapDataset(f"{dataset_dir}/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(START_EPOCH,config.NUM_EPOCHS):
        avg_D_loss, avg_G_loss = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE
        )

        
        print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] D_loss: {avg_D_loss:.4f} | G_loss: {avg_G_loss:.4f}")
            

        if epoch % 5 == 0:
            save_generated_images(gen,val_loader,epoch,config.DEVICE,generated_img_dir)
            save_checkpoint(gen,disc,opt_gen,opt_disc,epoch,checkpoint_dir)

main()