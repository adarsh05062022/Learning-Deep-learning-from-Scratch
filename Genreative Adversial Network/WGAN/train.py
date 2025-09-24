import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Critic, Generator, initialize_weight
from dataset_util import CelebADataset
from util import save_generated_images,save_checkpoint,load_latest_checkpoint

# Hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNEL_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 15
FEATURE_CRITIC = 64
FEATURE_GEN = 64
CRITIC_ITERATION = 5
WEIGHT_CLIP = 0.005


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNEL_IMG)],[0.5 for _ in range(CHANNEL_IMG)]
        ),
    ]
)
image_dir = "/data1/adarsh/MODEL_TRAINING/GAN/WGAN/generated_images"
checkpoint_dir = "/data1/adarsh/MODEL_TRAINING/GAN/WGAN/checkpoints"
dataset_path = '/data1/adarsh/CELEB_FACE_DATASET/img_align_celeba'
dataset = CelebADataset(root_dir=dataset_path,transform=transforms)


dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

# Initialization

gen = Generator(Z_DIM,CHANNEL_IMG,FEATURE_GEN).to(device)
critic = Critic(CHANNEL_IMG,FEATURE_CRITIC).to(device)

initialize_weight(gen)
initialize_weight(critic)

# initialization of optimiser

opt_gen = optim.RMSprop(gen.parameters(),lr=1e-5)
opt_critic = optim.RMSprop(critic.parameters(),lr=LEARNING_RATE)



def train():
    start_epoch = load_latest_checkpoint(gen, critic, opt_gen, opt_critic,checkpoint_dir)
    gen.train()
    critic.train()

    for epoch in range(start_epoch,NUM_EPOCHS):
        for batch_index , data in enumerate(dataloader):
            data = data.to(device)
            curr_batch_size = data.shape[0]

            # Train Critic = max E[critic(real)] - E[critic(fake)]
            for _ in range(CRITIC_ITERATION):
                noise = torch.randn(curr_batch_size,Z_DIM,1,1).to(device)

                fake = gen(noise)
                critic_real = critic(data).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                loss_critic = -(torch.mean(critic_real)-torch.mean(critic_fake))
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

                #clip critic weights between -0.01 to 0.01
                for p in critic.parameters():
                    p.data.clamp(-WEIGHT_CLIP,WEIGHT_CLIP)

            # train gen max E[critic(gen_fake)] or min -E[critic(gen_fake)] 
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            if batch_index % 100 ==0:
                print(f"[Epochs {epoch+1}/{NUM_EPOCHS}] [Batch {batch_index}/{len(dataloader)}] [D_loss : {loss_critic.item()}] [G_loss : {loss_gen.item()}]")
            
        save_generated_images(gen, epoch,device,image_dir)
        save_checkpoint(gen,critic,opt_gen,opt_critic,epoch,checkpoint_dir)

train()







