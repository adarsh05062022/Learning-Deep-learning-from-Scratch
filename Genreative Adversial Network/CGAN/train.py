import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import Critic, Generator, initialize_weight,gradient_penalty
from dataset_util import CelebADataset
from util import save_generated_images,save_checkpoint,load_latest_checkpoint

# Hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
IMAGE_SIZE = 64
CHANNEL_IMG = 1
NUM_CLASSES = 10
GEN_EMBEDDING = 100
Z_DIM = 100
NUM_EPOCHS = 15
FEATURE_CRITIC = 64
FEATURE_GEN = 64
CRITIC_ITERATION = 5
LAMBDA_GP  = 10

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.CenterCrop(64), 
        transforms.Normalize(
            [0.5 for _ in range(CHANNEL_IMG)],[0.5 for _ in range(CHANNEL_IMG)]
        ),
    ]
)
image_dir = "/data1/adarsh/MODEL_TRAINING/GAN/CGAN/generated_images"
checkpoint_dir = "/data1/adarsh/MODEL_TRAINING/GAN/CGAN/checkpoints"

# dataset_path = '/data1/adarsh/CELEB_FACE_DATASET/img_align_celeba'


# dataset = CelebADataset(root_dir=dataset_path,transform=transforms)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)


dataloader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=2)

# Initialization

gen = Generator(Z_DIM,CHANNEL_IMG,FEATURE_GEN,NUM_CLASSES,IMAGE_SIZE,GEN_EMBEDDING).to(device)
critic = Critic(CHANNEL_IMG,FEATURE_CRITIC,NUM_CLASSES,IMAGE_SIZE).to(device)

initialize_weight(gen)
initialize_weight(critic)

# initialization of optimiser

opt_gen = optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))



def train():
    start_epoch = load_latest_checkpoint(gen, critic, opt_gen, opt_critic,checkpoint_dir)
    gen.train()
    critic.train()

    for epoch in range(start_epoch,NUM_EPOCHS):
        for batch_index , (real,labels) in enumerate(dataloader):
            real = real.to(device)
            labels = labels.to(device)
            curr_batch_size = real.shape[0]

            # Train Critic = max E[critic(real)] - E[critic(fake)]
            for _ in range(CRITIC_ITERATION):
                noise = torch.randn(curr_batch_size,Z_DIM,1,1).to(device)

                fake = gen(noise,labels)
                critic_real = critic(real,labels).reshape(-1)
                critic_fake = critic(fake,labels).reshape(-1)
                gp = gradient_penalty(critic,labels,real,fake,device)
                loss_critic = -(torch.mean(critic_real)-torch.mean(critic_fake)) + LAMBDA_GP * gp
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()


            # train gen max E[critic(gen_fake)] or min -E[critic(gen_fake)] 
            gen_fake = critic(fake,labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            if batch_index % 100 ==0:
                print(f"[Epochs {epoch+1}/{NUM_EPOCHS}] [Batch {batch_index}/{len(dataloader)}] [D_loss : {loss_critic.item()}] [G_loss : {loss_gen.item()}]")
            
        save_generated_images(gen,labels, epoch,device,image_dir)
        save_checkpoint(gen,critic,opt_gen,opt_critic,epoch,checkpoint_dir)

train()







