import torch
import torch.nn as nn




class Generator(nn.Module):
    def __init__(self, channel_noise,channel_img,feature_g):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            self._block(channel_noise,feature_g * 16,4,1,0),
            self._block(feature_g * 16,feature_g * 8,4,2,1),
            self._block(feature_g * 8,feature_g * 4,4,2,1),
            self._block(feature_g * 4,feature_g * 2,4,2,1),
            nn.ConvTranspose2d(feature_g *2 ,channel_img,kernel_size=4,stride=2,padding=1),

            nn.Tanh()
            
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,z):
        z = z.view(z.size(0),z.size(1),1,1)
        return self.model(z)


class Critic(nn.Module):
    def __init__(self,channel_img,feature_d):
        super(Critic,self).__init__()   
        self.model = nn.Sequential(
            nn.Conv2d(channel_img,feature_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self._block(feature_d,feature_d*2,4,2,1),
            self._block(feature_d*2,feature_d*4,4,2,1),
            self._block(feature_d*4,feature_d*8,4,2,1),

            nn.Conv2d(feature_d*8,1,kernel_size=4,stride=2,padding=0)

            # in WGAN no sigmoid

        )


    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self,x):
        return self.model(x).view(-1,1)
    

def initialize_weight(model):
    # Accoring to DCGAN PAPER
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)
    


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# # Test Critic
# critic = Critic(3,8)
# x = torch.randn(8, 3, 64, 64)
# out = critic(x)
# assert out.shape == (8, 1, 1, 1), f"Critic Test Failed, got {out.shape}"

# # Test Generator
# z = torch.randn(8, 100, 1, 1)
# gen = Generator(100, 3, 8)
# assert gen(z).shape == (8, 3, 64, 64), f"Generator Test Failed, got {gen(z).shape}"





