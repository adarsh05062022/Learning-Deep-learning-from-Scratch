import torch
import math
import torch.nn as nn
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: shape (B,) integer timesteps
        device = t.device
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=device) * (-math.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
