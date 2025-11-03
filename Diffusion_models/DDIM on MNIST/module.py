import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import SinusoidalPosEmb


def timestep_embedding(timesteps, dim):
    """
    Sinusoidal embedding, timesteps is (B,) integers.
    Returns (B, dim)
    """
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0,1))
    return emb



class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        else:
            self.time_mlp = None

    def forward(self, x, t_emb=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        if self.time_mlp is not None:
            assert t_emb is not None
            # add time embedding as bias (FiLM-style)
            temb = self.time_mlp(t_emb).view(x.shape[0], -1, 1, 1)
            h = h + temb
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

# ---------- attention block ----------
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = x.view(b, c, h*w)  # (B, C, N)
        q = self.q(x)  # (B, C, N)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(torch.einsum("bcn,bcm->bnm", q, k) / math.sqrt(c), dim=-1)  # (B,N,N)
        out = torch.einsum("bnm,bcm->bcn", attn, v)
        out = self.proj(out)
        out = out.view(b, c, h, w)
        return out + x_in

# ---------- UNet ----------
class BigUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_ch=128,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        time_emb_dim=512,
        attn_resolutions=(16, 8),
        groups=8,
        dropout=0.1,
        debug=False,
    ):
        super().__init__()
        self.debug = debug

        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # initial conv
        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # channel schedule per resolution
        chs = [base_ch * m for m in ch_mult]  # e.g. [128, 256, 256, 512]

        # ========= Down path =========
        self.down_blocks = nn.ModuleList()
        self.down_samp   = nn.ModuleList()

        curr_ch = base_ch
        for i, out_ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(curr_ch, out_ch, time_emb_dim=time_emb_dim, groups=groups, dropout=dropout))
                curr_ch = out_ch
            self.down_blocks.append(blocks)

            # downsample except on last level
            if i < len(chs) - 1:
                self.down_samp.append(nn.Conv2d(curr_ch, curr_ch, 3, stride=2, padding=1))
            else:
                self.down_samp.append(nn.Identity())

        # ========= Mid =========
        self.mid_block1 = ResidualBlock(curr_ch, curr_ch, time_emb_dim=time_emb_dim, groups=groups, dropout=dropout)
        self.mid_attn   = AttentionBlock(curr_ch)
        self.mid_block2 = ResidualBlock(curr_ch, curr_ch, time_emb_dim=time_emb_dim, groups=groups, dropout=dropout)

        # ========= Up path =========
        self.up_blocks = nn.ModuleList()
        self.up_samp   = nn.ModuleList()

        # We’ll mirror the down path in reverse
        # Important: keep a separate "build-time" channel tracker for the expected input to the first block.
        build_curr_ch = curr_ch  # starts at bottleneck channels

        # reversed list of (level_idx, out_ch at that level)
        for i, out_ch in reversed(list(enumerate(chs))):
            blocks = nn.ModuleList()

            # First block at this level consumes concatenation: (build_curr_ch + out_ch) -> out_ch
            blocks.append(ResidualBlock(build_curr_ch + out_ch, out_ch, time_emb_dim=time_emb_dim, groups=groups, dropout=dropout))

            # Remaining blocks are out_ch -> out_ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim=time_emb_dim, groups=groups, dropout=dropout))

            self.up_blocks.append(blocks)

            # After finishing this level, channels are out_ch
            build_curr_ch = out_ch

            # Upsample on all but the final (topmost) level
            if i > 0:
                self.up_samp.append(nn.ConvTranspose2d(build_curr_ch, build_curr_ch, 4, stride=2, padding=1))
            else:
                self.up_samp.append(nn.Identity())

        # output head
        self.out_norm = nn.GroupNorm(groups, build_curr_ch)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(build_curr_ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        # time embedding
        temb = timestep_embedding(t, self.time_mlp[0].in_features)
        temb = self.time_mlp(temb)

        hs = []
        h = self.init_conv(x)

        # ===== Down =====
        for blocks, down in zip(self.down_blocks, self.down_samp):
            for block in blocks:
                h = block(h, temb)
                if self.debug:
                    print(f"down block -> {tuple(h.shape)}")
            hs.append(h)   # one skip per resolution
            h = down(h)
            if self.debug:
                print(f"down samp -> {tuple(h.shape)}")

        # ===== Mid =====
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        if self.debug:
            print(f"mid -> {tuple(h.shape)}")

        # ===== Up =====
        for blocks, up in zip(self.up_blocks, self.up_samp):
            skip = hs.pop()

            # align spatial sizes (prefer upsampling the skip to decoder size)
            if h.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=h.shape[2:], mode="nearest")

            # concat once per level
            h = torch.cat([h, skip], dim=1)
            if self.debug:
                print(f"concat -> {tuple(h.shape)} (into first up block expecting {blocks[0].norm1.num_channels})")

            # first block: (concat) -> out_ch
            h = blocks[0](h, temb)

            # remaining blocks: out_ch -> out_ch
            for block in blocks[1:]:
                h = block(h, temb)
                if self.debug:
                    print(f"up block -> {tuple(h.shape)}")

            # go to next (larger) resolution
            h = up(h)
            if self.debug:
                print(f"up samp -> {tuple(h.shape)}")

        # ===== Out =====
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out_conv(h)
