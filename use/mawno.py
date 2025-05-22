import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.fft
from pytorch_wavelets import DWT1D, IDWT1D
from einops import rearrange
import time

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        dropout = 0.5  
        dim_head = dim // heads
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, spatial_size=None):
        #start_time = time.time()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        return self.to_out(out)

class WaveConv1d(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(WaveConv1d, self).__init__()
        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = config.level

        self.dwt_ = DWT1D(wave='coif2', J=self.level, mode=config.mode).to(config.device)
        dummy_input = torch.randn(1, in_channels, 256).to(config.device)
        self.mode_data, self.coe_data = self.dwt_(dummy_input)
        self.modes1 = self.mode_data.shape[-1]
        self.dwt1d = DWT1D(wave='coif2', J=self.level, mode=config.mode).to(config.device)
        self.idwt1d = IDWT1D(wave='coif2', mode=config.mode).to(config.device)
        self.sa_c = SelfAttention(dim=self.out_channels, heads=8).to(config.device)

        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1).to(config.device))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1).to(config.device))

    def mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x = x.to(self.config.device)

        x_ft, x_coeff = self.dwt1d(x)


        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1]).to(self.config.device)
        out_ft = self.mul1d(x_ft, self.weights1)

        x_coeff[-1] = x_coeff[-1].permute(0, 2, 1)
        x_coeff[-1] = self.sa_c(x_coeff[-1])
        x_coeff[-1] = F.gelu(x_coeff[-1])
        x_coeff[-1] = x_coeff[-1].permute(0, 2, 1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)

        x = self.idwt1d((out_ft, x_coeff))
        return x
    
class Block(nn.Module):
    def __init__(self, config, dim):
        super(Block, self).__init__()
        self.config = config
        self.filter = WaveConv1d(config, dim, dim).to(config.device)
        self.conv = nn.Conv1d(dim, dim, 3, padding=1).to(config.device)

    def forward(self, x):
        x = x.to(self.config.device)
        x1 = self.filter(x)
        x2 = self.conv(x)
        x = x1 + x2

        x = F.gelu(x)
        return x

class MAWNONet(nn.Module):
    def __init__(self, config):
        super(MAWNONet, self).__init__()

        self.config = config
        self.setup_seed(config.model_parameters['seed'])
        self.config.prob_dim = config.model_parameters['prob_dim']
        self.config.fc_map_dim = config.model_parameters['fc_map_dim']

        self.input_channels = 256
        self.embed_dim = config.embed_dim
        self.ln = nn.LayerNorm(256)
        self.conv_layer = nn.Conv1d(self.input_channels, self.embed_dim, kernel_size=3, padding=1,dilation=1).to(config.device)

        self.blocks = nn.ModuleList([
            Block(config, self.embed_dim).to(config.device)
            for _ in range(config.depth)])

        self.fc1 = nn.Linear(self.embed_dim, config.fc_map_dim).to(config.device)
        self.fc1_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(config.fc_map_dim, 256).to(config.device)

    @staticmethod
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def forward(self, x):

        x = x.to(self.config.device)
        x = self.conv_layer(x)
        x = self.ln(x)

        x = x.permute(0, 2, 1)
        for i, blk in enumerate(self.blocks):
            # print(f"Before block {i}, x shape: {x.shape}")
            x = blk(x)
            # print(f"After block {i}, x shape: {x.shape}")
        x = x.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x