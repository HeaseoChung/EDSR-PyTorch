import torch.nn as nn
from torch.cuda import amp

class ResBlock(nn.Module):
    def __init__(self, n_feats, res_scale=1.0):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, bias=True, padding=3//2))
            if i == 0:
                m.append(nn.ReLU(True))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res * self.res_scale

class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, num_feats=64, num_blocks=16, res_scale=1.0):
        super(EDSR, self).__init__()
        self.head = nn.Conv2d(num_channels, num_feats, kernel_size=3, padding=3//2)
        body = [ResBlock(num_feats, res_scale) for _ in range(num_blocks)]
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(
            nn.Conv2d(num_feats, num_feats * (scale_factor ** 2), kernel_size=3, stride=1, padding=1), 
            nn.PixelShuffle(scale_factor),
            nn.ReLU(True),
            nn.Conv2d(num_feats, num_channels, kernel_size=3, stride=1, padding=1), 
        )
    
    @amp.autocast()
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x