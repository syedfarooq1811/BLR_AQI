import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialUNet(nn.Module):
    """
    Lightweight Spatial U-Net for mapping 12 station inputs to 100x100m grid.
    Input shape: (B, C, H, W)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # Shape: (B, in_channels, H, W)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        d1 = self.up(e2)
        
        # padding if shapes differ due to maxpool
        diffY = e1.size()[2] - d1.size()[2]
        diffX = e1.size()[3] - d1.size()[3]
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        out = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Global Residual Connection: Ensure street-level super-resolution preserves station-level RMSE accuracy
        out = out + x
        # Shape: (B, out_channels, H, W)
        return out
