import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    def __init__(self, channels, embedding_dim):
        super().__init__()
        self.gamma_layer = nn.Linear(embedding_dim, channels)
        self.beta_layer = nn.Linear(embedding_dim, channels)

    def forward(self, x, lang_embedding):
        gamma = self.gamma_layer(lang_embedding).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_layer(lang_embedding).unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.film_layer = FiLMLayer(out_channels, embedding_dim)

    def forward(self, x1, x2, lang_embedding):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1,[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.film_layer(x, lang_embedding)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class LanguageConditionedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, embedding_dim, bilinear=True):
        super(LanguageConditionedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, embedding_dim, bilinear)
        self.up2 = Up(512, 256 // factor, embedding_dim, bilinear)
        self.up3 = Up(256, 128 // factor, embedding_dim, bilinear)
        self.up4 = Up(128, 64, embedding_dim, bilinear)
        
        self.outc = OutConv(64, n_classes)
        # --- FIX: We now output a dense spatial vector field instead of a global pool! ---
        self.vec_out = OutConv(64, 2) 

    def forward(self, image, lang_embedding):
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4, lang_embedding)
        x = self.up2(x, x3, lang_embedding)
        x = self.up3(x, x2, lang_embedding)
        x = self.up4(x, x1, lang_embedding)
        
        logits = self.outc(x)
        vec_field = self.vec_out(x)
        
        return logits, vec_field