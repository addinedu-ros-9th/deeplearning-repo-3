import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# stress channel weight (reduce and expand)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

# (expand and reduce)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18),
        ])
        self.output = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        out = [block(x) for block in self.blocks]
        return self.output(torch.cat(out, dim=1))

class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4

        self.se5 = SEBlock(512)
        self.se4 = SEBlock(256)
        self.se3 = SEBlock(128)
        self.se2 = SEBlock(64)

        self.aspp = ASPP(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.3)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.3)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.3)
        )

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(), nn.Dropout2d(0.3)
        )

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x5 = self.se5(x5)
        x = self.aspp(x5)

        x = self.up4(x)
        x = self.dec4(torch.cat([x, self.se4(x4)], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, self.se3(x3)], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, self.se2(x2)], dim=1))

        x = self.up1(x)
        x1_resized = F.interpolate(x1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, x1_resized], dim=1))

        return self.final(x)
