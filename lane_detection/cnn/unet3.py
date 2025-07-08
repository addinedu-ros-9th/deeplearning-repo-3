import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# CBAM Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

# ASPP 모듈
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
        return self.output(torch.cat([block(x) for block in self.blocks], dim=1))

# 전체 네트워크 구조
class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64
        self.enc2 = resnet.layer1  # 64
        self.enc3 = resnet.layer2  # 128
        self.enc4 = resnet.layer3  # 256
        self.enc5 = resnet.layer4  # 512

        self.cbam2 = CBAM(64)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(256)
        self.cbam5 = CBAM(512)

        self.aspp = ASPP(512, 256)

        # Decoder (DeepLabV3+ 스타일)
        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.ReLU())

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        # Auxiliary classifier
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.cbam2(self.enc2(x1))
        x3 = self.cbam3(self.enc3(x2))
        x4 = self.cbam4(self.enc4(x3))
        x5 = self.cbam5(self.enc5(x4))

        x = self.aspp(x5)

        # Auxiliary classifier output (classification-level loss 지원 시 사용)
        aux_out = self.aux_classifier(x)

        x = self.up4(x)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x1_resized = F.interpolate(x1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, x1_resized], dim=1))

        seg_out = self.final(x)
        return seg_out, aux_out
