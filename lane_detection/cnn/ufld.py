import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.atrous_block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.atrous_block4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)  # 드롭아웃 추가
        )
    def forward(self, x):
        size = x.shape[2:]
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block2(x)
        x3 = self.atrous_block3(x)
        x4 = self.atrous_block4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x_cat)
class UFLDLikeModel(nn.Module):
    def __init__(self, num_lanes=4, num_points=18):
        super(UFLDLikeModel, self).__init__()
        self.num_lanes = num_lanes
        self.num_points = num_points
        # Backbone: ResNet18
        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool and fc
        # ASPP-like module
        self.aspp = ASPP(512, 64)

        # Regression heads for X and Y coordinates
        self.classifier_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_lanes * num_points)
        )
        self.classifier_y = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_lanes * num_points)
        )
        # Existence prediction
        self.exist_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_lanes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        x = self.aspp(features)
        pred_x = self.classifier_x(x).view(-1, self.num_lanes, self.num_points)
        pred_y = self.classifier_y(x).view(-1, self.num_lanes, self.num_points)
        pred_exist = self.exist_pred(x)
        return pred_x, pred_y, pred_exist