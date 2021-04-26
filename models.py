import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# AUTOENCODER MODELS

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.module = nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.module(x)

class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2])

    def forward(self, x):
        x = self.backbone(x)
        return x

class ResnetDecoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.upsample = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
        )
        self.project = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.project(x)
        x = F.sigmoid(x)
        return x

# DQN MODELS

class DQNBase(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        resnet18 = torchvision.models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet18.children())[:-2]) 
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.Conv2d(128, 32, 1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, num_actions),
        )

    def forward(self, x):
        x = self.backbone(x) # N,512,8,5
        x = self.reduce_channels(x) # N,32,8,5
        x = x.flatten(1,-1) # N,32*8*5 = N,1280
        x = self.regressor(x) # N,num_actions
        return x

class DQNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        pass

    def forward(self, x):
        pass










