import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2, 2) # kernel 2 stride 2
        self.nonlin = nn.ELU()

    def forward(self, x):
        #x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.nonlin(x)
        #x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, padding=1, stride=1)
        self.norm2 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.norm3 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.norm4 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        self.nonlin = nn.ELU()

    def forward(self, x):

        #x = self.norm1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlin(x)
        x = self.conv2(x)
        x = self.norm3(x)
        x = self.nonlin(x)
        x = self.conv3(x)
        x = self.norm4(x)
        x = self.nonlin(x)
        x = self.conv4(x)
        x = F.sigmoid(x)
        return x

