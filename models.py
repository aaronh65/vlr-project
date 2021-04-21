import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 56, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 56, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

