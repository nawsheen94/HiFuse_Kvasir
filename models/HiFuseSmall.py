import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class GlobalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class HFFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFFBlock, self).__init__()
        self.local_block = LocalBlock(in_channels, out_channels)
        self.global_block = GlobalBlock(in_channels, out_channels)

    def forward(self, x):
        local_feat = self.local_block(x)
        global_feat = self.global_block(x)
        return local_feat + global_feat

class HiFuseSmall(nn.Module):
    def __init__(self, num_classes):
        super(HiFuseSmall, self).__init__()
        self.hff1 = HFFBlock(3, 64)
        self.hff2 = HFFBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.hff1(x))
        x = self.pool(self.hff2(x))
        x = x.view(-1, 128 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
