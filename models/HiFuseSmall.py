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
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class GlobalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

class HFFBlock(nn.Module):
    def __init__(self, local_channels, global_channels, out_channels):
        super(HFFBlock, self).__init__()
        self.local_conv = nn.Conv2d(local_channels, out_channels, kernel_size=1)
        self.global_conv = nn.Conv2d(global_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, local_features, global_features):
        local_out = self.local_conv(local_features)
        global_out = self.global_conv(global_features)
        combined = local_out + global_out
        return F.relu(self.bn(combined))

class HiFuseSmall(nn.Module):
    def __init__(self, num_classes):
        super(HiFuseSmall, self).__init__()
        self.local_block1 = LocalBlock(3, 32)
        self.global_block1 = GlobalBlock(3, 32)
        self.hff1 = HFFBlock(32, 32, 64)

        self.local_block2 = LocalBlock(64, 64)
        self.global_block2 = GlobalBlock(64, 64)
        self.hff2 = HFFBlock(64, 64, 128)

        self.local_block3 = LocalBlock(128, 128)
        self.global_block3 = GlobalBlock(128, 128)
        self.hff3 = HFFBlock(128, 128, 256)

        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        local_features1 = self.local_block1(x)
        global_features1 = self.global_block1(x)
        hff_out1 = self.hff1(local_features1, global_features1)

        local_features2 = self.local_block2(hff_out1)
        global_features2 = self.global_block2(hff_out1)
        hff_out2 = self.hff2(local_features2, global_features2)

        local_features3 = self.local_block3(hff_out2)
        global_features3 = self.global_block3(hff_out2)
        hff_out3 = self.hff3(local_features3, global_features3)

        x = hff_out3.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return
