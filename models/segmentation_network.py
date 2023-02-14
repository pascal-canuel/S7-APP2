import torch
import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES

# Based from U-Net
# Specs: max 1M parameters
# Goal: IoU > 0.8

class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        # Down 1
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, padding=0, stride=1)  # 32x52x52
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)  # 32x52x52
        self.relu_1_2 = nn.ReLU()
        self.conv_1_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)  # 32x52x52
        self.relu_1_3 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # 32x26x26
        self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)  # 64x26x26
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)  # 64x26x26
        self.relu_2_2 = nn.ReLU()
        self.conv_2_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)  # 64x26x26
        self.relu_2_3 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, padding=0, stride=2)  # 64x13x13
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)  # 128x13x13
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)  # 256x13x13
        self.relu_3_2 = nn.ReLU()
        self.conv_3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)  # 128x13x13
        self.relu_3_3 = nn.ReLU()
        self.conv_3_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)  # 256x13x13
        self.relu_3_4 = nn.ReLU()
        self.conv_3_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)  # 128x13x13
        self.relu_3_5 = nn.ReLU()
        self.conv_3_6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)  # 64x13x13
        self.relu_3_6 = nn.ReLU()

        # Up 4
        self.upsample_4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, padding=0,
                                             stride=2)  # 64x26x26
        # Concat
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)  # 64x26x26
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)  # 32x26x26
        self.relu_4_2 = nn.ReLU()
        self.conv_4_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1)  # 32x26x26
        self.relu_4_3 = nn.ReLU()

        # Up 5
        self.upsample_5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, padding=0,
                                             stride=2)  # 32x52x52
        # Concat
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1)  # 32x52x52
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)  # 16x52x52
        self.relu_5_2 = nn.ReLU()
        self.conv_5_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1)  # 16x52x52
        self.relu_5_3 = nn.ReLU()

        self.conv_5_4 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=2, padding=1, stride=1)  # 4x53x53
    def forward(self, x):
        # Down 1
        y = self.conv_1_1(x)
        y = self.relu_1_1(y)
        y = self.conv_1_2(y)
        y = self.relu_1_2(y)
        y = self.conv_1_3(y)
        y1 = self.relu_1_3(y)

        # Down 2
        y = self.maxpool_2(y1)
        y = self.conv_2_1(y)
        y = self.relu_2_1(y)
        y = self.conv_2_2(y)
        y = self.relu_2_3(y)
        y = self.conv_2_3(y)
        y2 = self.relu_2_2(y)

        # Down 3
        y = self.maxpool_3(y2)
        y = self.conv_3_1(y)
        y = self.relu_3_1(y)
        y = self.conv_3_2(y)
        y = self.relu_3_2(y)
        y = self.conv_3_3(y)
        y = self.relu_3_3(y)
        y = self.conv_3_4(y)
        y = self.relu_3_4(y)
        y = self.conv_3_5(y)
        y = self.relu_3_5(y)
        y = self.conv_3_6(y)
        y = self.relu_3_6(y)

        # Up 4
        y = self.upsample_4(y)
        y = torch.cat((y, y2), 1)
        y = self.conv_4_1(y)
        y = self.relu_4_1(y)
        y = self.conv_4_2(y)
        y = self.relu_4_2(y)
        y = self.conv_4_3(y)
        y = self.relu_4_3(y)

        # Up 5
        y = self.upsample_5(y)
        y = torch.cat((y, y1), 1)
        y = self.conv_5_1(y)
        y = self.relu_5_1(y)
        y = self.conv_5_2(y)
        y = self.relu_5_2(y)
        y = self.conv_5_3(y)
        y = self.relu_5_3(y)
        y = self.conv_5_4(y)  # 32x4x53x53

        out = y

        return out
