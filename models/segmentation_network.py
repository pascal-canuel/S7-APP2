import torch
import torch.nn as nn

from constants import SEGMENTATION_N_OUTPUTS


# Based from U-Net
# Specs: max 1M parameters
# Goal: IoU > 0.8

class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        relu = nn.ReLU()
        leaky_relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, padding=0, stride=1),  # 32x52x52
            relu,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),  # 32x52x52
            relu,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),  # 32x52x52
            relu,
        )

        self._encoder_2 = nn.Sequential(
            maxpool,  # 32x26x26
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),  # 64x26x26
            relu,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),  # 64x26x26
            relu,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),  # 64x26x26
            relu,
        )

        self._encoder_3 = nn.Sequential(
            maxpool,  # 64x13x13
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),  # 128x13x13
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),  # 256x13x13
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),  # 128x13x13
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),  # 256x13x13
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),  # 128x13x13
            relu,
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),  # 64x13x13
            relu,
        )

        self._upsample_4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, padding=0,
                                             stride=2)  # 64x26x26
        self._decoder_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1),  # 64x26x26
            relu,
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),  # 32x26x26
            relu,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1),  # 32x26x26
            relu,
        )

        self._upsample_5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, padding=0,
                                             stride=2)  # 32x52x52
        self._decoder_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=1),  # 32x52x52
            relu,
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1),  # 16x52x52
            relu,
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1),  # 16x52x52
            relu,
        )

        self._conv_out = nn.Conv2d(in_channels=16, out_channels=SEGMENTATION_N_OUTPUTS, kernel_size=2, padding=1,
                                  stride=1)  # 4x53x53

    def forward(self, x):
        y1 = self._encoder_1(x)
        y2 = self._encoder_2(y1)
        y = self._encoder_3(y2)

        y = self._upsample_4(y)
        y = torch.cat((y, y2), 1)
        y = self._decoder_4(y)

        y = self._upsample_5(y)
        y = torch.cat((y, y1), 1)
        y = self._decoder_5(y)

        out = self._conv_out(y)

        return out
