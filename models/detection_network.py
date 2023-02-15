import torch
import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES, DETECTION_N_OUTPUTS

# Based from AlexNet and YOLO
# Specs: max 400k parameters
# Goal: mAP > 0.35

N_OUTPUTS = N_CLASSES * DETECTION_N_OUTPUTS

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=(6, 6), padding=4, stride=1)
        self.relu_1_1 = nn.LeakyReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=1)
        self.relu_3_1 = nn.LeakyReLU()
        self.conv_4_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_4_1 = nn.LeakyReLU()
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_5_1 = nn.LeakyReLU()
        self.maxpool_6 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_7_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=1)
        self.relu_7_1 = nn.LeakyReLU()
        # self.conv_8_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        # self.relu_8_1 = nn.LeakyReLU()
        self.conv_9_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_9_1 = nn.LeakyReLU()
        self.maxpool_10 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_11_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(7, 7), stride=1)
        self.relu_11_1 = nn.LeakyReLU()
        self.conv_12_1 = nn.Conv2d(in_channels=256, out_channels=N_OUTPUTS, kernel_size=(1, 1), stride=1)
        self.sigmoid_12_1 = nn.Sigmoid()

    def forward(self, x):
        y = self.conv_1_1(x)
        y = self.relu_1_1(y)
        y = self.maxpool_2(y)

        y = self.conv_3_1(y)
        y = self.relu_3_1(y)
        y = self.conv_4_1(y)
        y = self.relu_4_1(y)
        y = self.conv_5_1(y)
        y = self.relu_5_1(y)
        y = self.maxpool_6(y)

        y = self.conv_7_1(y)
        y = self.relu_7_1(y)
        # y = self.conv_8_1(y)
        # y = self.relu_8_1(y)
        y = self.conv_9_1(y)
        y = self.relu_9_1(y)
        y = self.maxpool_10(y)

        y = self.conv_11_1(y)
        y = self.relu_11_1(y)
        y = self.conv_12_1(y)
        y = self.sigmoid_12_1(y)

        out = y
        # y = y.view(y.shape[0], -1)
        # out = y

        out = out.view(out.shape[0], 3, -1)

        return out
