import torch
import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES, DETECTION_N_OUTPUTS

# Based from AlexNet and YOLO
# Specs: max 400k parameters
# Goal: mAP > 0.35

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_1_1 = nn.LeakyReLU()
        self.conv_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=0, stride=1)
        self.relu_2_1 = nn.LeakyReLU()
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_1 = nn.LeakyReLU()

        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_5_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, stride=1)
        self.relu_5_1 = nn.LeakyReLU()
        self.conv_6_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_6_1 = nn.LeakyReLU()
        self.conv_7_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, stride=1)
        self.relu_7_1 = nn.LeakyReLU()
        self.conv_8_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_8_1 = nn.LeakyReLU()

        self.maxpool_9 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_10_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, stride=1)
        self.relu_10_1 = nn.LeakyReLU()
        self.conv_11_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_11_1 = nn.LeakyReLU()
        self.conv_12_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, stride=1)
        self.relu_12_1 = nn.LeakyReLU()
        self.conv_13_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_13_1 = nn.LeakyReLU()

        self.conv_14_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=2)
        self.relu_14_1 = nn.LeakyReLU()

        # ?x1x1
        self.conv_15_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(7, 7), padding=0, stride=1)
        self.relu_15_1 = nn.ReLU()

        # N_OUTPUTSx1x1
        self.conv_16_1 = nn.Conv2d(in_channels=256, out_channels=N_CLASSES * DETECTION_N_OUTPUTS, kernel_size=(1, 1), padding=0, stride=1)
        self.sigmoid_16_1 = nn.Sigmoid()

    def forward(self, x):
        y = self.conv_1_1(x)
        y = self.relu_1_1(y)
        y = self.conv_2_1(y)
        y = self.relu_2_1(y)
        y = self.conv_3_1(y)
        y = self.relu_3_1(y)

        y = self.maxpool_4(y)

        y = self.conv_5_1(y)
        y = self.relu_5_1(y)
        y = self.conv_6_1(y)
        y = self.relu_6_1(y)
        y = self.conv_7_1(y)
        y = self.relu_7_1(y)
        y = self.conv_8_1(y)
        y = self.relu_8_1(y)

        y = self.maxpool_9(y)

        y = self.conv_10_1(y)
        y = self.relu_10_1(y)
        y = self.conv_11_1(y)
        y = self.relu_11_1(y)
        y = self.conv_12_1(y)
        y = self.relu_12_1(y)
        y = self.conv_13_1(y)
        y = self.relu_13_1(y)

        y = self.conv_14_1(y)
        y = self.relu_14_1(y)

        y = self.conv_15_1(y)
        y = self.relu_15_1(y)

        y = self.conv_16_1(y)
        y = self.sigmoid_16_1(y)

        out = y

        # Reshape
        out = out.view(out.shape[0], 3, -1)

        return out
