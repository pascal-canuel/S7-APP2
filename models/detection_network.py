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

        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=(5, 5), padding=2, stride=1)
        self.relu_1_1 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv_3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_5_1 = nn.ReLU()
        self.conv_6_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_6_1 = nn.ReLU()
        self.conv_7_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_7_1 = nn.ReLU()
        self.maxpool_8 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.lin_9 = nn.Linear(in_features=64 * 5 * 5, out_features=64)
        self.relu_9 = nn.ReLU()
        self.lin_10 = nn.Linear(in_features=64, out_features=64)
        self.relu_10 = nn.ReLU()
        self.lin_11 = nn.Linear(in_features=64, out_features=N_OUTPUTS)
        self.sigmoid_11 = nn.Sigmoid()

    def forward(self, x):
        y = self.conv_1_1(x)
        y = self.relu_1_1(y)
        y = self.maxpool_2(y)
        y = self.conv_3_1(y)
        y = self.relu_3_1(y)
        y = self.maxpool_4(y)
        y = self.conv_5_1(y)
        y = self.relu_5_1(y)
        y = self.conv_6_1(y)
        y = self.relu_6_1(y)
        y = self.conv_7_1(y)
        y = self.relu_7_1(y)
        y = self.maxpool_8(y)
        y = y.view(y.shape[0], -1)
        y = self.lin_9(y)
        y = self.relu_9(y)
        y = self.lin_10(y)
        y = self.relu_10(y)
        y = self.lin_11(y)
        y = self.sigmoid_11(y)
        out = y

        out = out.view(out.shape[0], 3, -1)

        return out
