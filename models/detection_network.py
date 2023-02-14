import torch
import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES, DETECTION_N_OUTPUTS

# Based from AlexNet and YOLO
# Specs: max 400k parameters
# Goal: mAP > 0.35

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=(5, 5), padding=2, stride=1)
        self.relu_1_1 = nn.ReLU()
        self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_4_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_4_1 = nn.ReLU()
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.lin_6 = nn.Linear(in_features=8 * 26 * 26, out_features=32)
        self.relu_6 = nn.ReLU()
        self.lin_7 = nn.Linear(in_features=32, out_features=32)
        self.relu_7 = nn.ReLU()
        self.lin_8 = nn.Linear(in_features=32, out_features=64)
        self.relu_8 = nn.ReLU()
        self.lin_9 = nn.Linear(in_features=64, out_features=DETECTION_N_OUTPUTS)
        self.sigmoid_9 = nn.Sigmoid()

    def forward(self, x):
        y = self.conv_1_1(x)
        residual = y
        y = self.relu_1_1(y)
        y = self.conv_2_1(y)
        y = self.relu_2_1(y)
        y = self.conv_3_1(y)
        y += residual
        y = self.relu_3_1(y)
        y = self.conv_4_1(y)
        y = self.relu_4_1(y)
        y = self.maxpool_5(y)
        y = y.view(y.shape[0], -1)
        y = self.lin_6(y)
        y = self.relu_6(y)
        y = self.lin_7(y)
        y = self.relu_7(y)
        y = self.lin_8(y)
        y = self.relu_8(y)
        y = self.lin_9(y)
        y = self.sigmoid_9(y)
        out = y

        return out
