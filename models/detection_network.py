import torch
import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES

# Based from YOLO
# Specs: max 400k parameters
# Goal: mAP > 0.35

class DetectionNetwork(nn.Module):
    def __init__(self):
        super(DetectionNetwork, self).__init__()

        self.fc1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=4, kernel_size=(3, 3))
        self.fc2 = nn.BatchNorm2d(num_features=4)
        self.fc3 = nn.ReLU()
        self.fc4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3))
        self.fc6 = nn.BatchNorm2d(num_features=2)
        self.fc7 = nn.ReLU()
        self.fc8 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc9 = nn.Linear(in_features=2 * 11 * 11, out_features=3 * 5)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)
        y4 = self.fc4(y3)
        y5 = self.fc5(y4)
        y6 = self.fc6(y5)
        y7 = self.fc7(y6)
        y8 = self.fc8(y7)
        flat = y8.view(y8.shape[0], -1)
        out = self.fc9(flat)

        return out
