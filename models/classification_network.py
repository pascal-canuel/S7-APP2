import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES


# Based from AlexNet and ResNet
# Specs: max 200k parameters
# Goal: accuracy > 90%

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        # TODO: test with stride of 2
        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=(5, 5), padding=2, stride=1)
        self.relu_1_1 = nn.ReLU()
        # TODO: test with max pool with stride of 1
        self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_4_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_4_1 = nn.ReLU()
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.lin_6 = nn.Linear(in_features=8 * 26 * 26, out_features=32)
        self.relu_6 = nn.ReLU()
        self.lin_7 = nn.Linear(in_features=32, out_features=64)
        self.relu_7 = nn.ReLU()
        self.lin_8 = nn.Linear(in_features=64, out_features=N_CLASSES)
        self.sigmoid_8 = nn.Sigmoid()

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
        # torch.flatten(y, start_dim=1)
        y = self.lin_6(y)
        y = self.relu_6(y)
        y = self.lin_7(y)
        y = self.relu_7(y)
        y = self.lin_8(y)
        y = self.sigmoid_8(y)
        out = y

        return out
