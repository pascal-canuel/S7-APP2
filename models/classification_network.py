import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES


# Based from AlexNet and ResNet
# Specs: max 200k parameters
# Goal: accuracy > 90%

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        # extraction
        # self.fc1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=4, kernel_size=(3, 3))
        # self.fc2 = nn.BatchNorm2d(num_features=4)
        # self.fc3 = nn.ReLU()
        # self.fc4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.fc5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3))
        # self.fc6 = nn.BatchNorm2d(num_features=2)
        # self.fc7 = nn.ReLU()
        # self.fc8 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=(5, 5), padding=2, stride=1)
        self.relu_1_1 = nn.ReLU()

        # self.maxpool_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv_2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_2_1 = nn.ReLU()

        # self.maxpool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv_3_1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()

        self.maxpool_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        # classification
        # self.fc9 = nn.Linear(in_features=2 * 11 * 11, out_features=N_CLASSES)
        # self.fc10 = nn.Sigmoid()

        self.lin_5 = nn.Linear(in_features=8 * 26 * 26, out_features=32)
        self.relu_5 = nn.ReLU()

        self.lin_6 = nn.Linear(in_features=32, out_features=32)
        self.relu_6 = nn.ReLU()

        self.lin_7 = nn.Linear(in_features=32, out_features=N_CLASSES)
        self.sigmoid_7 = nn.Sigmoid()

    def forward(self, x):
        # extraction
        # y1 = self.fc1(x)
        # y2 = self.fc2(y1)
        # y3 = self.fc3(y2)
        # y4 = self.fc4(y3)
        # y5 = self.fc5(y4)
        # y6 = self.fc6(y5)
        # y7 = self.fc7(y6)
        # y8 = self.fc8(y7)

        y = self.conv_1_1(x)
        y = self.relu_1_1(y)
        # y = self.maxpool_2(y)
        y = self.conv_2_1(y)
        y = self.relu_2_1(y)
        # y = self.maxpool_3(y)
        y = self.conv_3_1(y)
        y = self.relu_3_1(y)
        y = self.maxpool_4(y)

        # classification
        # flat = y8.view(y8.shape[0], -1)
        # y9 = self.fc9(flat)
        # y10 = self.fc10(y9)

        # out = y10

        y = y.view(y.shape[0], -1)
        y = self.lin_5(y)
        y = self.relu_5(y)
        y = self.lin_6(y)
        y = self.relu_6(y)
        y = self.lin_7(y)
        y = self.sigmoid_7(y)

        out = y

        return out
