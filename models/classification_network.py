import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES


# Based from AlexNet and ResNet
# Specs: max 200k parameters
# Goal: accuracy > 90%

class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()

        relu = nn.ReLU()

        self._classifier = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=5, padding=2),
            relu,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            relu,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            relu,
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            relu,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32 * 12 * 12, out_features=32),
            relu,
            nn.Linear(in_features=32, out_features=32),
            relu,
            nn.Linear(in_features=32, out_features=N_CLASSES),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self._classifier(x)
