import torch
import torch.nn as nn

from constants import INPUT_CHANNELS, N_CLASSES


class SegmentationNetwork(nn.Module):
    def __init__(self):
        super(SegmentationNetwork, self).__init__()

        self.hidden = 32

        # Down 1
        self.conv_1_1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_1_1 = nn.ReLU()
        self.conv_1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_1_2 = nn.ReLU()

        # Down 2
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_2_1 = nn.ReLU()
        self.conv_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_2_2 = nn.ReLU()

        # Down 3
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_1 = nn.ReLU()
        self.conv_3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_3_2 = nn.ReLU()

        # Down 4
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_4_1 = nn.ReLU()
        self.conv_4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_4_2 = nn.ReLU()

        # Down 5
        self.maxpool_5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_5_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_5_1 = nn.ReLU()
        self.conv_5_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_5_2 = nn.ReLU()

        # Up 6
        self.upsample_6 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=2)
        self.conv_6_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_6_1 = nn.ReLU()
        self.conv_6_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_6_2 = nn.ReLU()

        # Up 7
        self.upsample_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=2)
        self.conv_7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_7_1 = nn.ReLU()
        self.conv_7_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_7_2 = nn.ReLU()

        # Up 8
        self.upsample_8 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(2, 2), stride=2)
        self.conv_8_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_8_1 = nn.ReLU()
        self.conv_8_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_8_2 = nn.ReLU()

        # Up 9
        self.upsample_9 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=2)
        self.conv_9_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_9_1 = nn.ReLU()
        self.conv_9_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1)
        self.relu_9_2 = nn.ReLU()

        self.output_conv = nn.Conv2d(self.hidden, N_CLASSES, kernel_size=1)

    def forward(self, x):
        # Down 1
        y = self.conv_1_1(x)
        y = self.relu_1_1(y)
        y = self.conv_1_2(y)
        y = self.relu_1_2(y)
        y1 = y

        # Down 2
        y = self.maxpool_2(y)
        y = self.conv_2_1(y)
        y = self.relu_2_1(y)
        y = self.conv_2_2(y)
        y = self.relu_2_2(y)
        y2 = y

        # Down 3
        y = self.maxpool_3(y)
        y = self.conv_3_1(y)
        y = self.relu_3_1(y)
        y = self.conv_3_2(y)
        y = self.relu_3_2(y)
        y3 = y

        # Down 4
        y = self.maxpool_4(y)
        y = self.conv_4_1(y)
        y = self.relu_4_1(y)
        y = self.conv_4_2(y)
        y = self.relu_4_2(y)
        y4 = y

        # Down 5
        y = self.maxpool_5(y)
        y = self.conv_5_1(y)
        y = self.relu_5_1(y)
        y = self.conv_5_2(y)
        y = self.relu_5_2(y)

        # Up 6
        y = self.upsample_6(y)
        y_concat = torch.cat((y, y4), dim=1)
        y = self.conv_6_1(y_concat)
        y = self.relu_6_1(y)
        y = self.conv_6_2(y)
        y = self.relu_6_2(y)

        # Up 7
        y = self.upsample_7(y)
        y_concat = torch.cat((y, y3), dim=1)
        y = self.conv_7_1(y_concat)
        y = self.relu_7_1(y)
        y = self.conv_7_2(y)
        y = self.relu_7_2(y)

        # Up 8
        y = self.upsample_8(y)
        y_concat = torch.cat((y, y2), dim=1)
        y = self.conv_8_1(y_concat)
        y = self.relu_8_1(y)
        y = self.conv_8_2(y)
        y = self.relu_8_2(y)

        # Up 9
        y = self.upsample_9(y)
        y_concat = torch.cat((y, y1), dim=1)
        y = self.conv_9_1(y_concat)
        y = self.relu_9_1(y)
        y = self.conv_9_2(y)
        y = self.relu_9_2(y)

        # Out
        out = self.output_conv(y)

        return out
