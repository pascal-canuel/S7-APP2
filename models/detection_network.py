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

        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self._detection = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=32, kernel_size=5, padding=2, stride=2),
            relu,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            relu,
            maxpool,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            relu,
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            relu,
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            relu,
            maxpool,
            nn.Flatten(),
            nn.Linear(in_features=64 * 6 * 6, out_features=64),
            relu,
            nn.Linear(in_features=64, out_features=64),
            relu,
            nn.Linear(in_features=64, out_features=N_OUTPUTS),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self._detection(x)
        out = y.view(y.shape[0], 3, -1)

        return out


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # Presence classification
        output_obj_tresh = output[:, :, 0]
        target_obj_presence = target[:, :, 0]
        bce_loss_val = self.bce_loss(output_obj_tresh, target_obj_presence)

        # Class index classification
        filtered_output = torch.zeros_like(output, device='cuda')
        sorted_target = torch.zeros_like(target, device='cuda')

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                for k in range(target.shape[2]):
                    if target[i, j, 0] != 0:
                        sorted_target[i, target[i, j, 4].long(), k] = target[i, j, k]
                        filtered_output[i, j, k] = output[i, j, k]

        sorted_target_class_idx = torch.squeeze(sorted_target[:, :, 4]).long()
        ce_loss_val = self.ce_loss(filtered_output[:, :, 4:], sorted_target_class_idx)

        # Dimension regression
        Lx = self.mse_loss(filtered_output[:, :, 1], sorted_target[:, :, 1])
        Ly = self.mse_loss(filtered_output[:, :, 2], sorted_target[:, :, 2])
        Lwh = self.mse_loss(filtered_output[:, :, 3].sqrt(), sorted_target[:, :, 3].sqrt())

        mse_loss_val = Lx + Ly + 2 * Lwh

        # Ponderate loss
        loss = 10 * mse_loss_val + bce_loss_val + 0.5 * (1 - bce_loss_val) + ce_loss_val

        return loss
