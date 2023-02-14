import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(LocalizationLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self.ce_loss_presence = nn.CrossEntropyLoss()
        self.ce_loss_class = nn.CrossEntropyLoss()
        self.mse_loss_xywh = nn.MSELoss()

    def forward(self, output, target):
        # Classification
        output_obj_presence = output[:, :, 0]
        target_obj_presence = target[:, :, 0]
        ce_loss_presence_val = self.ce_loss_presence(output_obj_presence, target_obj_presence)

        output_obj_idx = output[:, :, 4]
        target_obj_idx = target[:, :, 4]
        ce_loss_class_val = self.ce_loss_class(output_obj_idx, target_obj_idx)

        # Regression
        output_regression = output[:, :, 1:4]
        target_regression = target[:, :, 1:4]

        mse_loss_xywh_val = self.mse_loss_xywh(output_regression, target_regression)

        # Ponderate
        loss_tot = self._alpha * mse_loss_xywh_val + self._beta * ce_loss_presence_val + ce_loss_class_val

        return loss_tot
