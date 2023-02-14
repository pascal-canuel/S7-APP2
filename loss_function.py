import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()
        self.ce_loss_presence = nn.BCELoss()
        self.ce_loss_class = nn.CrossEntropyLoss()
        self.mse_loss_xywh = nn.MSELoss()

    def forward(self, output, target):
        # output: (N, 3, 7) -> treshold, x, y, wh, circle, triangle, cross
        # target: (N, 3, 5) -> presence, x, y, wh, class index

        # Classification
        output_obj_tresh = output[:, :, 0]
        target_obj_presence = target[:, :, 0]
        ce_loss_presence_val = self.ce_loss_presence(output_obj_tresh, target_obj_presence)

        output_obj_class = output[:, :, 4:7]
        target_obj_idx = target[:, :, 4].long()
        ce_loss_class_val = self.ce_loss_class(output_obj_class, target_obj_idx)

        # Regression
        output_regression = output[:, :, 1:4]
        target_regression = target[:, :, 1:4]

        mse_loss_xywh_val = self.mse_loss_xywh(output_regression, target_regression)

        # Ponderate
        loss_tot = 5 * mse_loss_xywh_val + ce_loss_presence_val + 0.5 * (1 - ce_loss_presence_val) + ce_loss_class_val

        return loss_tot
