import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self, alpha):
        super(LocalizationLoss, self).__init__()
        self._alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum')
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # Classification
        output_obj_presence = output[:, :, 0]
        target_obj_presence = target[:, :, 0]

        output_obj_idx = output[:, :, 4]
        target_obj_idx = target[:, :, 4]

        output_classification = torch.cat((output_obj_presence, output_obj_idx))
        target_classification = torch.cat((target_obj_presence, target_obj_idx))
        ce_loss_val = self.ce_loss(output_classification, target_classification)

        # Regression
        output_regression = output[:, :, 1:4]
        target_regression = target[:, :, 1:4]

        mse_loss_val = self.mse_loss(output_regression, target_regression)

        # Ponderate
        loss_tot = self._alpha * ce_loss_val + mse_loss_val

        return loss_tot
