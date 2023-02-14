import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()
        self.ce_loss_presence = nn.BCELoss()
        # self.ce_loss_class = nn.CrossEntropyLoss()
        self.ce_loss_class = nn.BCELoss(reduction='sum')
        self.mse_loss_xywh = nn.MSELoss()

    def forward(self, output, target):
        # output: (N, 3, 7) -> treshold, x, y, wh, circle, triangle, cross
        # target: (N, 3, 5) -> presence, x, y, wh, class index

        # Classification
        output_obj_tresh = output[:, :, 0] # Nx3
        target_obj_presence = target[:, :, 0] # Nx3
        ce_loss_presence_val = self.ce_loss_presence(output_obj_tresh, target_obj_presence)

        # output_obj_class = output[:, :, 4:7]
        # target_obj_idx = target[:, :, 4].long()
        # ce_loss_class_val = self.ce_loss_class(output_obj_class, target_obj_idx)

        output_obj_class = output[:, :, 4:7] # Nx3x3
        
        target_obj_idx = target[:, :, 4].long() # Nx3
        target_obj_class = torch.zeros_like(output_obj_class) # Nx3x3
        for i in range(target_obj_idx.shape[0]): # for each batch
            for j in range(target_obj_idx.shape[1]): # for each object
                target_obj_class[i, j, target_obj_idx[i, j]] = 1 and target_obj_presence[i, j]

        ce_loss_class_val = self.ce_loss_class(output_obj_class, target_obj_class)

        # Regression
        # output_regression = output[:, :, 1:4]
        # target_regression = target[:, :, 1:4]
        #
        # mse_loss_xywh_val = self.mse_loss_xywh(output_regression, target_regression)

        Lx = self.mse_loss_xywh(output[:, :, 1], target[:, :, 1])
        Ly = self.mse_loss_xywh(output[:, :, 2], target[:, :, 2])
        Lwh = self.mse_loss_xywh(output[:, :, 3].sqrt(), target[:, :, 3].sqrt())

        mse_loss_xywh_val = Lx + Ly + 2*Lwh

        # Ponderate
        loss_tot = 5 * mse_loss_xywh_val + ce_loss_presence_val + 0.5 * (1 - ce_loss_presence_val) + ce_loss_class_val

        return loss_tot
