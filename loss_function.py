import torch
import torch.nn as nn


class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()
        self.bce_loss_presence = nn.BCELoss()
        self.ce_loss_class = nn.CrossEntropyLoss()
        # self.bce_loss_class = nn.BCELoss(reduction='sum')
        self.mse_loss_xywh = nn.MSELoss()

    def forward(self, output, target):
        # pred = output
        # t = torch.zeros((*target.shape[0:2], 7), device='cuda')
        # t_sorted = torch.zeros((*target.shape[0:2], 7), device='cuda')
        # t[:, :, :5] += target
        # t[:, :, 5] = torch.where(t[:, :, 4] == 1.0, 1.0, 0.0)
        # t[:, :, 6] = torch.where(t[:, :, 4] == 2.0, 1.0, 0.0)
        # t[:, :, 4] = torch.where(t[:, :, 4] == 0.0, 1.0, 0.0) * t[:, :, 0]
        # for i, t1 in enumerate(t):
        #     for j, t2 in enumerate(t1):
        #         if t2[4] == 1.0:
        #             t_sorted[i, 0, :] = t[i, j, :]
        #         elif t2[5] == 1.0:
        #             t_sorted[i, 1, :] = t[i, j, :]
        #         elif t2[6] == 1.0:
        #             t_sorted[i, 2, :] = t[i, j, :]
        # bce_loss = torch.nn.BCELoss()
        # mse_loss = torch.nn.MSELoss()
        # l_obj = bce_loss(pred[:, :, 0], t_sorted[:, :, 0])
        # l_xywh = mse_loss(pred[:, :, 1:4], t_sorted[:, :, 1:4])
        # l_class = bce_loss(pred[:, :, 4:7], t_sorted[:, :, 4:7])
        #
        # return 5 * l_xywh + l_obj + 0.5 * (1 - l_obj) + l_class

        # output: (N, 3, 7) -> treshold, x, y, wh, circle, triangle, cross
        # target: (N, 3, 5) -> presence, x, y, wh, class index

        # Classification
        output_obj_tresh = output[:, :, 0] # Nx3
        target_obj_presence = target[:, :, 0] # Nx3
        ce_loss_presence_val = self.bce_loss_presence(output_obj_tresh, target_obj_presence)

        # output_obj_class = output[:, :, 4:7]
        target_obj_idx = target[:, :, 4].long()
        # ce_loss_class_val = self.ce_loss_class(output_obj_class, target_obj_idx)

        # TRICKSHOT
        # output_trickshot = output

        # never predict a circle with this one
        output_trickshot = torch.zeros_like(output)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(1, output.shape[2]):
                    if target[i, j, 0] != 0:
                        output_trickshot[i, j, k] = output[i, j, k]

        output_obj_class = output_trickshot[:, :, 4:7] # Nx3x3

        # # sort
        # target_obj_idx = target[:, :, 4].long() # Nx3
        # target_obj_class = torch.zeros_like(output_obj_class) # Nx3x3
        # for i in range(target_obj_idx.shape[0]): # for each batch
        #     for j in range(target_obj_idx.shape[1]): # for each object
        #         target_obj_class[i, j, target_obj_idx[i, j]] = 1 and target_obj_presence[i, j]

        ce_loss_class_val = self.ce_loss_class(output_obj_class, target_obj_idx)
        # ce_loss_class_val = self.bce_loss_class(output_obj_class, target_obj_class)

        # Regression
        # output_regression = output[:, :, 1:4]
        # target_regression = target[:, :, 1:4]
        #
        # mse_loss_xywh_val = self.mse_loss_xywh(output_regression, target_regression)

        Lx = self.mse_loss_xywh(output_trickshot[:, :, 1], target[:, :, 1])
        Ly = self.mse_loss_xywh(output_trickshot[:, :, 2], target[:, :, 2])
        Lwh = self.mse_loss_xywh(output_trickshot[:, :, 3].sqrt(), target[:, :, 3].sqrt())

        mse_loss_xywh_val = Lx + Ly + 2*Lwh

        # Ponderate
        loss_tot = 5 * mse_loss_xywh_val + ce_loss_presence_val + 0.5 * (1 - ce_loss_presence_val) + 3 * ce_loss_class_val

        return loss_tot
