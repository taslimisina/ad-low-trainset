import torch
from torch import nn
import torch.nn.functional as F

class KldLoss(nn.Module):
    def __init__(self, reduction, temperature):
        super(KldLoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.KLDivLoss(reduction=reduction)
        self.temperature = temperature

    def forward(self, teacher_outs, student_outs):
        kd_loss = self.criterion(F.log_softmax(student_outs / self.temperature, dim=1),
                                 F.softmax(teacher_outs / self.temperature, dim=1))
        if self.reduction == "none":
            kd_loss = kd_loss.sum(dim=1)
        return kd_loss



class MseDirectionLoss(nn.Module):
    def __init__(self, lamda, reduction="mean"):
        super(MseDirectionLoss, self).__init__()
        self.lamda = lamda
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction=reduction)
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, output_real, output_pred):
        y_pred_0 = output_pred
        y_0 = output_real

        # different terms of loss
        abs_loss_0 = self.criterion(y_pred_0, y_0)
        loss_0 = 1 - self.similarity_loss(y_pred_0.view(y_pred_0.shape[0], -1), y_0.view(y_0.shape[0], -1))
        if self.reduction == "mean":
            loss_0 = torch.mean(loss_0)

        total_loss = loss_0 + self.lamda * (
                abs_loss_0)

        return total_loss