import torch
import torch.nn as nn


def loss_selector(loss_type):
    if loss_type == "mse":
        return nn.MSELoss(reduction="mean")
    elif loss_type == "mse_with_mask":
        return mse_with_mask
    elif loss_type == "mae":
        return nn.L1Loss(reduction="mean")
    elif loss_type == "mae_with_mask":
        return mae_with_mask
    else:
        return loss_type


def mse_with_mask(y_true, y_pred):
    target = y_true[:, 0:1, :, :]
    mask = y_true[:, 1:2, :, :]
    if torch.sum(mask) > 0:
        return torch.sum(torch.square(target - y_pred) * mask) / torch.sum(mask)
    else:
        return torch.mean(torch.square(target - y_pred))


def mae_with_mask(y_true, y_pred):
    target = y_true[:, 0:1, :, :]
    mask = y_true[:, 1:2, :, :]
    if torch.sum(mask) > 0:
        return torch.sum(torch.abs(target - y_pred) * mask) / torch.sum(mask)
    else:
        return torch.mean(torch.abs(target - y_pred))
