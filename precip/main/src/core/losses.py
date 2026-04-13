import torch
import torch.nn as nn

class BernoulliGammaLoss(nn.Module):
    def __init__(self):
        super(BernoulliGammaLoss, self).__init__()

    def forward(self, pred, true):
        """
        pred: (B, 3, H, W) - [occurrence, shape, scale]
        true: (B, 1, H, W) - target precipitation
        """
        eps = 1e-5
        
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
            
        occurrence = torch.sigmoid(pred[:, 0, :, :]).clamp(eps, 1 - eps)
        shape_parameter = torch.exp(pred[:, 1, :, :].clamp(-5, 5)).clamp(eps, 1e3)
        scale_parameter = torch.exp(pred[:, 2, :, :].clamp(-5, 5)).clamp(eps, 1e3)
        
        bool_rain = (true > 0).float()
        epsilon = 1e-6

        loss = (-torch.mean((1 - bool_rain) * torch.log(1 - occurrence + epsilon) + 
                             bool_rain * (torch.log(occurrence + epsilon) + 
                                          (shape_parameter - 1) * torch.log(true + epsilon) -
                                          shape_parameter * torch.log(scale_parameter + epsilon) -
                                          torch.lgamma(shape_parameter + epsilon) -
                                          true / (scale_parameter + epsilon))))

        return loss

import math

class GaussianLoss(nn.Module):
    def __init__(self):
        super(GaussianLoss, self).__init__()

    def forward(self, pred, true):
        """
        pred: (B, 2, H, W) - [mean, log_var]
        true: (B, 1, H, W) - target temperature
        """
        if true.dim() == 4 and true.size(1) == 1:
            true = true.squeeze(1)
            
        mean = pred[:, 0, :, :]
        log_var = pred[:, 1, :, :] # This is ln(sigma^2)
        
        loss = 0.5 * torch.mean(math.log(2 * math.pi) + log_var + torch.exp(-log_var) * (true - mean)**2)
        return loss
