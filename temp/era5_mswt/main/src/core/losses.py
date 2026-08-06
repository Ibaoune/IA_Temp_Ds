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


def _validate_xiong_fields(pred, target):
    """Validate the deterministic temperature fields used by Xiong losses."""
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise TypeError("pred and target must both be PyTorch tensors.")

    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape; "
            f"got pred={tuple(pred.shape)} and target={tuple(target.shape)}."
        )

    if pred.ndim != 4:
        raise ValueError(
            "Xiong losses expect four-dimensional fields with shape (B, 1, H, W); "
            f"got {pred.ndim} dimensions."
        )

    if pred.shape[0] < 1:
        raise ValueError("Xiong losses require a non-empty batch (B >= 1).")

    if pred.shape[1] != 1:
        raise ValueError(
            "Xiong losses expect exactly one temperature channel; "
            f"got C={pred.shape[1]}."
        )

    height, width = pred.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError(
            "Xiong losses require H >= 2 and W >= 2; "
            f"got H={height} and W={width}."
        )

    if pred.device != target.device:
        raise ValueError(
            "pred and target must be on the same device; "
            f"got pred={pred.device} and target={target.device}."
        )

    if pred.dtype != target.dtype:
        raise ValueError(
            "pred and target must have the same dtype; "
            f"got pred={pred.dtype} and target={target.dtype}."
        )

    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("Xiong losses require floating-point tensors.")

    if not bool(torch.isfinite(pred).all()):
        raise ValueError("pred contains NaN or infinite values.")

    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains NaN or infinite values.")


def _validate_xiong_parameters(weight, eps, loss_name):
    """Cast and validate scalar hyperparameters shared by the Xiong losses."""
    try:
        weight = float(weight)
        eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{loss_name} weight and eps must be real numbers.") from exc

    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{loss_name} weight must be finite and >= 0; got {weight}.")

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"{loss_name} eps must be finite and > 0; got {eps}.")

    return weight, eps


def _xiong_computation_fields(pred, target):
    """Use float32 accumulation for low-precision inputs without changing device."""
    if pred.dtype in {torch.float16, torch.bfloat16}:
        return pred.float(), target.float()
    return pred, target


class XiongContinuityLoss(nn.Module):
    r"""RMSE with Xiong's spatial-continuity constraint.

    For a field ``F`` on an ``H x W`` grid, the continuity energy is

    ``C(F) = sum_{i=1}^{H-1} sum_j (F[i,j] - F[i+1,j])^2``
    ``     + sum_i sum_{j=1}^{W-1} (F[i,j] - F[i,j+1])^2``.

    This module implements

    ``L_c = sqrt(mean((pred - target)^2) + eps)``
    ``      + weight * mean_b(|C(pred_b) - C(target_b)|)``.

    The spatial reductions are sums, as defined in the paper; only the final
    constraint error is averaged over the batch (and singleton channel).

    Reference
    ---------
    Minquan Xiong (2025), "Impact of Physical Constraints on Deep
    Learning-Based Downscaling Prediction of Temperature", Journal of
    Meteorological Research, 39(4), 904-919.
    DOI: 10.1007/s13351-025-4061-1.

    Parameters
    ----------
    weight : float, default=1.0e-4
        Multiplicative continuity weight ``w_c``.
    eps : float, default=1.0e-8
        Positive numerical-stability constant used in the RMSE.
    """

    def __init__(self, weight=1.0e-4, eps=1.0e-8):
        super().__init__()
        self.weight, self.eps = _validate_xiong_parameters(
            weight, eps, self.__class__.__name__
        )

    def forward(self, pred, target):
        _validate_xiong_fields(pred, target)
        pred_compute, target_compute = _xiong_computation_fields(pred, target)

        rmse = torch.sqrt(
            torch.mean((pred_compute - target_compute) ** 2) + self.eps
        )

        pred_vertical = pred_compute[..., 1:, :] - pred_compute[..., :-1, :]
        pred_horizontal = pred_compute[..., :, 1:] - pred_compute[..., :, :-1]

        target_vertical = target_compute[..., 1:, :] - target_compute[..., :-1, :]
        target_horizontal = target_compute[..., :, 1:] - target_compute[..., :, :-1]

        continuity_pred = (
            pred_vertical.square().sum(dim=(-2, -1))
            + pred_horizontal.square().sum(dim=(-2, -1))
        )
        continuity_target = (
            target_vertical.square().sum(dim=(-2, -1))
            + target_horizontal.square().sum(dim=(-2, -1))
        )

        continuity_error = torch.abs(
            continuity_pred - continuity_target
        ).mean()
        loss = (rmse + self.weight * continuity_error).to(dtype=pred.dtype)

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "XiongContinuityLoss produced a NaN or infinite value."
            )

        return loss


class XiongDirectionalLoss(nn.Module):
    r"""RMSE with Xiong's directional-consistency constraint.

    For a field ``F`` on an ``H x W`` grid, the directional quantity is

    ``D(F) = sum_{i=1}^{H-1} sum_j atan2(F[i+1,j], F[i,j])``
    ``     + sum_i sum_{j=1}^{W-1} atan2(F[i,j+1], F[i,j])``.

    PyTorch uses ``atan2(y, x)``, so every ordered pair ``(current, neighbor)``
    is evaluated as ``torch.atan2(neighbor, current)``. This module implements

    ``L_d = sqrt(mean((pred - target)^2) + eps)``
    ``      + weight * mean_b(|D(pred_b) - D(target_b)|)``.

    Reference
    ---------
    Minquan Xiong (2025), "Impact of Physical Constraints on Deep
    Learning-Based Downscaling Prediction of Temperature", Journal of
    Meteorological Research, 39(4), 904-919.
    DOI: 10.1007/s13351-025-4061-1.

    Parameters
    ----------
    weight : float, default=1.0e-4
        Multiplicative directional weight ``w_d``.
    eps : float, default=1.0e-8
        Positive numerical-stability constant used in the RMSE.
    """

    def __init__(self, weight=1.0e-4, eps=1.0e-8):
        super().__init__()
        self.weight, self.eps = _validate_xiong_parameters(
            weight, eps, self.__class__.__name__
        )

    def forward(self, pred, target):
        _validate_xiong_fields(pred, target)
        pred_compute, target_compute = _xiong_computation_fields(pred, target)

        rmse = torch.sqrt(
            torch.mean((pred_compute - target_compute) ** 2) + self.eps
        )

        pred_angle_vertical = torch.atan2(
            pred_compute[..., 1:, :],
            pred_compute[..., :-1, :],
        )
        pred_angle_horizontal = torch.atan2(
            pred_compute[..., :, 1:],
            pred_compute[..., :, :-1],
        )

        target_angle_vertical = torch.atan2(
            target_compute[..., 1:, :],
            target_compute[..., :-1, :],
        )
        target_angle_horizontal = torch.atan2(
            target_compute[..., :, 1:],
            target_compute[..., :, :-1],
        )

        direction_pred = (
            pred_angle_vertical.sum(dim=(-2, -1))
            + pred_angle_horizontal.sum(dim=(-2, -1))
        )
        direction_target = (
            target_angle_vertical.sum(dim=(-2, -1))
            + target_angle_horizontal.sum(dim=(-2, -1))
        )

        direction_error = torch.abs(direction_pred - direction_target).mean()
        loss = (rmse + self.weight * direction_error).to(dtype=pred.dtype)

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "XiongDirectionalLoss produced a NaN or infinite value."
            )

        return loss


def _validate_serifi_fields(prediction, target):
    """Validate scalar spatial fields used by :class:`SerifiGradientLoss`."""
    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise TypeError("prediction and target must both be PyTorch tensors.")

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape; "
            f"got prediction={tuple(prediction.shape)} "
            f"and target={tuple(target.shape)}."
        )

    if prediction.ndim != 4:
        raise ValueError(
            "SerifiGradientLoss expects four-dimensional fields with shape "
            f"(B, 1, H, W); got {prediction.ndim} dimensions."
        )

    if prediction.shape[0] < 1:
        raise ValueError("SerifiGradientLoss requires a non-empty batch (B >= 1).")

    if prediction.shape[1] != 1:
        raise ValueError(
            "SerifiGradientLoss expects exactly one scalar-field channel; "
            f"got C={prediction.shape[1]}."
        )

    height, width = prediction.shape[-2:]
    if height < 2 or width < 2:
        raise ValueError(
            "SerifiGradientLoss requires H >= 2 and W >= 2; "
            f"got H={height} and W={width}."
        )

    if prediction.device != target.device:
        raise ValueError(
            "prediction and target must be on the same device; "
            f"got prediction={prediction.device} and target={target.device}."
        )

    if prediction.dtype != target.dtype:
        raise ValueError(
            "prediction and target must have the same dtype; "
            f"got prediction={prediction.dtype} and target={target.dtype}."
        )

    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("SerifiGradientLoss requires floating-point tensors.")

    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction contains NaN or infinite values.")

    if not bool(torch.isfinite(target).all()):
        raise ValueError("target contains NaN or infinite values.")


def _validate_serifi_gradient_weight(gradient_weight):
    """Convert and validate the Serifi spatial-gradient multiplier."""
    try:
        gradient_weight = float(gradient_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "SerifiGradientLoss gradient_weight must be a real number; "
            f"got {gradient_weight!r}."
        ) from exc

    if not math.isfinite(gradient_weight) or gradient_weight < 0:
        raise ValueError(
            "SerifiGradientLoss gradient_weight must be finite and >= 0; "
            f"got {gradient_weight}."
        )

    return gradient_weight


def _serifi_computation_fields(prediction, target):
    """Accumulate low-precision inputs in float32 on their original device."""
    if prediction.dtype in {torch.float16, torch.bfloat16}:
        return prediction.float(), target.float()
    return prediction, target


class SerifiGradientLoss(nn.Module):
    r"""L1 data loss with the spatial-gradient constraint of Serifi et al.

    For scalar fields with shape ``(B, 1, H, W)``, forward differences are

    ``dx(F) = F[..., :, 1:] - F[..., :, :-1]``
    ``dy(F) = F[..., 1:, :] - F[..., :-1, :]``.

    This module implements the local, axis-wise loss

    ``L_data = mean(|prediction - target|)``
    ``L_grad,x = mean(|dx(prediction) - dx(target)|)``
    ``L_grad,y = mean(|dy(prediction) - dy(target)|)``
    ``L = L_data + gradient_weight * (L_grad,x + L_grad,y)``.

    The two spatial derivatives are compared locally and averaged separately;
    this is neither a temporal-gradient term nor a global conservation law.

    Reference
    ---------
    Agon Serifi, Tobias Günther, and Nikolina Ban (2021), "Spatio-Temporal
    Downscaling of Climate Data Using Convolutional and Error-Predicting
    Neural Networks", Frontiers in Climate, 3:656479.
    DOI: 10.3389/fclim.2021.656479.

    Parameters
    ----------
    gradient_weight : float, default=1.0
        Non-negative multiplier of the summed x/y spatial-gradient losses.
    """

    def __init__(self, gradient_weight=1.0):
        super().__init__()
        self.gradient_weight = _validate_serifi_gradient_weight(gradient_weight)

    def forward(self, prediction, target):
        _validate_serifi_fields(prediction, target)
        prediction_compute, target_compute = _serifi_computation_fields(
            prediction, target
        )

        data_loss = torch.mean(torch.abs(prediction_compute - target_compute))

        pred_dx = (
            prediction_compute[:, :, :, 1:]
            - prediction_compute[:, :, :, :-1]
        )
        true_dx = target_compute[:, :, :, 1:] - target_compute[:, :, :, :-1]

        pred_dy = (
            prediction_compute[:, :, 1:, :]
            - prediction_compute[:, :, :-1, :]
        )
        true_dy = target_compute[:, :, 1:, :] - target_compute[:, :, :-1, :]

        gradient_x_loss = torch.mean(torch.abs(pred_dx - true_dx))
        gradient_y_loss = torch.mean(torch.abs(pred_dy - true_dy))

        loss = (
            data_loss
            + self.gradient_weight * (gradient_x_loss + gradient_y_loss)
        ).to(dtype=prediction.dtype)

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "SerifiGradientLoss produced a NaN or infinite value."
            )

        return loss
