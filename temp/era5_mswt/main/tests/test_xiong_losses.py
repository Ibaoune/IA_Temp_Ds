import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import xarray as xr
import yaml

from src.core.config import Config, load_config
from src.core.losses import (
    BernoulliGammaLoss,
    GaussianLoss,
    XiongContinuityLoss,
    XiongDirectionalLoss,
)


MAIN_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = MAIN_DIR / "configs" / "unet"
XIONG_CONFIGS = {
    "xiong_continuity": CONFIG_DIR / "config_arch1_xiong_continuity.yaml",
    "xiong_directional": CONFIG_DIR / "config_arch1_xiong_directional.yaml",
}
EPS = 1.0e-8


def _rmse(pred, target, eps=EPS):
    return torch.sqrt(torch.mean((pred - target) ** 2) + eps)


def _continuity_error(pred, target):
    pred_vertical = pred[..., 1:, :] - pred[..., :-1, :]
    pred_horizontal = pred[..., :, 1:] - pred[..., :, :-1]
    target_vertical = target[..., 1:, :] - target[..., :-1, :]
    target_horizontal = target[..., :, 1:] - target[..., :, :-1]

    continuity_pred = (
        pred_vertical.square().sum(dim=(-2, -1))
        + pred_horizontal.square().sum(dim=(-2, -1))
    )
    continuity_target = (
        target_vertical.square().sum(dim=(-2, -1))
        + target_horizontal.square().sum(dim=(-2, -1))
    )
    return torch.abs(continuity_pred - continuity_target).mean()


def _direction_error(pred, target):
    direction_pred = (
        torch.atan2(pred[..., 1:, :], pred[..., :-1, :]).sum(dim=(-2, -1))
        + torch.atan2(pred[..., :, 1:], pred[..., :, :-1]).sum(dim=(-2, -1))
    )
    direction_target = (
        torch.atan2(target[..., 1:, :], target[..., :-1, :]).sum(dim=(-2, -1))
        + torch.atan2(target[..., :, 1:], target[..., :, :-1]).sum(dim=(-2, -1))
    )
    return torch.abs(direction_pred - direction_target).mean()


def _direction_error_with_reversed_atan2_arguments(pred, target):
    direction_pred = (
        torch.atan2(pred[..., :-1, :], pred[..., 1:, :]).sum(dim=(-2, -1))
        + torch.atan2(pred[..., :, :-1], pred[..., :, 1:]).sum(dim=(-2, -1))
    )
    direction_target = (
        torch.atan2(target[..., :-1, :], target[..., 1:, :]).sum(dim=(-2, -1))
        + torch.atan2(target[..., :, :-1], target[..., :, 1:]).sum(dim=(-2, -1))
    )
    return torch.abs(direction_pred - direction_target).mean()


def _yaml_dict(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_config_with_temp_results(path, tmp_path):
    data = _yaml_dict(path)
    data["paths"]["results_dir"] = str(tmp_path / "results")
    copied_path = tmp_path / path.name
    copied_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_config(path=str(copied_path)), data


@pytest.mark.parametrize(
    ("loss_class", "constraint_fn"),
    [
        (XiongContinuityLoss, _continuity_error),
        (XiongDirectionalLoss, _direction_error),
    ],
)
def test_identical_fields_have_zero_errors(loss_class, constraint_fn):
    target = torch.tensor(
        [[[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]]]
    )
    pred = target.clone()

    raw_rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    constraint_error = constraint_fn(pred, target)
    loss = loss_class(weight=1.0e-4, eps=EPS)(pred, target)

    assert raw_rmse.item() == 0.0
    assert constraint_error.item() == 0.0
    # The mandated stabilized RMSE is sqrt(0 + eps), not mathematically zero.
    assert loss.item() == pytest.approx(math.sqrt(EPS), rel=1.0e-6)
    assert loss.item() == pytest.approx(0.0, abs=1.1 * math.sqrt(EPS))


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
def test_zero_weight_is_exactly_stabilized_rmse(loss_class):
    pred = torch.tensor([[[[1.0, 3.0], [2.0, 5.0]]]], dtype=torch.float64)
    target = torch.tensor([[[[2.0, 2.0], [4.0, 1.0]]]], dtype=torch.float64)

    actual = loss_class(weight=0.0, eps=EPS)(pred, target)

    assert torch.allclose(actual, _rmse(pred, target), rtol=1.0e-12, atol=0.0)


def test_different_continuity_increases_continuity_loss():
    target = torch.zeros((1, 1, 2, 2))
    pred = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    weight = 0.1

    continuity_error = _continuity_error(pred, target)
    rmse = _rmse(pred, target)
    loss = XiongContinuityLoss(weight=weight, eps=EPS)(pred, target)

    assert continuity_error.item() == pytest.approx(10.0)
    assert loss.item() > rmse.item()
    assert torch.allclose(loss, rmse + weight * continuity_error)


def test_different_directions_increase_directional_loss():
    target = torch.ones((1, 1, 2, 2))
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    weight = 0.1

    direction_error = _direction_error(pred, target)
    rmse = _rmse(pred, target)
    loss = XiongDirectionalLoss(weight=weight, eps=EPS)(pred, target)

    assert direction_error.item() > 0.0
    assert loss.item() > rmse.item()
    assert torch.allclose(loss, rmse + weight * direction_error)


def test_directional_loss_uses_neighbor_as_first_atan2_argument():
    pred = torch.full((1, 1, 2, 2), -2.0, dtype=torch.float64)
    target = torch.tensor([[[[-2.0, -2.0], [-2.0, 1.0]]]], dtype=torch.float64)
    weight = 0.25

    expected_error = _direction_error(pred, target)
    reversed_error = _direction_error_with_reversed_atan2_arguments(pred, target)
    rmse = _rmse(pred, target)
    actual_loss = XiongDirectionalLoss(weight=weight, eps=EPS)(pred, target)
    actual_error = (actual_loss - rmse) / weight

    assert not torch.allclose(expected_error, reversed_error)
    assert torch.allclose(actual_error, expected_error, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
def test_backward_produces_finite_gradients(loss_class):
    torch.manual_seed(7)
    pred = (torch.rand((2, 1, 5, 6), dtype=torch.float64) + 1.0).requires_grad_(True)
    target = torch.rand((2, 1, 5, 6), dtype=torch.float64) + 1.0

    loss = loss_class()(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_preserves_dtype_and_finite_gradients(loss_class, dtype):
    pred = torch.zeros((1, 1, 2, 2), dtype=dtype, requires_grad=True)
    target = torch.zeros_like(pred)

    loss = loss_class(eps=EPS)(pred, target)
    loss.backward()

    assert loss.dtype == dtype
    assert loss.item() == pytest.approx(math.sqrt(EPS), abs=2.0e-6)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_directional_backward_is_finite_for_zero_neighbor_pairs():
    pred = torch.zeros((1, 1, 3, 3), requires_grad=True)
    target = torch.zeros_like(pred)

    XiongDirectionalLoss()(pred, target).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
def test_representative_grid_shape(loss_class):
    torch.manual_seed(11)
    pred = torch.rand((2, 1, 150, 180))
    target = torch.rand_like(pred)

    loss = loss_class()(pred, target)

    assert loss.ndim == 0
    assert loss.dtype == pred.dtype
    assert loss.device == pred.device
    assert torch.isfinite(loss)


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
@pytest.mark.parametrize(
    ("pred", "target", "message"),
    [
        (torch.zeros(1, 1, 2, 2), torch.zeros(2, 1, 2, 2), "same shape"),
        (torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), "four-dimensional"),
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2), "one temperature channel"),
        (torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), "H >= 2 and W >= 2"),
        (torch.zeros(1, 1, 2, 1), torch.zeros(1, 1, 2, 1), "H >= 2 and W >= 2"),
    ],
)
def test_invalid_shapes_raise_clear_errors(loss_class, pred, target, message):
    with pytest.raises(ValueError, match=message):
        loss_class()(pred, target)


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
def test_non_finite_fields_are_rejected(loss_class):
    pred = torch.zeros((1, 1, 2, 2))
    target = torch.zeros_like(pred)
    pred[..., 0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or infinite"):
        loss_class()(pred, target)


@pytest.mark.parametrize("loss_class", [XiongContinuityLoss, XiongDirectionalLoss])
def test_invalid_loss_hyperparameters_are_rejected(loss_class):
    with pytest.raises(ValueError, match=">= 0"):
        loss_class(weight=-1.0)
    with pytest.raises(ValueError, match="> 0"):
        loss_class(eps=0.0)
    with pytest.raises(ValueError, match="finite"):
        loss_class(weight=float("inf"))


@pytest.mark.parametrize("loss_type", ["xiong_continuity", "xiong_directional"])
def test_xiong_yaml_is_loaded_with_float_parameters(loss_type, tmp_path):
    cfg, data = _load_config_with_temp_results(XIONG_CONFIGS[loss_type], tmp_path)

    assert cfg.loss_type == loss_type
    assert cfg.variable == "temp"
    assert cfg.model_type == "unet1"
    assert isinstance(cfg.xiong_continuity_weight, float)
    assert isinstance(cfg.xiong_directional_weight, float)
    assert isinstance(cfg.xiong_eps, float)
    assert cfg.xiong_continuity_weight == pytest.approx(1.0e-4)
    assert cfg.xiong_directional_weight == pytest.approx(1.0e-4)
    assert cfg.xiong_eps == pytest.approx(1.0e-8)
    assert "LR_scheduler" not in data["training"]
    assert data["training"]["scheduler"] == {"enable": False, "type": "cosine"}


def test_old_config_gets_backward_compatible_xiong_defaults(tmp_path):
    cfg, _ = _load_config_with_temp_results(CONFIG_DIR / "config_arch1.yaml", tmp_path)

    assert cfg.loss_type == "gaussian"
    assert cfg.xiong_continuity_weight == pytest.approx(1.0e-4)
    assert cfg.xiong_directional_weight == pytest.approx(1.0e-4)
    assert cfg.xiong_eps == pytest.approx(1.0e-8)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("continuity_weight", -1.0, "continuity_weight must be >= 0"),
        ("directional_weight", -1.0, "directional_weight must be >= 0"),
        ("eps", 0.0, "eps must be > 0"),
    ],
)
def test_config_rejects_invalid_xiong_parameters(key, value, message, tmp_path):
    data = _yaml_dict(XIONG_CONFIGS["xiong_continuity"])
    data["paths"]["results_dir"] = str(tmp_path / "results")
    data["training"]["xiong"][key] = value

    with pytest.raises(ValueError, match=message):
        Config(data)


@pytest.mark.parametrize("loss_type", ["xiong_continuity", "xiong_directional"])
def test_config_rejects_xiong_loss_for_precipitation(loss_type, tmp_path):
    data = _yaml_dict(XIONG_CONFIGS[loss_type])
    data["paths"]["results_dir"] = str(tmp_path / "results")
    data["general"]["variable"] = "precip"

    with pytest.raises(ValueError, match="only supported.*temp"):
        Config(data)


class _FakeUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, use_gaussian, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gaussian = use_gaussian

    def forward(self, x, target_size):
        return x.new_zeros((x.shape[0], self.out_channels, *target_size))


@pytest.mark.parametrize(
    ("loss_type", "expected_channels"),
    [
        ("mse", 1),
        ("xiong_continuity", 1),
        ("xiong_directional", 1),
        ("gaussian", 2),
        ("bernoulli_gamma", 3),
    ],
)
@pytest.mark.parametrize("builder_module", ["training", "evaluation"])
def test_train_and_eval_builders_select_expected_channels(
    loss_type, expected_channels, builder_module, monkeypatch
):
    import src.models.unet_arch1 as unet_arch1
    from src.core import evaluation, training

    monkeypatch.setattr(unet_arch1, "UNet1", _FakeUNet1)
    builder = training._build_model if builder_module == "training" else evaluation._build_model
    cfg = SimpleNamespace(
        loss_type=loss_type,
        model_type="unet1",
        group_norm_enable=True,
        group_norm_num_groups=8,
        dropout_enable=True,
        dropout_value=0.1,
    )
    inputs = torch.zeros((2, 20, 4, 5))
    targets = torch.zeros((2, 1, 8, 9))

    model = builder(cfg, inputs, targets)

    assert model.unet.out_channels == expected_channels
    assert model.unet.use_gaussian is (expected_channels == 2)


def test_real_unet1_deterministic_head_outputs_one_channel():
    from src.models.unet_arch1 import UNet1

    model = UNet1(
        in_channels=2,
        out_channels=1,
        base_filters=8,
        use_gaussian=False,
        norm_type="group",
        num_groups=4,
        dropout=0.0,
    ).eval()
    inputs = torch.rand((1, 2, 8, 8))

    with torch.no_grad():
        outputs = model(inputs, target_size=(15, 18))

    assert outputs.shape == (1, 1, 15, 18)


@pytest.mark.parametrize(
    ("loss_name", "loss_factory"),
    [
        ("mse", lambda: nn.MSELoss()),
        ("gaussian", GaussianLoss),
        ("bernoulli_gamma", BernoulliGammaLoss),
    ],
)
def test_existing_losses_still_produce_finite_gradients(loss_name, loss_factory):
    torch.manual_seed(3)
    target = torch.rand((2, 1, 3, 4)) + 0.1
    if loss_name == "mse":
        pred = torch.rand((2, 1, 3, 4), requires_grad=True)
    elif loss_name == "gaussian":
        pred = torch.rand((2, 2, 3, 4), requires_grad=True)
    else:
        pred = torch.rand((2, 3, 3, 4), requires_grad=True)

    loss = loss_factory()(pred, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


class _TinyDownscaler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.output = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.output(x)


@pytest.mark.parametrize("loss_type", ["xiong_continuity", "xiong_directional"])
def test_one_epoch_train_save_reload_and_deterministic_netcdf(
    loss_type, tmp_path, monkeypatch
):
    from src.core import evaluation, training
    from src.core.utils import save_model

    cfg, _ = _load_config_with_temp_results(XIONG_CONFIGS[loss_type], tmp_path)
    cfg.device = torch.device("cpu")
    cfg.epochs = 1
    cfg.batch_size = 2
    cfg.validation_enable = True
    cfg.validation_percentage = 0.5
    cfg.scheduler_enable = False
    cfg.early_stopping_enable = True

    torch.manual_seed(19)
    x_train = torch.rand((4, 2, 4, 5)) + 0.5
    y_train = torch.rand((4, 1, 4, 5)) + 0.5

    monkeypatch.setattr(
        training,
        "_build_model",
        lambda cfg, x, y: _TinyDownscaler(x.shape[1]),
    )
    last_model, best_model, train_losses, val_losses, best_loss, best_epoch = (
        training.train_model(cfg, x_train, y_train)
    )

    assert len(train_losses) == 1
    assert len(val_losses) == 1
    assert math.isfinite(train_losses[0])
    assert math.isfinite(val_losses[0])
    assert math.isfinite(best_loss)
    assert best_epoch == 1

    last_path = save_model(
        cfg,
        last_model,
        train_losses=train_losses,
        val_losses=val_losses,
        tag="last",
        best_score=best_loss,
    )
    best_path = save_model(
        cfg,
        best_model,
        train_losses=train_losses,
        val_losses=val_losses,
        tag="best",
        best_score=best_loss,
    )
    assert Path(last_path).is_file()
    assert Path(best_path).is_file()

    monkeypatch.setattr(
        evaluation,
        "_build_model",
        lambda cfg, x, y: _TinyDownscaler(x.shape[1]),
    )
    monkeypatch.setattr(evaluation.use, "format_components_for_title", lambda **kwargs: "")
    monkeypatch.setattr(evaluation.use, "plot_losses", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation.use, "spatial_comparaison_plot", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        evaluation.use, "monthly_temp_comparaison_plot", lambda *args, **kwargs: None
    )

    x_test = torch.rand((2, 2, 4, 5)) + 0.5
    y_test = torch.rand((2, 1, 4, 5)) + 0.5
    lon = np.linspace(-18.0, -17.0, 5)
    lat = np.linspace(21.0, 22.0, 4)
    times = np.array(["2006-01-01", "2006-01-02"], dtype="datetime64[D]")

    evaluation.evaluate_and_save(cfg, x_test, y_test, lon, lat, times)

    output_path = (
        Path(cfg.results_dir)
        / cfg.experiment
        / "output_data"
        / "unet1_predictions_mswt.nc"
    )
    assert output_path.is_file()
    with xr.open_dataset(output_path) as dataset:
        assert set(dataset.data_vars) == {"air_temperature"}
        assert dataset["air_temperature"].shape == (2, 4, 5)
