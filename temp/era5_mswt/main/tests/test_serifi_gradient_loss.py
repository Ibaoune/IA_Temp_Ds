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
from src.core.losses import SerifiGradientLoss


MAIN_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = MAIN_DIR / "configs"
SERIFI_CONFIG = (
    CONFIG_DIR
    / "phy_ai"
    / "unet"
    / "config_arch1_serifi_gradient.yaml"
)
OLD_CONFIGS = [
    ("mse", CONFIG_DIR / "unet" / "config_unet1_mse.yaml"),
    ("gaussian", CONFIG_DIR / "unet" / "config_arch1.yaml"),
    (
        "xiong_continuity",
        CONFIG_DIR
        / "phy_ai"
        / "unet"
        / "config_arch1_xiong_continuity.yaml",
    ),
    (
        "xiong_directional",
        CONFIG_DIR
        / "phy_ai"
        / "unet"
        / "config_arch1_xiong_directional.yaml",
    ),
]


def _serifi_components(prediction, target):
    data_loss = torch.mean(torch.abs(prediction - target))

    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    true_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    gradient_x_loss = torch.mean(torch.abs(pred_dx - true_dx))

    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    true_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    gradient_y_loss = torch.mean(torch.abs(pred_dy - true_dy))

    return data_loss, gradient_x_loss, gradient_y_loss


def _yaml_dict(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_config_with_temp_results(path, tmp_path):
    data = _yaml_dict(path)
    data["paths"]["results_dir"] = str(tmp_path / "results")
    copied_path = tmp_path / path.name
    copied_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(path=str(copied_path)), data


def test_identical_fields_have_exactly_zero_loss():
    target = torch.tensor(
        [[[[1.0, 2.0, 4.0], [3.0, 5.0, 8.0], [7.0, 9.0, 10.0]]]]
    )
    prediction = target.clone()

    data_loss, gradient_x_loss, gradient_y_loss = _serifi_components(
        prediction, target
    )
    loss = SerifiGradientLoss()(prediction, target)

    assert data_loss.item() == 0.0
    assert gradient_x_loss.item() == 0.0
    assert gradient_y_loss.item() == 0.0
    assert loss.item() == 0.0


def test_zero_weight_is_exactly_the_l1_data_loss():
    prediction = torch.tensor(
        [[[[1.0, 3.0], [2.0, 5.0]]]],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[[[2.0, 2.0], [4.0, 1.0]]]],
        dtype=torch.float64,
    )

    actual = SerifiGradientLoss(gradient_weight=0.0)(prediction, target)
    expected = torch.mean(torch.abs(prediction - target))

    assert torch.allclose(actual, expected, rtol=1.0e-12, atol=0.0)


def test_constant_offset_has_no_spatial_gradient_penalty():
    target = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 6.0, 9.0]]]],
        dtype=torch.float64,
    )
    prediction = target + 2.5

    data_loss, gradient_x_loss, gradient_y_loss = _serifi_components(
        prediction, target
    )
    actual = SerifiGradientLoss(gradient_weight=7.0)(prediction, target)

    assert gradient_x_loss.item() == 0.0
    assert gradient_y_loss.item() == 0.0
    assert torch.allclose(actual, data_loss, rtol=1.0e-12, atol=0.0)


def test_exact_horizontal_gradient_case_has_loss_one_point_five():
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    prediction = torch.tensor(
        [[[[0.0, 1.0], [0.0, 1.0]]]],
        dtype=torch.float64,
    )

    data_loss, gradient_x_loss, gradient_y_loss = _serifi_components(
        prediction, target
    )
    actual = SerifiGradientLoss()(prediction, target)

    assert data_loss.item() == pytest.approx(0.5)
    assert gradient_x_loss.item() == pytest.approx(1.0)
    assert gradient_y_loss.item() == pytest.approx(0.0)
    assert actual.item() == pytest.approx(1.5)


def test_exact_vertical_gradient_case_has_loss_one_point_five():
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    prediction = torch.tensor(
        [[[[0.0, 0.0], [1.0, 1.0]]]],
        dtype=torch.float64,
    )

    data_loss, gradient_x_loss, gradient_y_loss = _serifi_components(
        prediction, target
    )
    actual = SerifiGradientLoss()(prediction, target)

    assert data_loss.item() == pytest.approx(0.5)
    assert gradient_x_loss.item() == pytest.approx(0.0)
    assert gradient_y_loss.item() == pytest.approx(1.0)
    assert actual.item() == pytest.approx(1.5)


def test_horizontal_and_vertical_gradient_losses_are_added_independently():
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float64)
    prediction = torch.tensor(
        [[[[0.0, 1.0], [1.0, 2.0]]]],
        dtype=torch.float64,
    )

    data_loss, gradient_x_loss, gradient_y_loss = _serifi_components(
        prediction, target
    )
    actual = SerifiGradientLoss()(prediction, target)

    assert data_loss.item() == pytest.approx(1.0)
    assert gradient_x_loss.item() == pytest.approx(1.0)
    assert gradient_y_loss.item() == pytest.approx(1.0)
    assert actual.item() == pytest.approx(3.0)


def test_backward_produces_finite_gradients():
    torch.manual_seed(23)
    prediction = torch.rand(
        (2, 1, 5, 6),
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.rand((2, 1, 5, 6), dtype=torch.float64)

    loss = SerifiGradientLoss(gradient_weight=1.5)(prediction, target)
    loss.backward()

    assert loss.ndim == 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_representative_spatial_dimensions():
    torch.manual_seed(29)
    prediction = torch.rand((2, 1, 150, 180), requires_grad=True)
    target = torch.rand_like(prediction)

    loss = SerifiGradientLoss()(prediction, target)
    loss.backward()

    assert loss.ndim == 0
    assert loss.dtype == prediction.dtype
    assert loss.device == prediction.device
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize(
    ("prediction", "target", "message"),
    [
        (
            torch.zeros(1, 1, 2, 2),
            torch.zeros(2, 1, 2, 2),
            "same shape",
        ),
        (
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            "four-dimensional",
        ),
        (
            torch.zeros(1, 2, 2, 2),
            torch.zeros(1, 2, 2, 2),
            "one scalar-field channel",
        ),
        (
            torch.zeros(0, 1, 2, 2),
            torch.zeros(0, 1, 2, 2),
            "non-empty batch",
        ),
        (
            torch.zeros(1, 1, 1, 2),
            torch.zeros(1, 1, 1, 2),
            "H >= 2 and W >= 2",
        ),
        (
            torch.zeros(1, 1, 2, 1),
            torch.zeros(1, 1, 2, 1),
            "H >= 2 and W >= 2",
        ),
    ],
)
def test_invalid_shapes_raise_clear_errors(prediction, target, message):
    with pytest.raises(ValueError, match=message):
        SerifiGradientLoss()(prediction, target)


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        ([[[[0.0, 0.0], [0.0, 0.0]]]], torch.zeros(1, 1, 2, 2)),
        (torch.zeros(1, 1, 2, 2), [[[[0.0, 0.0], [0.0, 0.0]]]]),
    ],
)
def test_non_tensor_fields_are_rejected(prediction, target):
    with pytest.raises(TypeError, match="PyTorch tensors"):
        SerifiGradientLoss()(prediction, target)


def test_non_floating_fields_are_rejected():
    prediction = torch.zeros((1, 1, 2, 2), dtype=torch.int64)
    target = torch.zeros_like(prediction)

    with pytest.raises(TypeError, match="floating-point"):
        SerifiGradientLoss()(prediction, target)


def test_dtype_mismatch_is_rejected():
    prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    target = torch.zeros((1, 1, 2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="same dtype"):
        SerifiGradientLoss()(prediction, target)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_supported_dtypes_preserve_dtype_and_finite_gradients(dtype):
    prediction = torch.tensor(
        [[[[0.0, 1.0], [1.0, 3.0]]]],
        dtype=dtype,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[[0.5, 0.5], [1.5, 2.0]]]],
        dtype=dtype,
    )

    try:
        loss = SerifiGradientLoss()(prediction, target)
        loss.backward()
    except RuntimeError as exc:
        if dtype in {torch.float16, torch.bfloat16}:
            pytest.skip(f"{dtype} is not supported on this PyTorch backend: {exc}")
        raise

    assert loss.dtype == dtype
    assert loss.device == prediction.device
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert prediction.grad.dtype == dtype
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_forward_and_backward_preserve_device():
    device = torch.device("cuda")
    prediction = torch.rand(
        (2, 1, 4, 5),
        device=device,
        requires_grad=True,
    )
    target = torch.rand_like(prediction)

    loss = SerifiGradientLoss()(prediction, target)
    loss.backward()

    assert loss.device == device
    assert prediction.grad is not None
    assert prediction.grad.device == device
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_device_mismatch_is_rejected():
    prediction = torch.zeros((1, 1, 2, 2), device="cuda")
    target = torch.zeros((1, 1, 2, 2), device="cpu")

    with pytest.raises(ValueError, match="same device"):
        SerifiGradientLoss()(prediction, target)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field_name", ["prediction", "target"])
def test_non_finite_fields_are_rejected(bad_value, field_name):
    prediction = torch.zeros((1, 1, 2, 2))
    target = torch.zeros_like(prediction)
    field = prediction if field_name == "prediction" else target
    field[..., 0, 0] = bad_value

    with pytest.raises(ValueError, match=f"{field_name} contains"):
        SerifiGradientLoss()(prediction, target)


@pytest.mark.parametrize(
    "gradient_weight",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_or_negative_gradient_weights_are_rejected(gradient_weight):
    with pytest.raises(ValueError, match="finite and >= 0"):
        SerifiGradientLoss(gradient_weight=gradient_weight)


@pytest.mark.parametrize("gradient_weight", [None, "not-a-number", [1.0]])
def test_non_numeric_gradient_weights_are_rejected(gradient_weight):
    with pytest.raises(ValueError, match="real number"):
        SerifiGradientLoss(gradient_weight=gradient_weight)


def test_serifi_yaml_loads_exact_experiment_parameters(tmp_path):
    cfg, data = _load_config_with_temp_results(SERIFI_CONFIG, tmp_path)

    assert cfg.model_type == "unet1"
    assert cfg.loss_type == "serifi_gradient"
    assert cfg.variable == "temp"
    assert cfg.experiment == "unet1_temperature_serifi_gradient"
    assert isinstance(cfg.serifi_gradient_weight, float)
    assert cfg.serifi_gradient_weight == pytest.approx(1.0)
    assert data["training"]["serifi"] == {"gradient_weight": 1.0}
    assert "xiong" not in data["training"]
    assert "LR_scheduler" not in data["training"]
    assert data["training"]["scheduler"] == {"enable": False, "type": "cosine"}


@pytest.mark.parametrize(("expected_loss_type", "path"), OLD_CONFIGS)
def test_old_configs_get_backward_compatible_serifi_default(
    expected_loss_type, path, tmp_path
):
    cfg, _ = _load_config_with_temp_results(path, tmp_path)

    assert cfg.loss_type == expected_loss_type
    assert isinstance(cfg.serifi_gradient_weight, float)
    assert cfg.serifi_gradient_weight == pytest.approx(1.0)


@pytest.mark.parametrize(
    "gradient_weight",
    [-1.0, float("nan"), float("inf"), "not-a-number"],
)
def test_config_rejects_invalid_serifi_gradient_weight(
    gradient_weight, tmp_path
):
    data = _yaml_dict(SERIFI_CONFIG)
    data["paths"]["results_dir"] = str(tmp_path / "results")
    data["training"]["serifi"]["gradient_weight"] = gradient_weight

    with pytest.raises(
        ValueError,
        match="training.serifi.gradient_weight",
    ):
        Config(data)


def test_config_rejects_non_mapping_serifi_section(tmp_path):
    data = _yaml_dict(SERIFI_CONFIG)
    data["paths"]["results_dir"] = str(tmp_path / "results")
    data["training"]["serifi"] = []

    with pytest.raises(ValueError, match="training.serifi must be a YAML mapping"):
        Config(data)


def test_serifi_configuration_is_not_restricted_to_temperature(tmp_path):
    data = _yaml_dict(SERIFI_CONFIG)
    data["paths"]["results_dir"] = str(tmp_path / "results")
    data["general"]["variable"] = "precip"
    data["general"]["target"] = "mswep"
    data["paths"]["mswep_path"] = "synthetic_mswep.nc"

    cfg = Config(data)

    assert cfg.variable == "precip"
    assert cfg.loss_type == "serifi_gradient"
    assert cfg.serifi_gradient_weight == pytest.approx(1.0)


class _FakeUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, use_gaussian, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_gaussian = use_gaussian

    def forward(self, x, target_size):
        return x.new_zeros((x.shape[0], self.out_channels, *target_size))


@pytest.mark.parametrize("builder_module", ["training", "evaluation"])
def test_train_and_eval_builders_select_one_output_channel(
    builder_module, monkeypatch
):
    import src.models.unet_arch1 as unet_arch1
    from src.core import evaluation, training

    monkeypatch.setattr(unet_arch1, "UNet1", _FakeUNet1)
    builder = (
        training._build_model
        if builder_module == "training"
        else evaluation._build_model
    )
    cfg = SimpleNamespace(
        loss_type="serifi_gradient",
        model_type="unet1",
        group_norm_enable=True,
        group_norm_num_groups=8,
        dropout_enable=True,
        dropout_value=0.1,
    )
    inputs = torch.zeros((2, 20, 4, 5))
    targets = torch.zeros((2, 1, 8, 9))

    model = builder(cfg, inputs, targets)

    assert model.unet.out_channels == 1
    assert model.unet.use_gaussian is False
    assert model(inputs).shape == (2, 1, 8, 9)


def test_real_unet1_forward_is_compatible_with_serifi_loss():
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
    target = torch.rand((1, 1, 15, 18))

    with torch.no_grad():
        prediction = model(inputs, target_size=target.shape[-2:])
        loss = SerifiGradientLoss()(prediction, target)

    assert prediction.shape == target.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)


class _TinyDownscaler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.output = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.output(x)


def test_one_epoch_train_validate_save_reload_and_deterministic_netcdf(
    tmp_path, monkeypatch
):
    from src.core import evaluation, losses, training
    from src.core.utils import load_model, save_model

    cfg, _ = _load_config_with_temp_results(SERIFI_CONFIG, tmp_path)
    cfg.device = torch.device("cpu")
    cfg.epochs = 1
    cfg.batch_size = 2
    cfg.validation_enable = True
    cfg.validation_percentage = 0.5
    cfg.scheduler_enable = False
    cfg.early_stopping_enable = True

    criterion_calls = []
    original_loss_class = losses.SerifiGradientLoss

    class _TrackingSerifiGradientLoss(original_loss_class):
        def forward(self, prediction, target):
            criterion_calls.append(
                (
                    torch.is_grad_enabled(),
                    tuple(prediction.shape),
                    tuple(target.shape),
                )
            )
            return super().forward(prediction, target)

    monkeypatch.setattr(
        losses,
        "SerifiGradientLoss",
        _TrackingSerifiGradientLoss,
    )
    monkeypatch.setattr(
        training,
        "_build_model",
        lambda cfg, x, y: _TinyDownscaler(x.shape[1]),
    )

    torch.manual_seed(31)
    x_train = torch.rand((4, 2, 4, 5))
    y_train = torch.rand((4, 1, 4, 5))

    last_model, best_model, train_losses, val_losses, best_loss, best_epoch = (
        training.train_model(cfg, x_train, y_train)
    )

    assert criterion_calls == [
        (True, (2, 1, 4, 5), (2, 1, 4, 5)),
        (False, (2, 1, 4, 5), (2, 1, 4, 5)),
    ]
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

    reloaded_model, reloaded_train_losses, reloaded_val_losses = load_model(
        cfg,
        _TinyDownscaler(x_train.shape[1]),
        tag="best",
    )
    assert reloaded_train_losses == train_losses
    assert reloaded_val_losses == val_losses
    for name, expected in best_model.state_dict().items():
        assert torch.equal(reloaded_model.state_dict()[name], expected.cpu())

    monkeypatch.setattr(
        evaluation,
        "_build_model",
        lambda cfg, x, y: _TinyDownscaler(x.shape[1]),
    )
    monkeypatch.setattr(
        evaluation.use,
        "format_components_for_title",
        lambda **kwargs: "",
    )
    monkeypatch.setattr(
        evaluation.use,
        "plot_losses",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluation.use,
        "spatial_comparaison_plot",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        evaluation.use,
        "monthly_temp_comparaison_plot",
        lambda *args, **kwargs: None,
    )

    x_test = torch.rand((2, 2, 4, 5))
    y_test = torch.rand((2, 1, 4, 5))
    lon = np.linspace(-18.0, -17.0, 5)
    lat = np.linspace(21.0, 22.0, 4)
    times = np.array(
        ["2006-01-01", "2006-01-02"],
        dtype="datetime64[D]",
    )

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
        assert "log_variance" not in dataset
        assert dataset["air_temperature"].shape == (2, 4, 5)
