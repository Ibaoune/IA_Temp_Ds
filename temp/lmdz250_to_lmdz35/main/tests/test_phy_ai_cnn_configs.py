import copy
from pathlib import Path

import pytest
import torch
import yaml

from src.core.config import load_config
from src.core.losses import (
    SerifiGradientLoss,
    XiongContinuityLoss,
    XiongDirectionalLoss,
)


MAIN_DIR = Path(__file__).resolve().parents[1]
CNN_CONFIG_DIR = MAIN_DIR / "configs" / "cnn"
PHY_AI_CNN_CONFIG_DIR = MAIN_DIR / "configs" / "phy_ai" / "cnn"

FULL_CASES = [
    (
        "cnn1",
        "xiong_continuity",
        "cnn1_temperature_xiong_continuity",
        "config_cnn1_xiong_continuity.yaml",
    ),
    (
        "cnn1",
        "xiong_directional",
        "cnn1_temperature_xiong_directional",
        "config_cnn1_xiong_directional.yaml",
    ),
    (
        "cnn1",
        "serifi_gradient",
        "cnn1_temperature_serifi_gradient",
        "config_cnn1_serifi_gradient.yaml",
    ),
    (
        "cnn10",
        "xiong_continuity",
        "cnn10_temperature_xiong_continuity",
        "config_cnn10_xiong_continuity.yaml",
    ),
    (
        "cnn10",
        "xiong_directional",
        "cnn10_temperature_xiong_directional",
        "config_cnn10_xiong_directional.yaml",
    ),
    (
        "cnn10",
        "serifi_gradient",
        "cnn10_temperature_serifi_gradient",
        "config_cnn10_serifi_gradient.yaml",
    ),
]

TEST_CASES = [
    (
        model_type,
        loss_type,
        f"{experiment}_test",
        filename.replace(".yaml", "_test.yaml"),
    )
    for model_type, loss_type, experiment, filename in FULL_CASES
]

ALL_CASES = FULL_CASES + TEST_CASES


def _yaml_dict(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_config_with_temp_results(filename, tmp_path):
    data = _yaml_dict(PHY_AI_CNN_CONFIG_DIR / filename)
    data["paths"]["results_dir"] = str(tmp_path / "results")
    copied_path = tmp_path / filename
    copied_path.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(path=str(copied_path)), data


@pytest.mark.parametrize(
    ("model_type", "loss_type", "experiment", "filename"),
    FULL_CASES,
)
def test_phy_ai_cnn_yaml_matches_its_classical_base(
    model_type, loss_type, experiment, filename
):
    actual = _yaml_dict(PHY_AI_CNN_CONFIG_DIR / filename)
    expected = copy.deepcopy(
        _yaml_dict(CNN_CONFIG_DIR / f"config_{model_type}.yaml")
    )

    expected["general"]["experiment"] = experiment
    expected["training"].pop("LR_scheduler", None)
    expected["training"]["loss_type"] = loss_type

    if loss_type.startswith("xiong_"):
        expected["training"]["xiong"] = {
            "continuity_weight": 1.0e-4,
            "directional_weight": 1.0e-4,
            "eps": 1.0e-8,
        }
    else:
        expected["training"]["serifi"] = {"gradient_weight": 1.0}

    assert actual == expected


@pytest.mark.parametrize(
    ("model_type", "loss_type", "experiment", "filename"),
    TEST_CASES,
)
def test_phy_ai_cnn_test_yaml_matches_reduced_full_config(
    model_type, loss_type, experiment, filename
):
    actual = _yaml_dict(PHY_AI_CNN_CONFIG_DIR / filename)
    full_filename = filename.replace("_test.yaml", ".yaml")
    expected = copy.deepcopy(
        _yaml_dict(PHY_AI_CNN_CONFIG_DIR / full_filename)
    )

    expected["general"]["experiment"] = experiment
    expected["training"]["epochs"] = 20
    expected["training"]["batch_size"] = 4
    expected["training"]["validation"]["enable"] = False
    expected["dates"] = {
        "train": {"start": "1980-01-01", "end": "1984-12-31"},
        "test": {"start": "1985-01-01", "end": "1985-12-31"},
    }
    expected["paths"]["results_dir"] = "./temp/results_test/"

    assert actual == expected


@pytest.mark.parametrize(
    ("model_type", "loss_type", "experiment", "filename"),
    ALL_CASES,
)
def test_phy_ai_cnn_yaml_loads_expected_model_and_loss(
    model_type, loss_type, experiment, filename, tmp_path
):
    cfg, data = _load_config_with_temp_results(filename, tmp_path)

    assert cfg.model_type == model_type
    assert cfg.cnn_mode == model_type
    assert data["cnn"]["mode"] == model_type
    assert cfg.loss_type == loss_type
    assert cfg.experiment == experiment
    assert cfg.variable == "temp"
    assert "LR_scheduler" not in data["training"]
    assert data["training"]["scheduler"] == {
        "enable": False,
        "type": "cosine",
    }

    if loss_type.startswith("xiong_"):
        assert cfg.xiong_continuity_weight == pytest.approx(1.0e-4)
        assert cfg.xiong_directional_weight == pytest.approx(1.0e-4)
        assert cfg.xiong_eps == pytest.approx(1.0e-8)
        assert "serifi" not in data["training"]
    else:
        assert cfg.serifi_gradient_weight == pytest.approx(1.0)
        assert "xiong" not in data["training"]


def _criterion(cfg):
    if cfg.loss_type == "xiong_continuity":
        return XiongContinuityLoss(
            weight=cfg.xiong_continuity_weight,
            eps=cfg.xiong_eps,
        )
    if cfg.loss_type == "xiong_directional":
        return XiongDirectionalLoss(
            weight=cfg.xiong_directional_weight,
            eps=cfg.xiong_eps,
        )
    return SerifiGradientLoss(
        gradient_weight=cfg.serifi_gradient_weight,
    )


@pytest.mark.parametrize(
    ("model_type", "loss_type", "experiment", "filename"),
    FULL_CASES,
)
def test_cnn_train_and_eval_builders_use_one_channel_and_finite_gradients(
    model_type, loss_type, experiment, filename, tmp_path
):
    from src.core import evaluation, training

    cfg, _ = _load_config_with_temp_results(filename, tmp_path)
    inputs = torch.rand((2, 2, 3, 4))
    target = torch.rand((2, 1, 4, 5)) + 1.0

    train_model = training._build_model(cfg, inputs, target)
    prediction = train_model(inputs)
    loss = _criterion(cfg)(prediction, target)
    loss.backward()

    assert prediction.shape == target.shape
    assert train_model.out_channels == 1
    assert train_model.mode == model_type
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in train_model.parameters()
    )

    eval_model = evaluation._build_model(cfg, inputs, target)
    with torch.no_grad():
        eval_prediction = eval_model(inputs)

    assert eval_prediction.shape == target.shape
    assert eval_model.out_channels == 1
    assert eval_model.mode == model_type
    assert torch.isfinite(eval_prediction).all()
