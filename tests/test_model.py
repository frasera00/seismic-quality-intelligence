import pytest

from scientific_ml_template.model import build_model


def test_build_model_returns_expected_configuration(
    baseline_config: dict,
) -> None:
    model = build_model(baseline_config)

    assert model.n_estimators == 300
    assert model.max_depth is None
    assert model.random_state == 42
    assert model.n_jobs == -1


def test_build_model_rejects_unsupported_model(
    baseline_config: dict,
) -> None:
    baseline_config["model"]["name"] = "unsupported_model"

    with pytest.raises(ValueError, match="Unsupported model"):
        build_model(baseline_config)
