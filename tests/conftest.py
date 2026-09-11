import pytest

from scientific_ml_template.config import load_config


@pytest.fixture
def baseline_config() -> dict:
    return load_config("configs/baseline.yaml")
