import pytest
from src.pipeline.config import load_config
import yaml
import os


def test_load_config(tmp_path):
    # Setup
    config_data = {"run_name": "test", "model": {"params": {"a": 1}}}
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Execute
    loaded = load_config(str(config_file))

    # Verify
    assert loaded["run_name"] == "test"
    assert loaded["model"]["params"]["a"] == 1


def test_load_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")
