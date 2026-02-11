import yaml
import importlib
from typing import Dict, Any, List
from sklearn.pipeline import Pipeline
from src.utils.logging import get_logger


class ProcessingPipelineFactory:
    """
    Creates a data processing pipeline from a configuration dictionary.
    """

    def __init__(self):
        self.logger = get_logger("PipelineFactory")

    def create_from_yaml(self, yaml_path: str) -> Pipeline:
        self.logger.info(f"Loading pipeline config from {yaml_path}")
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return self.create_from_config(config)

    def create_from_config(self, config: Dict[str, Any]) -> Pipeline:
        steps = []
        pipeline_config = config.get("pipeline_steps", [])

        for step_conf in pipeline_config:
            name = step_conf["name"]
            cls_path = step_conf["class"]
            params = step_conf.get("params", {})

            self.logger.info(f"Adding step: {name} ({cls_path})")

            # Dynamic import
            module_name, class_name = cls_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

            instance = cls(**params)
            steps.append((name, instance))

        return Pipeline(steps)
