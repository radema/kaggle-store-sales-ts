import yaml
import importlib
from typing import Dict, Any
from sklearn.pipeline import Pipeline


class FeaturePipelineFactory:
    def create_from_yaml(self, yaml_content: str) -> Pipeline:
        config = yaml.safe_load(yaml_content)
        return self.create_from_config(config)

    def create_from_config(self, config: Dict[str, Any]) -> Pipeline:
        steps = []
        for step_conf in config.get("steps", []):
            name = step_conf["name"]
            cls_path = step_conf["class"]
            params = step_conf.get("params", {})

            # Dynamic import
            module_name, class_name = cls_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

            instance = cls(**params)
            steps.append((name, instance))

        return Pipeline(steps)
