import pytest
from sklearn.pipeline import Pipeline
from src.pipeline.factory import FeaturePipelineFactory


def test_factory_parses_yaml():
    yaml_config = """
    steps:
      - name: "fill_na"
        class: "src.features.impute.TimeSeriesImputer"
        params:
          method: "constant"
          value: -1
    """
    factory = FeaturePipelineFactory()
    pipeline = factory.create_from_yaml(yaml_config)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[0][0] == "fill_na"
    assert pipeline.steps[0][1].value == -1
