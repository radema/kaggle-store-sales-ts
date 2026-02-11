import yaml

import pandas as pd
from src.pipeline.processing import ProcessingPipelineFactory



def test_pipeline_factory_integration(tmp_path):
    # Create a minimal config
    config_path = tmp_path / "preprocessing.yaml"
    config = {
        "pipeline_steps": [
            {
                "name": "date_grid",
                "class": "src.features.alignment.DateGridTransformer",
                "params": {"groupby": ["store_nbr"], "fill_map": {"sales": 0.0}},
            }
        ]
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    factory = ProcessingPipelineFactory()
    pipeline = factory.create_from_yaml(str(config_path))

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-03"]),
            "store_nbr": [1, 1],
            "sales": [10.0, 30.0],
        }
    )

    pipeline.fit(df)
    result = pipeline.transform(df)
    assert len(result) == 3  # Gap filled
    assert result.loc[result["date"] == "2013-01-02", "sales"].values[0] == 0.0
