import yaml
from pathlib import Path
from src.data.loader import DataLoader
from src.pipeline.processing import ProcessingPipelineFactory
from src.utils.logging import get_logger


def run_preprocessing(config_path: str):
    """
    Main function to run the data preprocessing pipeline.
    """
    logger = get_logger("run_preprocessing")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    loader = DataLoader()
    df = loader.load_unified(config)
    
    # Create and run pipeline
    factory = ProcessingPipelineFactory()
    pipeline = factory.create_from_config(config)
    
    logger.info("Fitting and transforming data...")
    processed_df = pipeline.fit_transform(df)
    
    # Save output
    output_path = Path("data/processed/processed_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    run_preprocessing("configs/preprocessing.yaml")
