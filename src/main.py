import argparse
import sys
from src.pipeline.runners.hybrid import HybridRunner


def main():
    parser = argparse.ArgumentParser(description="Store Sales Forecasting Runner")
    parser.add_argument(
        "--runner",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="Model runner strategy to use.",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run. If missing, used timestamped name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/runs",
        help="Directory to save artifacts.",
    )

    args = parser.parse_args()

    if args.runner == "hybrid":
        runner = HybridRunner(args.config, run_name=args.run_name)
    else:
        print(f"Error: Unknown runner '{args.runner}'")
        sys.exit(1)

    print(f"Starting {args.runner} runner...")
    runner.run(output_dir=args.output_dir)
    print("Run completed successfully.")


if __name__ == "__main__":
    main()
