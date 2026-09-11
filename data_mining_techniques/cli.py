"""Command-line interface for the synthetic demo and full coursework pipeline."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .core import DataSchemaError
from .demo import run_demo
from .pipeline import PipelineError, TrainingConfig, run_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line contract."""

    parser = argparse.ArgumentParser(
        prog="data-mining-techniques",
        description=(
            "Run a dependency-light synthetic ranking demo or, with authorized data, "
            "the historical VU coursework pipeline."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser(
        "demo",
        help="run feature engineering and NDCG on bundled synthetic rows",
    )
    demo_parser.add_argument(
        "--output",
        type=Path,
        help="optionally write the demo result as JSON",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="run the historical model with authorized competition CSV files",
    )
    train_parser.add_argument(
        "--train-data",
        required=True,
        type=Path,
        help="path to the authorized labeled CSV",
    )
    train_parser.add_argument(
        "--test-data",
        required=True,
        type=Path,
        help="path to the authorized unlabeled CSV",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="generated metrics, figures, and submissions (default: artifacts)",
    )
    train_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="random seed for the historical query split and model (default: 42)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command and translate expected failures into CLI errors."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "demo":
            result = run_demo()
            rendered = json.dumps(result, indent=2)
            print(rendered)
            if arguments.output is not None:
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                arguments.output.write_text(rendered + "\n", encoding="utf-8")
            return 0

        config = TrainingConfig(
            training_data=arguments.train_data,
            test_data=arguments.test_data,
            output_directory=arguments.output_dir,
            random_state=arguments.random_state,
        )
        run_training_pipeline(config)
        return 0
    except (DataSchemaError, PipelineError, ValueError) as error:
        parser.error(str(error))
    return 2
