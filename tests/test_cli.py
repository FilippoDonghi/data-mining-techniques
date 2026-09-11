"""Tests for the dependency-light demo and actionable full-pipeline errors."""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from data_mining_techniques.cli import main
from data_mining_techniques.core import (
    FEATURE_COLUMNS,
    DataSchemaError,
    build_target_maps,
    make_features,
)
from data_mining_techniques.demo import run_demo, synthetic_training_data
from data_mining_techniques.pipeline import PipelineError, load_training, write_submission


def test_demo_runs_without_importing_lightgbm() -> None:
    sys.modules.pop("lightgbm", None)

    result = run_demo()

    assert result["rows"] == 12
    assert result["searches"] == 4
    assert result["feature_count"] == len(FEATURE_COLUMNS)
    assert 0.0 <= result["ndcg5_popularity"] <= 1.0
    assert "lightgbm" not in sys.modules


def test_cli_can_write_demo_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "nested" / "demo.json"

    assert main(["demo", "--output", str(destination)]) == 0
    assert destination.is_file()
    assert '"mode": "synthetic-demo"' in destination.read_text(encoding="utf-8")
    assert "synthetic-demo" in capsys.readouterr().out


def test_missing_training_file_has_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(PipelineError, match="restricted Kaggle data is not included"):
        load_training(missing)


def test_invalid_training_schema_lists_missing_columns(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("srch_id,click_bool\n1,1\n", encoding="utf-8")

    with pytest.raises(DataSchemaError, match=r"training CSV is missing .* required column"):
        load_training(incomplete)


def test_cli_reports_missing_file_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(
            [
                "train",
                "--train-data",
                str(tmp_path / "train.csv"),
                "--test-data",
                str(tmp_path / "test.csv"),
            ]
        )

    assert raised.value.code == 2
    error_output = capsys.readouterr().err
    assert "file not found" in error_output
    assert "Traceback" not in error_output


def test_submission_uses_assignment_header_names(tmp_path: Path) -> None:
    raw = synthetic_training_data()
    features = make_features(raw, build_target_maps(raw))

    class DummyRanker:
        def predict(self, frame: object) -> np.ndarray:
            return np.arange(len(features), dtype=float)

    output = tmp_path / "submission.csv"
    write_submission(features, DummyRanker(), output)

    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        assert next(reader) == ["SearchId", "PropertyId"]
