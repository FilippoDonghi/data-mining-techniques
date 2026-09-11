"""Full historical coursework pipeline.

The private data and heavyweight ML/plotting dependencies are needed only when this module
is executed through the train command. The synthetic demo and core imports stay lightweight.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    FEATURE_COLUMNS,
    build_target_maps,
    compute_ndcg_at_k,
    family_group,
    group_sizes,
    make_features,
    relevance_label,
    split_by_search,
    test_columns,
    training_columns,
    validate_columns,
)

FAMILY_WEIGHT = 1.6

METHODOLOGY_LIMITATIONS = [
    (
        "The historical training features apply target maps to some of the same rows used "
        "to build those maps. A production study should use out-of-fold target encoding."
    ),
    (
        "The historical validation split is random by search, not temporal. It therefore "
        "does not measure future-data drift."
    ),
    (
        "The family/non-family weighting experiment is exploratory: the weight was not "
        "selected on a separate tuning split and subgroup uncertainty was not estimated."
    ),
    (
        "The local NDCG implementation skips searches with zero ideal DCG. That convention "
        "must be checked against any external scorer."
    ),
]


class PipelineError(RuntimeError):
    """Raised with an actionable message for an unavailable input or dependency."""


@dataclass(frozen=True)
class TrainingConfig:
    """Paths and reproducibility settings for the full coursework run."""

    training_data: Path
    test_data: Path
    output_directory: Path
    random_state: int = 42


def _read_csv_checked(
    path: Path,
    required_columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise PipelineError(
            f"{label} file not found: {path}. The restricted Kaggle data is not included; "
            "pass an authorized CSV path explicitly."
        )

    try:
        header = pd.read_csv(path, nrows=0)
    except (OSError, pd.errors.ParserError) as error:
        raise PipelineError(f"Could not read {label} header from {path}: {error}") from error

    validate_columns(header.columns, required_columns, source=f"{label} CSV")
    try:
        return pd.read_csv(path, usecols=required_columns)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise PipelineError(f"Could not load {label} data from {path}: {error}") from error


def load_training(path: Path) -> pd.DataFrame:
    """Load and validate the labeled coursework data."""

    print(f"Loading training set from {path}...")
    frame = _read_csv_checked(path, training_columns(), label="training")
    frame["relevance"] = relevance_label(frame)
    print(f"  loaded {len(frame):,} rows, {frame['srch_id'].nunique():,} searches")
    return frame


def load_test(path: Path) -> pd.DataFrame:
    """Load and validate the unlabeled coursework data."""

    print(f"Loading test set from {path}...")
    frame = _read_csv_checked(path, test_columns(), label="test")
    print(f"  loaded {len(frame):,} rows, {frame['srch_id'].nunique():,} searches")
    return frame


def _load_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except (ImportError, OSError) as error:
        detail = str(error).splitlines()[0]
        raise PipelineError(
            "LightGBM could not be loaded. Install requirements-full.txt. On macOS, "
            "install the OpenMP runtime first (for Homebrew: brew install libomp). "
            f"Original error: {detail}"
        ) from error
    return lgb


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise PipelineError(
            "Matplotlib is required for the full pipeline. Install requirements-full.txt."
        ) from error
    return plt


def make_ranker(*, n_estimators: int, random_state: int) -> Any:
    """Construct the historical LambdaMART model."""

    lgb = _load_lightgbm()
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=255,
        max_depth=-1,
        min_child_samples=100,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        label_gain=[0, 1, 0, 0, 0, 31],
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def fit_with_early_stopping(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    random_state: int,
    family_weight: float | None = None,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    """Fit one validation model and return frames in LightGBM group order."""

    lgb = _load_lightgbm()
    sorted_training = training.sort_values("srch_id", kind="stable").reset_index(drop=True)
    sorted_validation = validation.sort_values("srch_id", kind="stable").reset_index(drop=True)
    sample_weight = None
    if family_weight is not None:
        sample_weight = np.where(
            family_group(sorted_training) == "family",
            family_weight,
            1.0,
        )

    ranker = make_ranker(n_estimators=1_500, random_state=random_state)
    ranker.fit(
        sorted_training[FEATURE_COLUMNS],
        sorted_training["relevance"],
        group=group_sizes(sorted_training),
        eval_set=[(sorted_validation[FEATURE_COLUMNS], sorted_validation["relevance"])],
        eval_group=[group_sizes(sorted_validation)],
        eval_at=[5],
        sample_weight=sample_weight,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    return ranker, sorted_training, sorted_validation


def fit_full(
    frame: pd.DataFrame,
    *,
    n_estimators: int,
    random_state: int,
    family_weight: float | None = None,
) -> Any:
    """Fit a full-data historical model using a selected iteration count."""

    sorted_frame = frame.sort_values("srch_id", kind="stable").reset_index(drop=True)
    sample_weight = None
    if family_weight is not None:
        sample_weight = np.where(
            family_group(sorted_frame) == "family",
            family_weight,
            1.0,
        )
    ranker = make_ranker(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    ranker.fit(
        sorted_frame[FEATURE_COLUMNS],
        sorted_frame["relevance"],
        group=group_sizes(sorted_frame),
        sample_weight=sample_weight,
    )
    return ranker


def make_eda_plots(frame: pd.DataFrame, output_directory: Path) -> None:
    """Generate the two compact EDA figures retained in the report."""

    plt = _load_pyplot()
    missing_percentage = frame.isna().mean().mul(100).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    missing_percentage.plot(kind="bar", color="#2b8cbe")
    plt.title("Top 15 Features by Missingness (%)")
    plt.ylabel("Missing %")
    plt.tight_layout()
    plt.savefig(output_directory / "missingness_top15.png", dpi=150)
    plt.close()

    position_bias = (
        frame[["random_bool", "position", "booking_bool"]]
        .dropna()
        .groupby(["random_bool", "position"], as_index=False)["booking_bool"]
        .mean()
    )
    plt.figure(figsize=(8, 5))
    for random_value, label in [(0, "Normal sort"), (1, "Random sort")]:
        subset = position_bias[position_bias["random_bool"] == random_value]
        plt.plot(
            subset["position"],
            subset["booking_bool"],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    plt.title("Position Bias by Sort Mode")
    plt.xlabel("Display position")
    plt.ylabel("Mean booking rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "position_bias.png", dpi=150)
    plt.close()


def make_importance_plot(ranker: Any, output_directory: Path) -> pd.DataFrame:
    """Write a top-20 gain-importance chart and table."""

    plt = _load_pyplot()
    importance = (
        pd.DataFrame(
            {
                "feature": ranker.feature_name_,
                "gain": ranker.booster_.feature_importance(importance_type="gain"),
            }
        )
        .sort_values("gain", ascending=True)
        .tail(20)
    )
    plt.figure(figsize=(8, 8))
    plt.barh(importance["feature"], importance["gain"], color="#2b8cbe")
    plt.title("LightGBM Feature Importance (gain) — Top 20")
    plt.tight_layout()
    plt.savefig(output_directory / "feature_importance_top20.png", dpi=150)
    plt.close()
    importance.to_csv(output_directory / "feature_importance_top20.csv", index=False)
    return importance


def popularity_baseline_ndcg(validation: pd.DataFrame) -> float:
    """Score the historical equal-weight popularity baseline."""

    scored = validation.copy()
    scored["popularity_score"] = 0.5 * scored["dest_prop_score"] + 0.5 * scored["prop_ctr"]
    return compute_ndcg_at_k(scored, "popularity_score", k=5)


def _group_ndcg(frame: pd.DataFrame, score_column: str) -> tuple[float, float]:
    query_group = frame.groupby("srch_id", as_index=False)["group"].first()
    family_ids = set(query_group.loc[query_group["group"] == "family", "srch_id"])
    non_family_ids = set(query_group.loc[query_group["group"] == "non_family", "srch_id"])
    family = compute_ndcg_at_k(
        frame[frame["srch_id"].isin(family_ids)],
        score_column,
        k=5,
    )
    non_family = compute_ndcg_at_k(
        frame[frame["srch_id"].isin(non_family_ids)],
        score_column,
        k=5,
    )
    return family, non_family


def write_submission(
    features: pd.DataFrame,
    ranker: Any,
    output_path: Path,
) -> None:
    """Write the header format shown in the coursework assignment brief."""

    scored = features[["srch_id", "prop_id"]].copy()
    scored["score"] = ranker.predict(features[FEATURE_COLUMNS])
    submission = scored.sort_values(
        ["srch_id", "score"],
        ascending=[True, False],
        kind="stable",
    )
    submission = submission[["srch_id", "prop_id"]].rename(
        columns={"srch_id": "SearchId", "prop_id": "PropertyId"}
    )
    submission.to_csv(output_path, index=False)
    print(f"  wrote {len(submission):,} rows to {output_path}")


def run_training_pipeline(config: TrainingConfig) -> dict[str, Any]:
    """Run the historical full-data experiment with explicit paths."""

    training_data = load_training(config.training_data)
    test_data = load_test(config.test_data)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    make_eda_plots(training_data, config.output_directory)

    training, validation = split_by_search(
        training_data,
        fraction_training=0.8,
        random_state=config.random_state,
    )
    fold_maps = build_target_maps(training)
    training_features = make_features(training, fold_maps)
    validation_features = make_features(validation, fold_maps)

    print("Training historical unmitigated LambdaMART...")
    unmitigated, _, sorted_validation = fit_with_early_stopping(
        training_features,
        validation_features,
        random_state=config.random_state,
    )
    sorted_validation["score_unmitigated"] = unmitigated.predict(sorted_validation[FEATURE_COLUMNS])
    ndcg_unmitigated = compute_ndcg_at_k(
        sorted_validation,
        "score_unmitigated",
        k=5,
    )
    ndcg_popularity = popularity_baseline_ndcg(sorted_validation)
    sorted_validation["group"] = family_group(sorted_validation)
    family_before, non_family_before = _group_ndcg(
        sorted_validation,
        "score_unmitigated",
    )

    print("Training exploratory family-weighted LambdaMART...")
    mitigated, _, _ = fit_with_early_stopping(
        training_features,
        validation_features,
        random_state=config.random_state,
        family_weight=FAMILY_WEIGHT,
    )
    sorted_validation["score_mitigated"] = mitigated.predict(sorted_validation[FEATURE_COLUMNS])
    ndcg_mitigated = compute_ndcg_at_k(
        sorted_validation,
        "score_mitigated",
        k=5,
    )
    family_after, non_family_after = _group_ndcg(
        sorted_validation,
        "score_mitigated",
    )
    make_importance_plot(unmitigated, config.output_directory)

    query_group = sorted_validation.groupby("srch_id", as_index=False)["group"].first()
    results: dict[str, Any] = {
        "data": {
            "train_rows": len(training_features),
            "valid_rows": len(validation_features),
            "train_searches": int(training_features["srch_id"].nunique()),
            "valid_searches": int(validation_features["srch_id"].nunique()),
            "book_rate": float(training_data["booking_bool"].mean()),
            "click_rate": float(training_data["click_bool"].mean()),
        },
        "model_performance": {
            "ndcg5_popularity": ndcg_popularity,
            "ndcg5_lambdamart": ndcg_unmitigated,
            "ndcg5_lambdamart_mitigated": ndcg_mitigated,
            "best_iter_unmitigated": int(unmitigated.best_iteration_ or unmitigated.n_estimators),
            "best_iter_mitigated": int(mitigated.best_iteration_ or mitigated.n_estimators),
        },
        "exploratory_fairness": {
            "group_definition": "family if srch_children_count > 0 else non_family",
            "family_share": float((query_group["group"] == "family").mean()),
            "ndcg5_family_before": family_before,
            "ndcg5_non_family_before": non_family_before,
            "gap_before": abs(family_before - non_family_before),
            "ndcg5_family_after": family_after,
            "ndcg5_non_family_after": non_family_after,
            "gap_after": abs(family_after - non_family_after),
        },
        "methodology_limitations": METHODOLOGY_LIMITATIONS,
    }
    metrics_path = config.output_directory / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))

    print("Refitting historical models on the full training set...")
    full_maps = build_target_maps(training_data)
    full_features = make_features(training_data, full_maps)
    full_unmitigated = fit_full(
        full_features,
        n_estimators=int(unmitigated.best_iteration_ or unmitigated.n_estimators),
        random_state=config.random_state,
    )
    full_mitigated = fit_full(
        full_features,
        n_estimators=int(mitigated.best_iteration_ or mitigated.n_estimators),
        random_state=config.random_state,
        family_weight=FAMILY_WEIGHT,
    )

    test_features = make_features(test_data, full_maps)
    write_submission(
        test_features,
        full_unmitigated,
        config.output_directory / "submission_unmitigated.csv",
    )
    write_submission(
        test_features,
        full_mitigated,
        config.output_directory / "submission_family_weighted.csv",
    )
    return results
