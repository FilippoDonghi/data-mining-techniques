"""Tests for Kaggle-independent feature and metric behavior."""

import warnings

import numpy as np
import pandas as pd
import pytest

from data_mining_techniques.core import (
    FEATURE_COLUMNS,
    DataSchemaError,
    build_target_maps,
    compute_ndcg_at_k,
    make_features,
    relevance_label,
    split_by_search,
    validate_columns,
)
from data_mining_techniques.demo import synthetic_training_data


def test_relevance_label_uses_booking_then_click_grades() -> None:
    frame = pd.DataFrame(
        {
            "click_bool": [0, 1, 1],
            "booking_bool": [0, 0, 1],
        }
    )

    assert relevance_label(frame).tolist() == [0, 1, 5]


def test_ndcg_zero_relevance_policy_is_explicit() -> None:
    frame = pd.DataFrame(
        {
            "srch_id": [1, 1, 2, 2],
            "relevance": [5, 0, 0, 0],
            "score": [1.0, 0.0, 1.0, 0.0],
        }
    )

    assert compute_ndcg_at_k(frame, "score", zero_idcg="skip") == pytest.approx(1.0)
    assert compute_ndcg_at_k(frame, "score", zero_idcg="zero") == pytest.approx(0.5)


def test_query_split_is_deterministic_and_disjoint() -> None:
    frame = synthetic_training_data()

    first_train, first_valid = split_by_search(frame, random_state=7)
    second_train, second_valid = split_by_search(frame, random_state=7)
    train_ids = set(first_train["srch_id"])
    valid_ids = set(first_valid["srch_id"])

    assert train_ids.isdisjoint(valid_ids)
    assert train_ids | valid_ids == set(frame["srch_id"])
    assert first_train.index.tolist() == second_train.index.tolist()
    assert first_valid.index.tolist() == second_valid.index.tolist()


def test_feature_engineering_handles_missing_competitor_rows_without_warning() -> None:
    frame = synthetic_training_data()
    maps = build_target_maps(frame)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = make_features(frame, maps)

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert len(features) == len(frame)
    assert features.loc[features["prop_id"] == 1002, "comp_diff_max"].eq(0).all()
    assert not any("All-NaN slice" in str(item.message) for item in caught)
    assert "price_usd_log" not in frame.columns


def test_schema_error_lists_missing_columns() -> None:
    with pytest.raises(DataSchemaError, match=r"missing 2 required column"):
        validate_columns(["present"], ["present", "alpha", "beta"], source="demo CSV")


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_query_split_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        split_by_search(
            pd.DataFrame({"srch_id": [1, 2]}),
            fraction_training=fraction,
        )


def test_ndcg_rejects_unknown_policy() -> None:
    frame = pd.DataFrame({"srch_id": [1], "relevance": [1], "score": [1]})
    with pytest.raises(ValueError, match="zero_idcg"):
        compute_ndcg_at_k(frame, "score", zero_idcg="unknown")  # type: ignore[arg-type]


def test_demo_competitor_counts_are_numeric() -> None:
    frame = synthetic_training_data()
    features = make_features(frame, build_target_maps(frame))

    assert np.issubdtype(features["comp_cheaper_count"].dtype, np.integer)
    assert features["comp_cheaper_count"].max() == 1
