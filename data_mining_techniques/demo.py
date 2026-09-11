"""Small deterministic demo that does not use the restricted competition data."""

from typing import Any

import numpy as np
import pandas as pd

from .core import (
    FEATURE_COLUMNS,
    build_target_maps,
    competitor_columns,
    compute_ndcg_at_k,
    make_features,
    relevance_label,
    split_by_search,
)


def synthetic_training_data() -> pd.DataFrame:
    """Return four tiny hotel-search lists with the original input schema."""

    rows = 12
    search_ids = np.repeat([101, 102, 103, 104], 3)
    destinations = np.repeat([501, 502, 503, 504], 3)
    properties = np.tile([1001, 1002, 1003], 4)
    frame = pd.DataFrame(
        {
            "srch_id": search_ids,
            "date_time": pd.date_range("2026-01-01", periods=rows, freq="h").astype(str),
            "site_id": np.tile([1, 1, 2], 4),
            "visitor_location_country_id": np.repeat([10, 11, 12, 13], 3),
            "visitor_hist_starrating": np.tile([3.0, 4.0, np.nan], 4),
            "visitor_hist_adr_usd": np.tile([110.0, 145.0, np.nan], 4),
            "prop_country_id": np.repeat([20, 21, 22, 23], 3),
            "prop_id": properties,
            "prop_starrating": np.tile([3, 4, 5], 4),
            "prop_review_score": np.tile([3.5, 4.0, 4.5], 4),
            "prop_brand_bool": np.tile([0, 1, 1], 4),
            "prop_location_score1": np.tile([1.2, 2.0, 2.8], 4),
            "prop_location_score2": np.tile([0.25, 0.55, 0.8], 4),
            "prop_log_historical_price": np.tile([4.3, 4.7, 5.0], 4),
            "price_usd": np.tile([90.0, 125.0, 180.0], 4),
            "promotion_flag": np.tile([1, 0, 0], 4),
            "srch_destination_id": destinations,
            "srch_length_of_stay": np.repeat([2, 3, 1, 4], 3),
            "srch_booking_window": np.repeat([7, 21, 3, 30], 3),
            "srch_adults_count": np.repeat([2, 2, 1, 2], 3),
            "srch_children_count": np.repeat([0, 1, 0, 2], 3),
            "srch_room_count": np.repeat([1, 1, 1, 2], 3),
            "srch_saturday_night_bool": np.repeat([0, 1, 0, 1], 3),
            "srch_query_affinity_score": np.tile([-3.0, -2.0, -1.0], 4),
            "orig_destination_distance": np.tile([120.0, 80.0, 40.0], 4),
            "random_bool": np.repeat([1, 1, 0, 1], 3),
            "position": np.tile([1, 2, 3], 4),
            "click_bool": [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            "booking_bool": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        }
    )

    for index, column in enumerate(competitor_columns("rate"), start=1):
        frame[column] = np.where(properties == 1001, 1 if index == 1 else np.nan, np.nan)
    for index, column in enumerate(competitor_columns("inv"), start=1):
        frame[column] = np.where(properties == 1003, 1 if index == 2 else np.nan, np.nan)
    for index, column in enumerate(competitor_columns("rate_percent_diff"), start=1):
        frame[column] = np.where(properties == 1001, 8.0 if index == 1 else np.nan, np.nan)
    return frame


def run_demo() -> dict[str, Any]:
    """Exercise schema checks, feature engineering, splitting, and NDCG."""

    frame = synthetic_training_data()
    frame["relevance"] = relevance_label(frame)
    training, validation = split_by_search(
        frame,
        fraction_training=0.75,
        random_state=42,
    )
    maps = build_target_maps(training)
    validation_features = make_features(validation, maps)
    validation_features["popularity_score"] = (
        0.5 * validation_features["dest_prop_score"] + 0.5 * validation_features["prop_ctr"]
    )
    ndcg = compute_ndcg_at_k(
        validation_features,
        "popularity_score",
        k=5,
    )

    return {
        "mode": "synthetic-demo",
        "rows": len(frame),
        "searches": int(frame["srch_id"].nunique()),
        "training_searches": int(training["srch_id"].nunique()),
        "validation_searches": int(validation["srch_id"].nunique()),
        "feature_count": len(FEATURE_COLUMNS),
        "ndcg5_popularity": ndcg,
        "note": (
            "Synthetic smoke-test metric only; it is not comparable with the "
            "historical coursework result."
        ),
    }
