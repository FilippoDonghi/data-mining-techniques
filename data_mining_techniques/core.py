"""Kaggle-independent feature engineering and ranking metrics.

This module intentionally depends only on NumPy and pandas. Importing it does not require
LightGBM, OpenMP, the private coursework data, or a writable project directory.
"""

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

PRICE_CLIP = 10_000.0
PRICE_PER_NIGHT_CLIP = 5_000.0

FEATURE_COLUMNS = [
    "site_id",
    "visitor_location_country_id",
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_country_id",
    "prop_starrating",
    "prop_review_score",
    "prop_brand_bool",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd_log",
    "price_per_night_log",
    "promotion_flag",
    "srch_destination_id",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "random_bool",
    "comp_cheaper_count",
    "comp_pricier_count",
    "comp_unavailable_count",
    "comp_diff_max",
    "price_log_z",
    "loc2_z",
    "star_z",
    "review_z",
    "hist_price_z",
    "price_rank_pct",
    "loc2_rank_pct",
    "star_rank_pct",
    "prop_ctr",
    "prop_book_rate",
    "prop_mean_position",
    "dest_prop_score",
    "dest_star_score",
    "dest_popularity",
    "star_hist_diff",
    "price_hist_diff",
    "month",
    "dow",
    "review_rank_pct",
    "srch_query_length",
    "party_size",
    "price_per_person_log",
]


class DataSchemaError(ValueError):
    """Raised when an input table does not have the expected columns."""


def competitor_columns(suffix: str) -> list[str]:
    """Return the eight Expedia competitor columns with the requested suffix."""

    return [f"comp{i}_{suffix}" for i in range(1, 9)]


def training_columns() -> list[str]:
    """Return columns read from the authorized coursework training data."""

    base = [
        "srch_id",
        "date_time",
        "site_id",
        "visitor_location_country_id",
        "visitor_hist_starrating",
        "visitor_hist_adr_usd",
        "prop_country_id",
        "prop_id",
        "prop_starrating",
        "prop_review_score",
        "prop_brand_bool",
        "prop_location_score1",
        "prop_location_score2",
        "prop_log_historical_price",
        "price_usd",
        "promotion_flag",
        "srch_destination_id",
        "srch_length_of_stay",
        "srch_booking_window",
        "srch_adults_count",
        "srch_children_count",
        "srch_room_count",
        "srch_saturday_night_bool",
        "srch_query_affinity_score",
        "orig_destination_distance",
        "random_bool",
    ]
    return (
        base
        + competitor_columns("rate")
        + competitor_columns("inv")
        + competitor_columns("rate_percent_diff")
        + ["position", "click_bool", "booking_bool"]
    )


def test_columns() -> list[str]:
    """Return columns read from the authorized coursework test data."""

    labels = {"position", "click_bool", "booking_bool"}
    return [column for column in training_columns() if column not in labels]


def validate_columns(
    columns: Iterable[str],
    required: Iterable[str],
    *,
    source: str = "input",
) -> None:
    """Raise a readable error listing required columns that are absent."""

    missing = sorted(set(required) - set(columns))
    if missing:
        joined = ", ".join(missing)
        raise DataSchemaError(f"{source} is missing {len(missing)} required column(s): {joined}")


def relevance_label(frame: pd.DataFrame) -> pd.Series:
    """Map a booking to 5, a click without booking to 1, and other rows to 0."""

    validate_columns(
        frame.columns,
        {"click_bool", "booking_bool"},
        source="relevance input",
    )
    click_only = frame["click_bool"] * (1 - frame["booking_bool"])
    return 5 * frame["booking_bool"] + click_only


def compute_ndcg_at_k(
    frame: pd.DataFrame,
    score_column: str,
    *,
    k: int = 5,
    zero_idcg: Literal["skip", "zero"] = "skip",
) -> float:
    """Compute exponential-gain NDCG per search and average it.

    The historical coursework metric skipped searches with no positive relevance.
    Set zero_idcg to "zero" to include those searches as zero. The private competition
    scorer should be checked before comparing either policy with a leaderboard result.
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    if zero_idcg not in {"skip", "zero"}:
        raise ValueError("zero_idcg must be either 'skip' or 'zero'")
    validate_columns(
        frame.columns,
        {"srch_id", "relevance", score_column},
        source="NDCG input",
    )

    scores: list[float] = []
    for _, group in frame.groupby("srch_id", sort=False):
        truth = group["relevance"].to_numpy(dtype=float)
        predicted = group[score_column].to_numpy(dtype=float)
        predicted = np.nan_to_num(predicted, nan=-np.inf)

        order = np.argsort(-predicted, kind="stable")
        ranked_truth = truth[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(ranked_truth) + 2))
        dcg = float(np.sum((2**ranked_truth - 1) * discounts))

        ideal = np.sort(truth)[::-1][:k]
        idcg = float(np.sum((2**ideal - 1) * discounts))
        if idcg > 0:
            scores.append(dcg / idcg)
        elif zero_idcg == "zero":
            scores.append(0.0)

    return float(np.mean(scores)) if scores else 0.0


def family_group(frame: pd.DataFrame) -> np.ndarray:
    """Return the exploratory coursework family/non-family query label."""

    validate_columns(
        frame.columns,
        {"srch_children_count"},
        source="family-group input",
    )
    return np.where(frame["srch_children_count"] > 0, "family", "non_family")


def _add_price_features(frame: pd.DataFrame) -> None:
    price = frame["price_usd"].clip(lower=0, upper=PRICE_CLIP)
    frame["price_usd_log"] = np.log1p(price)
    stay = frame["srch_length_of_stay"].clip(lower=1)
    per_night = (frame["price_usd"] / stay).clip(
        lower=0,
        upper=PRICE_PER_NIGHT_CLIP,
    )
    frame["price_per_night_log"] = np.log1p(per_night)


def _add_competitor_features(frame: pd.DataFrame) -> None:
    rates = frame[competitor_columns("rate")].to_numpy()
    inventory = frame[competitor_columns("inv")].to_numpy()
    differences = frame[competitor_columns("rate_percent_diff")].to_numpy()

    frame["comp_cheaper_count"] = (rates == 1).sum(axis=1).astype(np.int8)
    frame["comp_pricier_count"] = (rates == -1).sum(axis=1).astype(np.int8)
    frame["comp_unavailable_count"] = (inventory == 1).sum(axis=1).astype(np.int8)

    clipped = np.clip(differences, 0, 100)
    available = ~np.isnan(clipped)
    safe_values = np.where(available, clipped, -np.inf)
    maxima = safe_values.max(axis=1)
    frame["comp_diff_max"] = np.where(available.any(axis=1), maxima, 0.0)


def _add_within_query_features(frame: pd.DataFrame) -> None:
    grouped = frame.groupby("srch_id", sort=False)

    def add_zscore(column: str, output: str) -> None:
        mean = grouped[column].transform("mean")
        standard_deviation = grouped[column].transform("std")
        standard_deviation = standard_deviation.where(
            standard_deviation > 0,
            np.nan,
        )
        frame[output] = (frame[column] - mean) / standard_deviation

    add_zscore("price_usd_log", "price_log_z")
    add_zscore("prop_location_score2", "loc2_z")
    add_zscore("prop_starrating", "star_z")
    add_zscore("prop_review_score", "review_z")
    add_zscore("prop_log_historical_price", "hist_price_z")

    frame["price_rank_pct"] = grouped["price_usd"].rank(method="average", pct=True)
    frame["loc2_rank_pct"] = grouped["prop_location_score2"].rank(
        method="average",
        pct=True,
    )
    frame["star_rank_pct"] = grouped["prop_starrating"].rank(
        method="average",
        pct=True,
    )
    frame["review_rank_pct"] = grouped["prop_review_score"].rank(
        method="average",
        pct=True,
    )
    frame["srch_query_length"] = grouped["srch_id"].transform("count").astype(np.int16)


def _add_visitor_history_features(frame: pd.DataFrame) -> None:
    frame["star_hist_diff"] = (frame["visitor_hist_starrating"] - frame["prop_starrating"]).abs()
    frame["price_hist_diff"] = (frame["visitor_hist_adr_usd"] - frame["price_usd"]).abs()
    party_size = (frame["srch_adults_count"] + frame["srch_children_count"]).clip(lower=1)
    frame["party_size"] = party_size.astype(np.int16)
    per_night = np.expm1(frame["price_per_night_log"])
    per_person = (per_night / party_size).clip(upper=PRICE_PER_NIGHT_CLIP)
    frame["price_per_person_log"] = np.log1p(per_person)


def _add_date_features(frame: pd.DataFrame) -> None:
    parsed = pd.to_datetime(frame["date_time"], errors="coerce")
    frame["month"] = parsed.dt.month.fillna(-1).astype(np.int8)
    frame["dow"] = parsed.dt.dayofweek.fillna(-1).astype(np.int8)


def build_target_maps(training_frame: pd.DataFrame) -> dict[str, Any]:
    """Build the historical smoothed popularity features from labeled rows.

    These maps preserve the coursework approach. Callers training a model should use
    out-of-fold encodings rather than applying maps to the rows that created them.
    """

    required = {
        "booking_bool",
        "click_bool",
        "position",
        "prop_id",
        "prop_starrating",
        "random_bool",
        "srch_destination_id",
    }
    validate_columns(training_frame.columns, required, source="target-map input")
    if training_frame.empty:
        raise DataSchemaError("target-map input must contain at least one row")

    click_prior = float(training_frame["click_bool"].mean())
    book_prior = float(training_frame["booking_bool"].mean())
    position_prior = float(training_frame["position"].mean())
    smoothing = 25.0

    unbiased = training_frame[training_frame["random_bool"] == 1]
    if len(unbiased) < 5_000:
        unbiased = training_frame

    property_aggregates = unbiased.groupby("prop_id").agg(
        n=("click_bool", "size"),
        clicks=("click_bool", "sum"),
        books=("booking_bool", "sum"),
    )
    mean_position = training_frame.groupby("prop_id")["position"].mean().rename("mean_pos")
    property_aggregates = property_aggregates.join(mean_position, how="left")
    property_aggregates["prop_ctr"] = (property_aggregates["clicks"] + smoothing * click_prior) / (
        property_aggregates["n"] + smoothing
    )
    property_aggregates["prop_book_rate"] = (
        property_aggregates["books"] + smoothing * book_prior
    ) / (property_aggregates["n"] + smoothing)

    relevance = unbiased[["srch_destination_id", "prop_id", "click_bool", "booking_bool"]].copy()
    relevance["relevance"] = relevance_label(unbiased)
    destination_mean = relevance.groupby("srch_destination_id")["relevance"].mean()
    destination_prior = float(relevance["relevance"].mean())
    destination_smoothing = 10.0

    destination_property = relevance.groupby(["srch_destination_id", "prop_id"]).agg(
        n=("relevance", "size"),
        rel_sum=("relevance", "sum"),
    )
    destination_property = destination_property.join(
        destination_mean.rename("dest_mean"),
        on="srch_destination_id",
    )
    destination_property["dest_prop_score"] = (
        destination_property["rel_sum"] + destination_smoothing * destination_property["dest_mean"]
    ) / (destination_property["n"] + destination_smoothing)

    destination_star = unbiased.groupby(["srch_destination_id", "prop_starrating"]).agg(
        n=("booking_bool", "size"),
        books=("booking_bool", "sum"),
    )
    destination_star["dest_star_score"] = (
        destination_star["books"] + destination_smoothing * book_prior
    ) / (destination_star["n"] + destination_smoothing)

    destination_popularity = unbiased.groupby("srch_destination_id").agg(
        n=("booking_bool", "size"),
        books=("booking_bool", "sum"),
    )
    destination_popularity["dest_popularity"] = (
        destination_popularity["books"] + destination_smoothing * book_prior
    ) / (destination_popularity["n"] + destination_smoothing)

    return {
        "prop_ctr": property_aggregates["prop_ctr"],
        "prop_book_rate": property_aggregates["prop_book_rate"],
        "prop_mean_position": property_aggregates["mean_pos"],
        "dest_prop_score": destination_property["dest_prop_score"],
        "dest_star_score": destination_star["dest_star_score"],
        "dest_popularity": destination_popularity["dest_popularity"],
        "fallback_ctr": click_prior,
        "fallback_book": book_prior,
        "fallback_pos": position_prior,
        "fallback_dest": destination_prior,
        "fallback_dest_star": book_prior,
        "fallback_dest_pop": book_prior,
    }


def _apply_target_maps(frame: pd.DataFrame, maps: dict[str, Any]) -> None:
    frame["prop_ctr"] = frame["prop_id"].map(maps["prop_ctr"]).fillna(maps["fallback_ctr"])
    frame["prop_book_rate"] = (
        frame["prop_id"].map(maps["prop_book_rate"]).fillna(maps["fallback_book"])
    )
    frame["prop_mean_position"] = (
        frame["prop_id"].map(maps["prop_mean_position"]).fillna(maps["fallback_pos"])
    )

    destination_property_index = pd.MultiIndex.from_frame(frame[["srch_destination_id", "prop_id"]])
    destination_scores = maps["dest_prop_score"].reindex(destination_property_index)
    frame["dest_prop_score"] = destination_scores.to_numpy()
    frame["dest_prop_score"] = frame["dest_prop_score"].fillna(maps["fallback_dest"])

    destination_star_index = pd.MultiIndex.from_frame(
        frame[["srch_destination_id", "prop_starrating"]]
    )
    star_scores = maps["dest_star_score"].reindex(destination_star_index)
    frame["dest_star_score"] = star_scores.to_numpy()
    frame["dest_star_score"] = frame["dest_star_score"].fillna(maps["fallback_dest_star"])

    popularity = maps["dest_popularity"].reindex(frame["srch_destination_id"])
    frame["dest_popularity"] = popularity.to_numpy()
    frame["dest_popularity"] = frame["dest_popularity"].fillna(maps["fallback_dest_pop"])


def make_features(
    frame: pd.DataFrame,
    target_maps: dict[str, Any],
) -> pd.DataFrame:
    """Return a feature-enriched copy of a raw training or test frame."""

    validate_columns(frame.columns, test_columns(), source="feature input")
    output = frame.copy()
    _add_price_features(output)
    _add_competitor_features(output)
    _add_within_query_features(output)
    _add_visitor_history_features(output)
    _add_date_features(output)
    _apply_target_maps(output, target_maps)
    return output


def split_by_search(
    frame: pd.DataFrame,
    *,
    fraction_training: float = 0.8,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic random split without dividing a search across folds."""

    validate_columns(frame.columns, {"srch_id"}, source="split input")
    if not 0 < fraction_training < 1:
        raise ValueError("fraction_training must be strictly between 0 and 1")

    search_ids = frame["srch_id"].drop_duplicates().to_numpy(copy=True)
    if len(search_ids) < 2:
        raise ValueError("split input must contain at least two searches")

    generator = np.random.default_rng(random_state)
    generator.shuffle(search_ids)
    cut = max(1, min(len(search_ids) - 1, int(fraction_training * len(search_ids))))
    training_ids = set(search_ids[:cut])
    training = frame[frame["srch_id"].isin(training_ids)].copy()
    validation = frame[~frame["srch_id"].isin(training_ids)].copy()
    return training, validation


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Return LightGBM group sizes; callers must first sort by search ID."""

    validate_columns(frame.columns, {"srch_id"}, source="group-size input")
    return frame.groupby("srch_id", sort=False).size().to_numpy()
