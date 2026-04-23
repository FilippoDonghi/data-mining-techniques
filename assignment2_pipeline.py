import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


TRAIN_PATH = "training_set_VU_DM.csv"
TEST_PATH = "test_set_VU_DM.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Keep the training subset manageable while still statistically meaningful.
MAX_TRAIN_ROWS = 1_200_000
RANDOM_STATE = 42

BASE_COLUMNS = [
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
    "comp1_rate",
    "comp1_inv",
    "comp1_rate_percent_diff",
    "comp2_rate",
    "comp2_inv",
    "comp2_rate_percent_diff",
    "comp3_rate",
    "comp3_inv",
    "comp3_rate_percent_diff",
    "comp4_rate",
    "comp4_inv",
    "comp4_rate_percent_diff",
    "comp5_rate",
    "comp5_inv",
    "comp5_rate_percent_diff",
    "comp6_rate",
    "comp6_inv",
    "comp6_rate_percent_diff",
    "comp7_rate",
    "comp7_inv",
    "comp7_rate_percent_diff",
    "comp8_rate",
    "comp8_inv",
    "comp8_rate_percent_diff",
    "click_bool",
    "booking_bool",
]

FEATURE_COLUMNS = [
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
    "comp_rate_mean",
    "comp_inv_mean",
    "comp_price_diff_mean",
    "price_rank_pct",
    "price_vs_search_mean",
    "star_vs_search_mean",
    "review_vs_search_mean",
    "prop_pop_score",
    "dest_prop_pop_score",
]


def relevance_label(df: pd.DataFrame) -> pd.Series:
    click_only = df["click_bool"] * (1 - df["booking_bool"])
    return 5 * df["booking_bool"] + click_only


def compute_ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = 5) -> float:
    rows = []
    for _, grp in df.groupby("srch_id", sort=False):
        y_true = grp["relevance"].to_numpy(dtype=float)
        y_score = grp[score_col].to_numpy(dtype=float)
        order = np.argsort(-y_score)
        top_true = y_true[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(top_true) + 2))
        dcg = np.sum(top_true * discounts)

        ideal = np.sort(y_true)[::-1][:k]
        idcg = np.sum(ideal * discounts)
        if idcg > 0:
            rows.append(dcg / idcg)
    return float(np.mean(rows)) if rows else 0.0


def family_group(df: pd.DataFrame) -> pd.Series:
    # Proxy segmentation used for fairness analysis.
    return np.where(df["srch_children_count"] > 0, "family", "non_family")


def add_competitor_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    rate_cols = [f"comp{i}_rate" for i in range(1, 9)]
    inv_cols = [f"comp{i}_inv" for i in range(1, 9)]
    diff_cols = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]

    df["comp_rate_mean"] = df[rate_cols].mean(axis=1)
    df["comp_inv_mean"] = df[inv_cols].mean(axis=1)
    df["comp_price_diff_mean"] = df[diff_cols].mean(axis=1)
    return df


def add_search_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("srch_id", sort=False)
    df["price_rank_pct"] = group["price_usd"].rank(method="average", pct=True)
    df["price_vs_search_mean"] = df["price_usd"] - group["price_usd"].transform("mean")
    df["star_vs_search_mean"] = df["prop_starrating"] - group["prop_starrating"].transform("mean")
    df["review_vs_search_mean"] = df["prop_review_score"].fillna(0) - group["prop_review_score"].transform("mean").fillna(0)
    return df


def build_popularity_maps(df: pd.DataFrame):
    tmp = df[["prop_id", "srch_destination_id", "relevance"]].copy()

    global_prop = (
        tmp.groupby("prop_id", as_index=True)["relevance"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "rel_sum", "count": "n"})
    )
    global_mean = tmp["relevance"].mean()
    m_global = 20.0
    global_prop["score"] = (global_prop["rel_sum"] + m_global * global_mean) / (global_prop["n"] + m_global)

    dest_prop = (
        tmp.groupby(["srch_destination_id", "prop_id"], as_index=True)["relevance"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "rel_sum", "count": "n"})
    )
    dest_mean = tmp.groupby("srch_destination_id")["relevance"].mean()
    m_dest = 10.0
    dest_prop = dest_prop.join(dest_mean.rename("dest_mean"), on="srch_destination_id")
    dest_prop["score"] = (dest_prop["rel_sum"] + m_dest * dest_prop["dest_mean"]) / (dest_prop["n"] + m_dest)

    return global_prop["score"], dest_prop["score"], float(global_mean)


def apply_popularity_features(
    df: pd.DataFrame,
    global_score_map: pd.Series,
    dest_score_map: pd.Series,
    fallback_mean: float,
) -> pd.DataFrame:
    df["prop_pop_score"] = df["prop_id"].map(global_score_map).fillna(fallback_mean)
    index_key = pd.MultiIndex.from_frame(df[["srch_destination_id", "prop_id"]])
    dest_scores = dest_score_map.reindex(index_key).to_numpy()
    df["dest_prop_pop_score"] = np.where(np.isnan(dest_scores), df["prop_pop_score"], dest_scores)
    return df


def make_eda_plots(df: pd.DataFrame) -> None:
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    missing_pct.plot(kind="bar", color="#2b8cbe")
    plt.title("Top 15 Features by Missingness (%)")
    plt.ylabel("Missing %")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missingness_top15.png", dpi=150)
    plt.close()

    sample = df[["price_usd", "prop_starrating", "prop_review_score", "relevance"]].copy()
    sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
    sample = sample.sample(min(len(sample), 100_000), random_state=RANDOM_STATE)

    plt.figure(figsize=(8, 5))
    plt.scatter(sample["price_usd"], sample["relevance"], s=5, alpha=0.08, c="#f03b20")
    plt.xscale("log")
    plt.title("Price vs. Relevance (log-price scale)")
    plt.xlabel("price_usd (log scale)")
    plt.ylabel("relevance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_vs_relevance.png", dpi=150)
    plt.close()


def load_training_subset() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH, usecols=BASE_COLUMNS, nrows=MAX_TRAIN_ROWS)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["relevance"] = relevance_label(df)
    return df


def train_and_evaluate(df: pd.DataFrame):
    unique_searches = df["srch_id"].drop_duplicates().sample(frac=1.0, random_state=RANDOM_STATE)
    split_idx = int(0.8 * len(unique_searches))
    train_ids = set(unique_searches.iloc[:split_idx].tolist())

    train_df = df[df["srch_id"].isin(train_ids)].copy()
    valid_df = df[~df["srch_id"].isin(train_ids)].copy()

    global_pop, dest_pop, fallback_mean = build_popularity_maps(train_df)
    train_df = apply_popularity_features(train_df, global_pop, dest_pop, fallback_mean)
    valid_df = apply_popularity_features(valid_df, global_pop, dest_pop, fallback_mean)

    train_df = add_competitor_aggregates(train_df)
    valid_df = add_competitor_aggregates(valid_df)
    train_df = add_search_relative_features(train_df)
    valid_df = add_search_relative_features(valid_df)

    # Technique 1: recommender-style popularity ranking.
    valid_df["score_popularity"] = 0.4 * valid_df["prop_pop_score"] + 0.6 * valid_df["dest_prop_pop_score"]
    ndcg_pop = compute_ndcg_at_k(valid_df, "score_popularity", k=5)

    # Technique 2: gradient boosting pointwise rank surrogate.
    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.06,
        max_iter=300,
        l2_regularization=0.02,
        random_state=RANDOM_STATE,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["relevance"]
    X_valid = valid_df[FEATURE_COLUMNS]

    model.fit(X_train, y_train)
    valid_df["score_gbdt"] = model.predict(X_valid)
    ndcg_gbdt = compute_ndcg_at_k(valid_df, "score_gbdt", k=5)

    # Fairness audit by family vs non-family searches.
    valid_df["group"] = family_group(valid_df)
    query_group = valid_df.groupby("srch_id", as_index=False)["group"].first()
    family_ids = set(query_group.loc[query_group["group"] == "family", "srch_id"])
    non_family_ids = set(query_group.loc[query_group["group"] == "non_family", "srch_id"])

    ndcg_family_before = compute_ndcg_at_k(valid_df[valid_df["srch_id"].isin(family_ids)], "score_gbdt", k=5)
    ndcg_non_family_before = compute_ndcg_at_k(valid_df[valid_df["srch_id"].isin(non_family_ids)], "score_gbdt", k=5)

    # Pre-processing mitigation: up-weight family queries.
    train_df["group"] = family_group(train_df)
    sample_weight = np.where(train_df["group"] == "family", 1.6, 1.0)

    mitigated_model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.06,
        max_iter=300,
        l2_regularization=0.02,
        random_state=RANDOM_STATE,
    )
    mitigated_model.fit(X_train, y_train, sample_weight=sample_weight)
    valid_df["score_gbdt_mitigated"] = mitigated_model.predict(X_valid)

    ndcg_gbdt_mitigated = compute_ndcg_at_k(valid_df, "score_gbdt_mitigated", k=5)
    ndcg_family_after = compute_ndcg_at_k(valid_df[valid_df["srch_id"].isin(family_ids)], "score_gbdt_mitigated", k=5)
    ndcg_non_family_after = compute_ndcg_at_k(valid_df[valid_df["srch_id"].isin(non_family_ids)], "score_gbdt_mitigated", k=5)

    results = {
        "data": {
            "train_rows": int(len(train_df)),
            "valid_rows": int(len(valid_df)),
            "train_searches": int(train_df["srch_id"].nunique()),
            "valid_searches": int(valid_df["srch_id"].nunique()),
            "book_rate": float(df["booking_bool"].mean()),
            "click_rate": float(df["click_bool"].mean()),
        },
        "model_performance": {
            "ndcg5_popularity": ndcg_pop,
            "ndcg5_gbdt": ndcg_gbdt,
            "ndcg5_gbdt_mitigated": ndcg_gbdt_mitigated,
        },
        "fairness": {
            "group_definition": "family if srch_children_count > 0 else non_family",
            "family_share": float((query_group["group"] == "family").mean()),
            "ndcg5_family_before": ndcg_family_before,
            "ndcg5_non_family_before": ndcg_non_family_before,
            "gap_before": float(abs(ndcg_non_family_before - ndcg_family_before)),
            "ndcg5_family_after": ndcg_family_after,
            "ndcg5_non_family_after": ndcg_non_family_after,
            "gap_after": float(abs(ndcg_non_family_after - ndcg_family_after)),
        },
    }

    feature_importance_proxy = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "std": X_train[FEATURE_COLUMNS].std(numeric_only=True).fillna(0).values,
        }
    ).sort_values("std", ascending=False)
    feature_importance_proxy.head(15).to_csv(OUTPUT_DIR / "feature_variability_top15.csv", index=False)

    return mitigated_model, global_pop, dest_pop, fallback_mean, results


def fit_full_training_and_predict(
    model_template: HistGradientBoostingRegressor,
    global_pop: pd.Series,
    dest_pop: pd.Series,
    fallback_mean: float,
) -> None:
    # Retrain on the same row budget but all searches for final inference.
    full_df = load_training_subset()
    full_df = apply_popularity_features(full_df, global_pop, dest_pop, fallback_mean)
    full_df = add_competitor_aggregates(full_df)
    full_df = add_search_relative_features(full_df)

    X_full = full_df[FEATURE_COLUMNS]
    y_full = full_df["relevance"]
    sample_weight = np.where(family_group(full_df) == "family", 1.6, 1.0)

    final_model = HistGradientBoostingRegressor(
        max_depth=model_template.max_depth,
        learning_rate=model_template.learning_rate,
        max_iter=model_template.max_iter,
        l2_regularization=model_template.l2_regularization,
        random_state=RANDOM_STATE,
    )
    final_model.fit(X_full, y_full, sample_weight=sample_weight)

    usecols_test = [c for c in BASE_COLUMNS if c not in {"click_bool", "booking_bool", "date_time"}] + ["date_time"]
    usecols_test = [c for c in usecols_test if c in pd.read_csv(TEST_PATH, nrows=1).columns]

    ranked_parts = []
    for chunk in pd.read_csv(TEST_PATH, chunksize=400_000):
        chunk = apply_popularity_features(chunk, global_pop, dest_pop, fallback_mean)
        chunk = add_competitor_aggregates(chunk)
        chunk = add_search_relative_features(chunk)

        X_chunk = chunk[FEATURE_COLUMNS]
        chunk["score"] = final_model.predict(X_chunk)
        chunk = chunk.sort_values(["srch_id", "score"], ascending=[True, False])
        ranked_parts.append(chunk[["srch_id", "prop_id"]])

    submission = pd.concat(ranked_parts, ignore_index=True)
    submission.to_csv("submission_final.csv", index=False)


def main() -> None:
    df = load_training_subset()
    make_eda_plots(df)

    model, global_pop, dest_pop, fallback_mean, results = train_and_evaluate(df)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fit_full_training_and_predict(model, global_pop, dest_pop, fallback_mean)

    print("Finished.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
