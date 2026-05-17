import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIN_PATH = "training_set_VU_DM.csv"
TEST_PATH = "test_set_VU_DM.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
PRICE_CLIP = 10_000.0
PRICE_PER_NIGHT_CLIP = 5_000.0
FAMILY_WEIGHT = 1.6

#full feature column list as fed into LightGBM
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


def comp_cols(prefix):
    return [f"comp{i}_{prefix}" for i in range(1, 9)]


def train_usecols():
    base = [
        "srch_id", "date_time", "site_id", "visitor_location_country_id",
        "visitor_hist_starrating", "visitor_hist_adr_usd",
        "prop_country_id", "prop_id", "prop_starrating", "prop_review_score",
        "prop_brand_bool", "prop_location_score1", "prop_location_score2",
        "prop_log_historical_price", "price_usd", "promotion_flag",
        "srch_destination_id", "srch_length_of_stay", "srch_booking_window",
        "srch_adults_count", "srch_children_count", "srch_room_count",
        "srch_saturday_night_bool", "srch_query_affinity_score",
        "orig_destination_distance", "random_bool",
    ]
    return base + comp_cols("rate") + comp_cols("inv") + comp_cols("rate_percent_diff") + [
        "position", "click_bool", "booking_bool",
    ]


def test_usecols():
    drop = {"position", "click_bool", "booking_bool"}
    return [c for c in train_usecols() if c not in drop]


def relevance_label(df):
    click_only = df["click_bool"] * (1 - df["booking_bool"])
    return 5 * df["booking_bool"] + click_only


def compute_ndcg_at_k(df, score_col, k=5):
    #exponential-gain NDCG matching the Kaggle definition
    scores = []
    for _, grp in df.groupby("srch_id", sort=False):
        y_true = grp["relevance"].to_numpy(dtype=float)
        y_score = grp[score_col].to_numpy(dtype=float)
        order = np.argsort(-y_score)
        top_true = y_true[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(top_true) + 2))
        dcg = float(np.sum((2 ** top_true - 1) * discounts))

        ideal = np.sort(y_true)[::-1][:k]
        idcg = float(np.sum((2 ** ideal - 1) * discounts))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def family_group(df):
    return np.where(df["srch_children_count"] > 0, "family", "non_family")


#feature engineering helpers


def add_price_features(df):
    price = df["price_usd"].clip(lower=0, upper=PRICE_CLIP)
    df["price_usd_log"] = np.log1p(price)
    los = df["srch_length_of_stay"].clip(lower=1)
    per_night = (df["price_usd"] / los).clip(lower=0, upper=PRICE_PER_NIGHT_CLIP)
    df["price_per_night_log"] = np.log1p(per_night)
    return df


def add_competitor_features(df):
    rate = df[comp_cols("rate")].to_numpy()
    inv = df[comp_cols("inv")].to_numpy()
    diff = df[comp_cols("rate_percent_diff")].to_numpy()

    #NaN comparisons evaluate to False, so summing booleans counts non-NaN matches
    df["comp_cheaper_count"] = (rate == 1).sum(axis=1).astype(np.int8)
    df["comp_pricier_count"] = (rate == -1).sum(axis=1).astype(np.int8)
    df["comp_unavailable_count"] = (inv == 1).sum(axis=1).astype(np.int8)

    diff_clipped = np.clip(diff, 0, 100)
    max_diff = np.nanmax(diff_clipped, axis=1)
    df["comp_diff_max"] = np.where(np.isnan(max_diff), 0.0, max_diff)
    return df


def add_within_query_features(df):
    g = df.groupby("srch_id", sort=False)

    def zscore(col, out):
        mean = g[col].transform("mean")
        std = g[col].transform("std")
        #avoid divide-by-zero when all rows in a query share the same value
        std = std.where(std > 0, np.nan)
        df[out] = (df[col] - mean) / std

    zscore("price_usd_log", "price_log_z")
    zscore("prop_location_score2", "loc2_z")
    zscore("prop_starrating", "star_z")
    zscore("prop_review_score", "review_z")
    zscore("prop_log_historical_price", "hist_price_z")

    df["price_rank_pct"] = g["price_usd"].rank(method="average", pct=True)
    df["loc2_rank_pct"] = g["prop_location_score2"].rank(method="average", pct=True)
    df["star_rank_pct"] = g["prop_starrating"].rank(method="average", pct=True)
    df["review_rank_pct"] = g["prop_review_score"].rank(method="average", pct=True)
    df["srch_query_length"] = g["srch_id"].transform("count").astype(np.int16)
    return df


def add_visitor_history_features(df):
    df["star_hist_diff"] = (df["visitor_hist_starrating"] - df["prop_starrating"]).abs()
    df["price_hist_diff"] = (df["visitor_hist_adr_usd"] - df["price_usd"]).abs()
    party = (df["srch_adults_count"] + df["srch_children_count"]).clip(lower=1)
    df["party_size"] = party.astype(np.int8)
    per_night = np.expm1(df["price_per_night_log"])
    df["price_per_person_log"] = np.log1p((per_night / party).clip(upper=PRICE_PER_NIGHT_CLIP))
    return df


def add_date_features(df):
    dt = pd.to_datetime(df["date_time"], errors="coerce")
    df["month"] = dt.dt.month.fillna(-1).astype(np.int8)
    df["dow"] = dt.dt.dayofweek.fillna(-1).astype(np.int8)
    return df


#target encoding fit on a training fold then applied to any frame


def build_target_maps(train_df):
    click_prior = float(train_df["click_bool"].mean())
    book_prior = float(train_df["booking_bool"].mean())
    pos_prior = float(train_df["position"].mean())
    m = 25.0

    #use only random-order searches for unbiased CTR and booking rates;
    #when random_bool=0 Expedia's own ranking causes position bias that inflates
    #click/book rates for already-top-ranked hotels
    unbiased = train_df[train_df["random_bool"] == 1]
    if len(unbiased) < 5000:
        unbiased = train_df  #fallback for tiny folds

    prop_agg = unbiased.groupby("prop_id").agg(
        n=("click_bool", "size"),
        clicks=("click_bool", "sum"),
        books=("booking_bool", "sum"),
    )
    #mean position uses all data — position itself is still an informative signal
    pos_agg = train_df.groupby("prop_id")["position"].mean().rename("mean_pos")
    prop_agg = prop_agg.join(pos_agg, how="left")

    prop_agg["prop_ctr"] = (prop_agg["clicks"] + m * click_prior) / (prop_agg["n"] + m)
    prop_agg["prop_book_rate"] = (prop_agg["books"] + m * book_prior) / (prop_agg["n"] + m)

    #unbiased destination-level relevance scores
    rel = unbiased[["srch_destination_id", "prop_id", "click_bool", "booking_bool"]].copy()
    rel["relevance"] = relevance_label(unbiased)
    dest_mean = rel.groupby("srch_destination_id")["relevance"].mean()
    dest_global = float(rel["relevance"].mean())
    md = 10.0
    dest_prop = rel.groupby(["srch_destination_id", "prop_id"]).agg(
        n=("relevance", "size"),
        rel_sum=("relevance", "sum"),
    )
    dest_prop = dest_prop.join(dest_mean.rename("dest_mean"), on="srch_destination_id")
    dest_prop["dest_prop_score"] = (
        (dest_prop["rel_sum"] + md * dest_prop["dest_mean"]) / (dest_prop["n"] + md)
    )

    #destination-star affinity: how well does each star level perform at each destination?
    dest_star = unbiased.groupby(["srch_destination_id", "prop_starrating"]).agg(
        n=("booking_bool", "size"),
        books=("booking_bool", "sum"),
    )
    dest_star["dest_star_score"] = (
        (dest_star["books"] + md * book_prior) / (dest_star["n"] + md)
    )

    #overall destination booking rate
    dest_pop = unbiased.groupby("srch_destination_id").agg(
        n=("booking_bool", "size"),
        books=("booking_bool", "sum"),
    )
    dest_pop["dest_popularity"] = (dest_pop["books"] + md * book_prior) / (dest_pop["n"] + md)

    return {
        "prop_ctr": prop_agg["prop_ctr"],
        "prop_book_rate": prop_agg["prop_book_rate"],
        "prop_mean_position": prop_agg["mean_pos"],
        "dest_prop_score": dest_prop["dest_prop_score"],
        "dest_star_score": dest_star["dest_star_score"],
        "dest_popularity": dest_pop["dest_popularity"],
        "fallback_ctr": click_prior,
        "fallback_book": book_prior,
        "fallback_pos": pos_prior,
        "fallback_dest": dest_global,
        "fallback_dest_star": book_prior,
        "fallback_dest_pop": book_prior,
    }


def apply_target_maps(df, maps):
    df["prop_ctr"] = df["prop_id"].map(maps["prop_ctr"]).fillna(maps["fallback_ctr"])
    df["prop_book_rate"] = df["prop_id"].map(maps["prop_book_rate"]).fillna(maps["fallback_book"])
    df["prop_mean_position"] = df["prop_id"].map(maps["prop_mean_position"]).fillna(maps["fallback_pos"])

    idx = pd.MultiIndex.from_frame(df[["srch_destination_id", "prop_id"]])
    dest_scores = maps["dest_prop_score"].reindex(idx).to_numpy()
    df["dest_prop_score"] = np.where(np.isnan(dest_scores), maps["fallback_dest"], dest_scores)

    #destination-star affinity
    idx_ds = pd.MultiIndex.from_frame(df[["srch_destination_id", "prop_starrating"]])
    ds_scores = maps["dest_star_score"].reindex(idx_ds).to_numpy()
    df["dest_star_score"] = np.where(np.isnan(ds_scores), maps["fallback_dest_star"], ds_scores)

    #overall destination popularity
    dest_pop = maps["dest_popularity"].reindex(df["srch_destination_id"]).to_numpy()
    df["dest_popularity"] = np.where(np.isnan(dest_pop), maps["fallback_dest_pop"], dest_pop)
    return df


def make_features(df, target_maps):
    df = add_price_features(df)
    df = add_competitor_features(df)
    df = add_within_query_features(df)
    df = add_visitor_history_features(df)
    df = add_date_features(df)
    df = apply_target_maps(df, target_maps)
    return df


#data loading and split


def load_training():
    print("Loading training set...")
    df = pd.read_csv(TRAIN_PATH, usecols=train_usecols())
    df["relevance"] = relevance_label(df)
    print(f"  loaded {len(df):,} rows, {df['srch_id'].nunique():,} searches")
    return df


def split_by_search(df, frac_train=0.8):
    rng = np.random.default_rng(RANDOM_STATE)
    unique = df["srch_id"].unique()
    rng.shuffle(unique)
    cut = int(frac_train * len(unique))
    train_ids = set(unique[:cut])
    train_df = df[df["srch_id"].isin(train_ids)].copy()
    valid_df = df[~df["srch_id"].isin(train_ids)].copy()
    return train_df, valid_df


def group_sizes(df):
    #df must already be sorted by srch_id
    return df.groupby("srch_id", sort=False).size().to_numpy()


#modeling


def make_ranker(n_estimators):
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
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def fit_with_early_stopping(train_df, valid_df, sample_weight=None):
    train_df = train_df.sort_values("srch_id").reset_index(drop=True)
    valid_df = valid_df.sort_values("srch_id").reset_index(drop=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["relevance"]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["relevance"]

    train_grp = group_sizes(train_df)
    valid_grp = group_sizes(valid_df)

    ranker = make_ranker(n_estimators=1500)
    ranker.fit(
        X_train,
        y_train,
        group=train_grp,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_grp],
        eval_at=[5],
        sample_weight=sample_weight,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )
    return ranker, train_df, valid_df


def fit_full(full_df, n_estimators, sample_weight=None):
    full_df = full_df.sort_values("srch_id").reset_index(drop=True)
    X = full_df[FEATURE_COLUMNS]
    y = full_df["relevance"]
    grp = group_sizes(full_df)
    ranker = make_ranker(n_estimators=n_estimators)
    ranker.fit(X, y, group=grp, sample_weight=sample_weight)
    return ranker


def score_test(ranker, target_maps, out_path):
    print(f"Scoring test set -> {out_path}")
    test_df = pd.read_csv(TEST_PATH, usecols=test_usecols())
    test_df = make_features(test_df, target_maps)
    test_df["score"] = ranker.predict(test_df[FEATURE_COLUMNS])
    submission = (
        test_df[["srch_id", "prop_id", "score"]]
        .sort_values(["srch_id", "score"], ascending=[True, False])
    )
    submission[["srch_id", "prop_id"]].to_csv(out_path, index=False)
    print(f"  wrote {len(submission):,} rows")


#plots


def make_eda_plots(df):
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    missing_pct.plot(kind="bar", color="#2b8cbe")
    plt.title("Top 15 Features by Missingness (%)")
    plt.ylabel("Missing %")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missingness_top15.png", dpi=150)
    plt.close()

    sample = df[["price_usd", "relevance"]].replace([np.inf, -np.inf], np.nan).dropna()
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


def make_importance_plot(ranker):
    importance = pd.DataFrame(
        {
            "feature": ranker.feature_name_,
            "gain": ranker.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=True).tail(20)

    plt.figure(figsize=(8, 8))
    plt.barh(importance["feature"], importance["gain"], color="#2b8cbe")
    plt.title("LightGBM Feature Importance (gain) — Top 20")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_top20.png", dpi=150)
    plt.close()
    return importance


def popularity_baseline_ndcg(valid_df):
    valid_df = valid_df.copy()
    valid_df["score_pop"] = 0.5 * valid_df["dest_prop_score"] + 0.5 * valid_df["prop_ctr"]
    return compute_ndcg_at_k(valid_df, "score_pop", k=5)


def main():
    df = load_training()
    make_eda_plots(df)

    train_df, valid_df = split_by_search(df, frac_train=0.8)

    print("Building target encoding maps on training fold...")
    fold_maps = build_target_maps(train_df)

    print("Engineering features...")
    train_df = make_features(train_df, fold_maps)
    valid_df = make_features(valid_df, fold_maps)

    print("Training UNMITIGATED LambdaMART with early stopping...")
    ranker_unmit, train_sorted, valid_sorted = fit_with_early_stopping(train_df, valid_df)

    valid_sorted["score_unmit"] = ranker_unmit.predict(valid_sorted[FEATURE_COLUMNS])
    ndcg_unmit = compute_ndcg_at_k(valid_sorted, "score_unmit", k=5)
    ndcg_pop = popularity_baseline_ndcg(valid_sorted)
    print(f"  validation NDCG@5: lambdamart={ndcg_unmit:.4f}, popularity={ndcg_pop:.4f}")

    valid_sorted["group"] = family_group(valid_sorted)
    query_group = valid_sorted.groupby("srch_id", as_index=False)["group"].first()
    family_ids = set(query_group.loc[query_group["group"] == "family", "srch_id"])
    non_family_ids = set(query_group.loc[query_group["group"] == "non_family", "srch_id"])
    ndcg_family_before = compute_ndcg_at_k(
        valid_sorted[valid_sorted["srch_id"].isin(family_ids)], "score_unmit", k=5
    )
    ndcg_non_family_before = compute_ndcg_at_k(
        valid_sorted[valid_sorted["srch_id"].isin(non_family_ids)], "score_unmit", k=5
    )

    print("Training MITIGATED LambdaMART (family weight=1.6)...")
    sw_fold = np.where(family_group(train_sorted) == "family", FAMILY_WEIGHT, 1.0)
    ranker_mit, _, _ = fit_with_early_stopping(train_df, valid_df, sample_weight=sw_fold)

    valid_sorted["score_mit"] = ranker_mit.predict(valid_sorted[FEATURE_COLUMNS])
    ndcg_mit = compute_ndcg_at_k(valid_sorted, "score_mit", k=5)
    ndcg_family_after = compute_ndcg_at_k(
        valid_sorted[valid_sorted["srch_id"].isin(family_ids)], "score_mit", k=5
    )
    ndcg_non_family_after = compute_ndcg_at_k(
        valid_sorted[valid_sorted["srch_id"].isin(non_family_ids)], "score_mit", k=5
    )

    importance = make_importance_plot(ranker_unmit)
    importance.tail(20).to_csv(OUTPUT_DIR / "feature_variability_top15.csv", index=False)

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
            "ndcg5_lambdamart": ndcg_unmit,
            "ndcg5_lambdamart_mitigated": ndcg_mit,
            "best_iter_unmit": int(ranker_unmit.best_iteration_ or ranker_unmit.n_estimators),
            "best_iter_mit": int(ranker_mit.best_iteration_ or ranker_mit.n_estimators),
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
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    print("Refitting UNMITIGATED on full training set for Kaggle submission...")
    full_maps = build_target_maps(df)
    full_features_df = make_features(df.copy(), full_maps)
    best_iter_unmit = int(ranker_unmit.best_iteration_ or ranker_unmit.n_estimators)
    final_unmit = fit_full(full_features_df, n_estimators=best_iter_unmit)
    score_test(final_unmit, full_maps, "submission_final.csv")

    print("Refitting MITIGATED on full training set for report fairness numbers...")
    full_sorted = full_features_df.sort_values("srch_id").reset_index(drop=True)
    sw_full = np.where(family_group(full_sorted) == "family", FAMILY_WEIGHT, 1.0)
    best_iter_mit = int(ranker_mit.best_iteration_ or ranker_mit.n_estimators)
    final_mit = fit_full(full_features_df, n_estimators=best_iter_mit, sample_weight=sw_full)
    score_test(final_mit, full_maps, "submission_mitigated.csv")

    print("Done.")


if __name__ == "__main__":
    main()
