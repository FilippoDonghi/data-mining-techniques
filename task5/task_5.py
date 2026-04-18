import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
INPUT_PATH = Path("task1b/clean_imputed_simple.csv")
TASK4_RESULTS = Path("task4/task4_results.json")
OUTPUT_DIR = Path("task5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def group_programme(value):
    keep = [
        "Artificial Intelligence",
        "Computer Science",
        "Business Analytics",
        "Bioinformatics and Systems Biology",
        "Finance / FinTech",
        "Computational Science",
        "Econometrics / Data Science",
    ]
    if value in keep:
        return value
    return "Other"


def group_gender(value):
    keep = ["male", "female"]
    if value in keep:
        return value
    return "other"


def build_features(df):
    data = df.copy()

    ts = pd.to_datetime(data["timestamp"], errors="coerce")
    data["timestamp_hour"] = ts.dt.hour
    data["timestamp_weekday"] = ts.dt.dayofweek

    data["programme"] = data["programme"].apply(group_programme)
    data["gender"] = data["gender"].apply(group_gender)

    data["stats_course"] = data["stats_course"].apply(
        lambda x: 1 if x == "mu" else 0
    )

    binary_cols = ["ml_course", "ir_course", "db_course", "used_llm"]
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x == "yes" else 0)

    cols_to_drop = ["timestamp", "good_day_1", "good_day_2", "random_number"]
    data = data.drop(columns=cols_to_drop)

    return data


def main():
    #load and prepare data (same as task 4)
    df = pd.read_csv(INPUT_PATH)
    df = build_features(df)

    target_col = "stress"
    y = df[target_col]
    x = df.drop(columns=[target_col])

    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = []
    for col in x.columns:
        if col not in numeric_cols:
            categorical_cols.append(col)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE,
    )

    #preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    #load best params from task 4
    with open(TASK4_RESULTS, "r") as f:
        task4 = json.load(f)

    #retrain ridge with best params
    ridge_alpha = task4["models"]["ridge"]["best_params"]["regressor__alpha"]
    ridge_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=ridge_alpha)),
    ])
    ridge_pipe.fit(x_train, y_train)

    #retrain random forest with best params
    rf_params = task4["models"]["random_forest"]["best_params"]
    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=rf_params["regressor__n_estimators"],
            max_depth=rf_params["regressor__max_depth"],
            min_samples_split=rf_params["regressor__min_samples_split"],
            min_samples_leaf=rf_params["regressor__min_samples_leaf"],
            random_state=RANDOM_STATE,
        )),
    ])
    rf_pipe.fit(x_train, y_train)

    #predictions
    ridge_pred = ridge_pipe.predict(x_test)
    rf_pred = rf_pipe.predict(x_test)
    y_actual = y_test.to_numpy()

    #per-sample absolute errors
    ridge_errors = np.abs(y_actual - ridge_pred)
    rf_errors = np.abs(y_actual - rf_pred)

    #per-sample squared errors
    ridge_sq_errors = (y_actual - ridge_pred) ** 2
    rf_sq_errors = (y_actual - rf_pred) ** 2

    #compute metrics
    results = {}
    for name, errors, sq_errors in [
        ("ridge", ridge_errors, ridge_sq_errors),
        ("random_forest", rf_errors, rf_sq_errors),
    ]:
        mae = float(np.mean(errors))
        mse = float(np.mean(sq_errors))
        rmse = float(np.sqrt(mse))

        #how much the largest errors inflate MSE vs MAE
        sorted_errors = np.sort(errors)[::-1]
        top_5_abs = sorted_errors[:5]
        top_5_contribution_mse = float(np.sum(top_5_abs ** 2) / np.sum(sq_errors) * 100)
        top_5_contribution_mae = float(np.sum(top_5_abs) / np.sum(errors) * 100)

        results[name] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mse_minus_mae_squared": float(mse - mae ** 2),
            "top_5_errors": top_5_abs.tolist(),
            "top_5_pct_of_mse": top_5_contribution_mse,
            "top_5_pct_of_mae": top_5_contribution_mae,
            "error_std": float(np.std(errors)),
            "error_median": float(np.median(errors)),
            "error_max": float(np.max(errors)),
        }

        print(f"\n=== {name.upper()} ===")
        print(f"  MSE:  {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE:  {mae:.2f}")
        print(f"  Median absolute error: {np.median(errors):.2f}")
        print(f"  Max absolute error:    {np.max(errors):.2f}")
        print(f"  Top 5 errors contribute {top_5_contribution_mse:.1f}% of MSE")
        print(f"  Top 5 errors contribute {top_5_contribution_mae:.1f}% of MAE")

    #save results
    out_path = OUTPUT_DIR / "task5_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")

    #--- FIGURE: per-sample errors sorted, showing MSE vs MAE behavior ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #ridge
    sorted_idx = np.argsort(ridge_errors)[::-1]
    sample_positions = np.arange(len(sorted_idx))

    axes[0].bar(
        sample_positions, ridge_errors[sorted_idx],
        color="steelblue", alpha=0.7, label="Absolute error (MAE uses these)"
    )
    axes[0].bar(
        sample_positions, ridge_sq_errors[sorted_idx] / np.max(ridge_sq_errors) * np.max(ridge_errors),
        color="navy", alpha=0.3, label="Squared error (scaled)"
    )
    axes[0].axhline(
        np.mean(ridge_errors), color="black", linestyle="--", linewidth=1,
        label=f"MAE = {np.mean(ridge_errors):.1f}"
    )
    axes[0].set_xlabel("Test samples (sorted by error)")
    axes[0].set_ylabel("Error magnitude")
    axes[0].set_title("Ridge")
    axes[0].legend(fontsize=8)

    #random forest
    sorted_idx_rf = np.argsort(rf_errors)[::-1]

    axes[1].bar(
        sample_positions, rf_errors[sorted_idx_rf],
        color="coral", alpha=0.7, label="Absolute error (MAE uses these)"
    )
    axes[1].bar(
        sample_positions, rf_sq_errors[sorted_idx_rf] / np.max(rf_sq_errors) * np.max(rf_errors),
        color="darkred", alpha=0.3, label="Squared error (scaled)"
    )
    axes[1].axhline(
        np.mean(rf_errors), color="black", linestyle="--", linewidth=1,
        label=f"MAE = {np.mean(rf_errors):.1f}"
    )
    axes[1].set_xlabel("Test samples (sorted by error)")
    axes[1].set_ylabel("Error magnitude")
    axes[1].set_title("Random Forest")
    axes[1].legend(fontsize=8)

    fig.suptitle("Per-sample Errors: How Large Errors Inflate MSE More Than MAE", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task5_error_breakdown.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: task5_error_breakdown.png")


if __name__ == "__main__":
    main()
