import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
INPUT_PATH = Path("task1b/clean_imputed_simple.csv")
OUTPUT_DIR = Path("task4")
RESULTS_PATH = OUTPUT_DIR / "task4_results.json"


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


def train_models(x_train, y_train, numeric_cols, categorical_cols):
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

    #load best params from results
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    #ridge
    ridge_params = results["models"]["ridge"]["best_params"]
    ridge_alpha = ridge_params["regressor__alpha"]
    ridge_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=ridge_alpha)),
    ])
    ridge_pipe.fit(x_train, y_train)

    #random forest
    rf_params = results["models"]["random_forest"]["best_params"]
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

    return ridge_pipe, rf_pipe


def main():
    #load and prepare data
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

    #train both models with best params
    ridge_pipe, rf_pipe = train_models(x_train, y_train, numeric_cols, categorical_cols)

    #predictions
    ridge_pred = ridge_pipe.predict(x_test)
    rf_pred = rf_pipe.predict(x_test)

    #--- FIGURE 1: actual vs predicted scatter for both models ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #ridge
    axes[0].scatter(y_test, ridge_pred, alpha=0.6, color="steelblue", edgecolor="white")
    axes[0].plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Actual Stress")
    axes[0].set_ylabel("Predicted Stress")
    axes[0].set_title("Ridge Regression")
    axes[0].set_xlim(-5, 105)
    axes[0].set_ylim(-5, 105)

    #random forest
    axes[1].scatter(y_test, rf_pred, alpha=0.6, color="coral", edgecolor="white")
    axes[1].plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Actual Stress")
    axes[1].set_ylabel("Predicted Stress")
    axes[1].set_title("Random Forest")
    axes[1].set_xlim(-5, 105)
    axes[1].set_ylim(-5, 105)

    fig.suptitle("Actual vs Predicted Stress Level", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task4_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: task4_actual_vs_predicted.png")

    #--- FIGURE 2: residual distribution for both models ---
    ridge_residuals = y_test.to_numpy() - ridge_pred
    rf_residuals = y_test.to_numpy() - rf_pred

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    #ridge residuals
    axes2[0].hist(ridge_residuals, bins=12, color="steelblue", edgecolor="white")
    axes2[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes2[0].set_xlabel("Residual (Actual - Predicted)")
    axes2[0].set_ylabel("Frequency")
    axes2[0].set_title(f"Ridge (MAE={np.mean(np.abs(ridge_residuals)):.1f})")

    #random forest residuals
    axes2[1].hist(rf_residuals, bins=12, color="coral", edgecolor="white")
    axes2[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes2[1].set_xlabel("Residual (Actual - Predicted)")
    axes2[1].set_ylabel("Frequency")
    axes2[1].set_title(f"Random Forest (MAE={np.mean(np.abs(rf_residuals)):.1f})")

    fig2.suptitle("Residual Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task4_residuals.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: task4_residuals.png")


if __name__ == "__main__":
    main()
