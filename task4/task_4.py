import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
INPUT_PATH = Path("task1b/clean_imputed_simple.csv")
OUTPUT_DIR = Path("task4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def group_programme(value):
    """Group rare programmes into 'Other' to avoid sparse one-hot columns."""
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
    """Group rare genders into 'Other' to avoid sparse one-hot columns."""
    keep = ["male", "female"]
    if value in keep:
        return value
    return "other"


def build_features(df):
    data = df.copy()

    #parse timestamp into numeric features
    ts = pd.to_datetime(data["timestamp"], errors="coerce")
    data["timestamp_hour"] = ts.dt.hour
    data["timestamp_weekday"] = ts.dt.dayofweek

    #group rare categories
    data["programme"] = data["programme"].apply(group_programme)
    data["gender"] = data["gender"].apply(group_gender)

    #encode stats_course (mu/sigma) as binary
    data["stats_course"] = data["stats_course"].apply(
        lambda x: 1 if x == "mu" else 0
    )

    #encode binary yes/no columns as 0/1
    binary_cols = ["ml_course", "ir_course", "db_course", "used_llm"]
    for col in binary_cols:
        data[col] = data[col].apply(lambda x: 1 if x == "yes" else 0)

    #drop columns not useful for regression
    #random_number is noise (asked for 1-10, most entries are garbage/outliers)
    cols_to_drop = ["timestamp", "good_day_1", "good_day_2", "random_number"]
    data = data.drop(columns=cols_to_drop)

    return data


def evaluate_regression(y_true, y_pred):
    results = {}
    results["mse"] = float(mean_squared_error(y_true, y_pred))
    results["rmse"] = float(np.sqrt(results["mse"]))
    results["mae"] = float(mean_absolute_error(y_true, y_pred))
    results["r2"] = float(r2_score(y_true, y_pred))
    return results


def main():
    #load the cleaned dataset
    df = pd.read_csv(INPUT_PATH)
    df = build_features(df)

    #target: stress level (0-100)
    target_col = "stress"
    y = df[target_col]
    x = df.drop(columns=[target_col])

    #identify column types
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = []
    for col in x.columns:
        if col not in numeric_cols:
            categorical_cols.append(col)

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    #preprocessing pipeline
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

    #train/test split (80/20)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    #cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    #define models and their hyperparameter grids
    models = {
        "ridge": {
            "pipeline": Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", Ridge()),
            ]),
            "param_grid": {
                "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "regressor__n_estimators": [50, 100, 200],
                "regressor__max_depth": [None, 5, 10, 15],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
            },
        },
    }

    #store results
    summary = {
        "dataset": {
            "n_rows": int(df.shape[0]),
            "n_features_before_encoding": int(x.shape[1]),
            "target": target_col,
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "train_size": int(x_train.shape[0]),
            "test_size": int(x_test.shape[0]),
        },
        "models": {},
    }

    #train and evaluate each model
    for model_name, config in models.items():
        print(f"\nTraining {model_name}...")

        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(x_train, y_train)
        best_model = search.best_estimator_

        #predict on test set
        y_pred = best_model.predict(x_test)

        #compute metrics
        test_metrics = evaluate_regression(y_test.to_numpy(), y_pred)

        #cv score (convert from negative MSE)
        cv_mse = -search.best_score_
        cv_rmse = float(np.sqrt(cv_mse))

        summary["models"][model_name] = {
            "best_params": search.best_params_,
            "cv_mse": float(cv_mse),
            "cv_rmse": cv_rmse,
            "test_metrics": test_metrics,
        }

        print(f"  Best params: {search.best_params_}")
        print(f"  CV RMSE:  {cv_rmse:.3f}")
        print(f"  Test MSE: {test_metrics['mse']:.3f}")
        print(f"  Test RMSE:{test_metrics['rmse']:.3f}")
        print(f"  Test MAE: {test_metrics['mae']:.3f}")
        print(f"  Test R2:  {test_metrics['r2']:.3f}")

    #save results to JSON
    out_path = OUTPUT_DIR / "task4_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
