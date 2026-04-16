import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
INPUT_PATH = Path("task1b/clean_imputed_simple.csv")
OUTPUT_DIR = Path("task2a")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Use temporal information in a model-friendly numeric format.
    ts = pd.to_datetime(data["timestamp"], errors="coerce")
    data["timestamp_hour"] = ts.dt.hour
    data["timestamp_weekday"] = ts.dt.dayofweek

    # Free-text fields are excluded for this baseline classification setup.
    data = data.drop(columns=["timestamp", "good_day_1", "good_day_2"])

    return data


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = build_features(df)

    target_map = {"yes": 1, "no": 0}
    y = df["ir_course"].map(target_map)
    x = df.drop(columns=["ir_course"])

    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "decision_tree": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            "param_grid": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [None, 3, 5, 7, 10],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        },
        "knn": {
            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            "param_grid": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        },
    }

    summary = {
        "dataset": {
            "n_rows": int(df.shape[0]),
            "n_features_before_encoding": int(x.shape[1]),
            "target_distribution": {
                "no": int((y == 0).sum()),
                "yes": int((y == 1).sum()),
            },
            "train_size": int(x_train.shape[0]),
            "test_size": int(x_test.shape[0]),
        },
        "models": {},
    }

    for model_name, config in models.items():
        search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["param_grid"],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(x_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(x_test)
        y_proba = None
        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(x_test)[:, 1]

        summary["models"][model_name] = {
            "best_params": search.best_params_,
            "cv_best_f1": float(search.best_score_),
            "test_metrics": evaluate_model(y_test.to_numpy(), y_pred, y_proba),
        }

    out_path = OUTPUT_DIR / "task2a_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
