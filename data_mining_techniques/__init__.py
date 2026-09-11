"""Reusable feature engineering and evaluation helpers for the coursework project."""

from .core import (
    FEATURE_COLUMNS,
    DataSchemaError,
    build_target_maps,
    compute_ndcg_at_k,
    make_features,
    relevance_label,
    split_by_search,
)

__all__ = [
    "FEATURE_COLUMNS",
    "DataSchemaError",
    "build_target_maps",
    "compute_ndcg_at_k",
    "make_features",
    "relevance_label",
    "split_by_search",
]
