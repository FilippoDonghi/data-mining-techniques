import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv("clean_imputed_simple.csv")

# =========================
# 1) AGE
# =========================
# Transform birth year into age for better interpretability
df["age"] = 2026 - df["birth_year"]

# =========================
# 2) COURSES_COUNT
# =========================
# Count how many technical courses a student has taken,
# excluding ir_course because it is the target variable
yes_no_map = {"yes": 1, "no": 0}

df["ml_course_bin"] = df["ml_course"].map(yes_no_map)
df["db_course_bin"] = df["db_course"].map(yes_no_map)

# stats_course contains values such as mu/sigma,
# so we convert them into a binary indicator
df["stats_course_bin"] = df["stats_course"].apply(
    lambda x: 1 if str(x).strip().lower() in ["mu", "sigma", "yes"] else 0
)

df["courses_count"] = (
    df["ml_course_bin"] +
    df["db_course_bin"] +
    df["stats_course_bin"]
)

# =========================
# 3) LATE_BED
# =========================
# Binary feature: 1 if the student goes to bed after 1:30 AM
df["late_bed"] = (df["bedtime_hours"] > 1.5).astype(int)

# =========================
# 4) SPORT_STRESS_RATIO
# =========================
# Combine sports activity and stress level
# +1 avoids division by zero
df["sport_stress_ratio"] = df["sports_hours"] / (df["stress"] + 1)

# =========================
# Optional binary target for Task 2A
# =========================
df["ir_course_target"] = df["ir_course"].map(yes_no_map)

# Quick check
print(df[[
    "birth_year", "age",
    "ml_course", "db_course", "stats_course", "courses_count",
    "bedtime_hours", "late_bed",
    "sports_hours", "stress", "sport_stress_ratio",
    "ir_course", "ir_course_target"
]].head())

# Save the feature-engineered dataset
df.to_csv("clean_imputed_simple_feature_engineered.csv", index=False)