import re
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.impute import KNNImputer, SimpleImputer


# =========================
# CONFIG
# =========================
INPUT_PATH = "ODI-2026.xlsx"   # oppure: "ODI-2026-3.csv"
OUTPUT_DIR = Path("output_task1b")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# IO
# =========================
def read_input(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Formato non supportato: {ext}")


# =========================
# HELPERS
# =========================
def norm_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\xa0", " ").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if s == "":
        return np.nan
    if s.lower() in {"-", "--", "---", "nan", "none", "null"}:
        return np.nan
    return s


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {c: norm_col(c) for c in df.columns}

    patterns = {
        "timestamp": ["tijdstempel"],
        "programme": ["what programme are you in"],
        "ml_course": ["machine learning"],
        "ir_course": ["information retrieval"],
        "stats_course": ["statistics"],
        "db_course": ["databases"],
        "gender": ["gender"],
        "used_llm": ["llms"],
        "birthday_raw": ["birthday"],
        "room_estimate": ["estimate there are in the room"],
        "stress": ["stress level"],
        "sports_hours": ["sports"],
        "random_number": ["random number"],
        "bedtime_raw": ["went to bed"],
        "good_day_1": ["good day for you 1"],
        "good_day_2": ["good day for you 2"],
    }

    rename_map = {}

    for new_name, tokens in patterns.items():
        for old_name, old_norm in normalized.items():
            if all(token in old_norm for token in tokens):
                rename_map[old_name] = new_name
                break

    df = df.rename(columns=rename_map)

    expected_cols = [
        "timestamp", "programme", "ml_course", "ir_course", "stats_course",
        "db_course", "gender", "used_llm", "birthday_raw", "room_estimate",
        "stress", "sports_hours", "random_number", "bedtime_raw",
        "good_day_1", "good_day_2"
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonne non trovate: {missing}")

    return df[expected_cols].copy()


def normalize_binary(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    s = s.lower().strip()

    yes_values = {"yes", "y", "ja", "true", "1"}
    no_values = {"no", "n", "nee", "false", "0"}
    missing_values = {"unknown", "not sure", "dont know", "don't know"}

    if s in yes_values:
        return "yes"
    if s in no_values:
        return "no"
    if s in missing_values:
        return np.nan
    return s


def normalize_gender(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    s = s.lower().strip()

    mapping = {
        "male": "male",
        "female": "female",
        "non-binary": "non-binary",
        "gender fluid": "gender-fluid",
        "genderfluid": "gender-fluid",
        "not willing to say": "prefer-not-to-say",
        "not willing to answer": "prefer-not-to-say",
        "unknown": np.nan,
    }

    return mapping.get(s, s)


def normalize_programme(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan

    s_low = s.lower()

    patterns = [
        (r"\b(ai|artificial intelligence|master ai|masters ai|msc ai|ai master)\b", "Artificial Intelligence"),
        (r"\b(computer science|cs|computer science vuuva|computer science vu uva|computer science ai|master in computer science|masters in computer science|masters computer science|msc computer science|m computer science)\b", "Computer Science"),
        (r"\b(business analytics|ba|master of business analytics|ms business analytics)\b", "Business Analytics"),
        (r"\b(computational science|computational sciences|computational scieneces)\b", "Computational Science"),
        (r"\b(bioinformatics|systems biology|bioinformatics and systems biology|bioinformatics systems biology|bioinformatics and system biology|systems biology bioinformatics)\b", "Bioinformatics and Systems Biology"),
        (r"\b(econometrics|econometrics data science|econometrics and data science)\b", "Econometrics / Data Science"),
        (r"\b(finance and technology|finance technology|fintech|quant finance|quantitative finance|corporate finance|finance)\b", "Finance / FinTech"),
        (r"\b(data mining|big data engineering|bde)\b", "Data / Engineering"),
    ]

    for pattern, label in patterns:
        if re.search(pattern, s_low):
            return label

    return s.strip()


def parse_strict_number(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    s = s.replace(",", ".").strip()

    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return np.nan


def parse_number_with_noise(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    s = s.replace(",", ".").strip()

    match = re.search(r"-?\d+(\.\d+)?", s)
    if match:
        return float(match.group())
    return np.nan


def parse_sports_hours(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    s_low = s.lower().strip()

    if re.fullmatch(r"-?\d+", s_low):
        return float(s_low)

    if re.fullmatch(r"-?\d+(\.\d+)?\s*(hr|hrs|hour|hours)", s_low):
        return float(re.search(r"-?\d+(\.\d+)?", s_low).group())

    if re.fullmatch(r"-?\d+\s*-\s*-?\d+(\s*(hr|hrs|hour|hours))?", s_low):
        return np.nan

    return np.nan


def parse_bedtime_to_hour(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan

    s_low = s.lower().strip()

    invalid_phrases = {
        "pulled all nighter",
        "all nighter",
        "rond 1 uur",
        "no",
        "yes",
    }
    if s_low in invalid_phrases:
        return np.nan

    s_low = s_low.replace("🫶", "").strip()

    if re.fullmatch(r"\d{3,4}", s_low):
        s_low = s_low.zfill(4)
        s_low = f"{s_low[:2]}:{s_low[2:]}"

    if re.fullmatch(r"\d{1,2}\.\d{2}", s_low):
        s_low = s_low.replace(".", ":")

    if re.fullmatch(r"\d{1,2}", s_low):
        hh = int(s_low)
        if 0 <= hh <= 23:
            return float(hh)
        return np.nan

    try:
        if s_low in {"24:00", "2400"}:
            return 0.0

        dt = parser.parse(s_low, fuzzy=True)
        hh = dt.hour
        mm = dt.minute

        if hh == 24 and mm == 0:
            return 0.0
        if hh == 24 and mm > 0:
            return np.nan

        return hh + mm / 60.0
    except Exception:
        return np.nan


def parse_birth_year(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan

    s_low = s.lower().strip()

    if s_low in {"not willing to say", "not willing to answer", "unknown", "yes", "no"}:
        return np.nan
    if s_low in {"big bang"}:
        return np.nan

    if re.fullmatch(r"\d{4}", s_low):
        year = int(s_low)
        if 1960 <= year <= 2010:
            return float(year)
        return np.nan

    digit_groups = re.findall(r"\d+", s_low)
    has_year_like = any(len(g) == 4 for g in digit_groups) or re.search(r"\b\d{2}\b", s_low)

    if not has_year_like:
        return np.nan

    try:
        dt = parser.parse(s_low, dayfirst=True, fuzzy=True)
        if 1960 <= dt.year <= 2010:
            return float(dt.year)
        return np.nan
    except Exception:
        return np.nan


def parse_timestamp(x):
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def iqr_outliers_to_nan(series: pd.Series, k: float = 1.5):
    s = series.copy()
    valid = s.dropna()
    if len(valid) < 4:
        return s

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return s.where(s.between(lower, upper), np.nan)


# =========================
# LOAD + RENAME
# =========================
df = read_input(INPUT_PATH)
df = rename_columns(df)


# =========================
# BASIC STANDARDIZATION
# =========================
for col in ["programme", "gender", "good_day_1", "good_day_2", "birthday_raw", "bedtime_raw"]:
    df[col] = df[col].apply(clean_text)

for col in ["ml_course", "ir_course", "stats_course", "db_course", "used_llm"]:
    df[col] = df[col].apply(normalize_binary)

df["programme"] = df["programme"].apply(normalize_programme)
df["gender"] = df["gender"].apply(normalize_gender)
df["timestamp"] = df["timestamp"].apply(parse_timestamp)


# =========================
# PARSE STRUCTURED VARIABLES
# =========================
df["birth_year"] = df["birthday_raw"].apply(parse_birth_year)
df["room_estimate"] = df["room_estimate"].apply(parse_strict_number)
df["stress"] = df["stress"].apply(parse_number_with_noise)
df["sports_hours"] = df["sports_hours"].apply(parse_sports_hours)
df["random_number"] = df["random_number"].apply(parse_strict_number)
df["bedtime_hours"] = df["bedtime_raw"].apply(parse_bedtime_to_hour)


# =========================
# DOMAIN RULES
# =========================
# stress deve stare tra 0 e 100
df.loc[~df["stress"].between(0, 100, inclusive="both"), "stress"] = np.nan

# sport non può essere negativo
df.loc[df["sports_hours"] < 0, "sports_hours"] = np.nan

# bedtime espresso come ora [0, 24)
df.loc[(df["bedtime_hours"] < 0) | (df["bedtime_hours"] >= 24), "bedtime_hours"] = np.nan

# room estimate non può essere negativo
df.loc[df["room_estimate"] < 0, "room_estimate"] = np.nan


# =========================
# IQR OUTLIERS -> NaN
# =========================
for col in ["room_estimate", "sports_hours"]:
    df[col] = iqr_outliers_to_nan(df[col], k=1.5)


# =========================
# OPTIONAL: keep only cleaned structured columns
# =========================
clean_df = df[
    [
        "timestamp",
        "programme",
        "ml_course",
        "ir_course",
        "stats_course",
        "db_course",
        "gender",
        "used_llm",
        "birth_year",
        "room_estimate",
        "stress",
        "sports_hours",
        "random_number",
        "bedtime_hours",
        "good_day_1",
        "good_day_2",
    ]
].copy()


# =========================
# REPORT BEFORE IMPUTATION
# =========================
def missing_report(dataframe: pd.DataFrame) -> pd.Series:
    return dataframe.isna().sum().sort_values(ascending=False)


report_before = missing_report(clean_df)


# =========================
# IMPUTATION SETUP
# =========================
numeric_cols = [
    "birth_year",
    "room_estimate",
    "stress",
    "sports_hours",
    "random_number",
    "bedtime_hours",
]

categorical_cols = [
    "programme",
    "ml_course",
    "ir_course",
    "stats_course",
    "db_course",
    "gender",
    "used_llm",
]

text_cols = ["good_day_1", "good_day_2"]
time_cols = ["timestamp"]


# =========================
# APPROACH 1: SIMPLE IMPUTATION
# median for numeric + most_frequent for categorical/text
# =========================
simple_df = clean_df.copy()

num_imp = SimpleImputer(strategy="median")
cat_imp = SimpleImputer(strategy="most_frequent")

simple_df[numeric_cols] = num_imp.fit_transform(simple_df[numeric_cols])
simple_df[categorical_cols] = cat_imp.fit_transform(simple_df[categorical_cols])
simple_df[text_cols] = cat_imp.fit_transform(simple_df[text_cols])

# timestamp lo lascio invariato; se vuoi puoi anche escluderlo dai modelli


# =========================
# APPROACH 2: KNN IMPUTATION
# KNN for numeric + most_frequent for categorical/text
# =========================
knn_df = clean_df.copy()

knn_imp = KNNImputer(n_neighbors=5, weights="distance")
knn_df[numeric_cols] = knn_imp.fit_transform(knn_df[numeric_cols])
knn_df[categorical_cols] = cat_imp.fit_transform(knn_df[categorical_cols])
knn_df[text_cols] = cat_imp.fit_transform(knn_df[text_cols])


# =========================
# SAVE
# =========================
clean_df.to_csv(OUTPUT_DIR / "clean_no_impute.csv", index=False)
simple_df.to_csv(OUTPUT_DIR / "clean_imputed_simple.csv", index=False)
knn_df.to_csv(OUTPUT_DIR / "clean_imputed_knn.csv", index=False)

with open(OUTPUT_DIR / "cleaning_report.txt", "w", encoding="utf-8") as f:
    f.write("=== SHAPE ===\n")
    f.write(f"raw rows: {len(df)}\n")
    f.write(f"clean rows: {len(clean_df)}\n\n")

    f.write("=== MISSING BEFORE IMPUTATION ===\n")
    f.write(report_before.to_string())
    f.write("\n\n")

    f.write("=== MISSING AFTER SIMPLE IMPUTATION ===\n")
    f.write(missing_report(simple_df).to_string())
    f.write("\n\n")

    f.write("=== MISSING AFTER KNN IMPUTATION ===\n")
    f.write(missing_report(knn_df).to_string())
    f.write("\n")

print("Fatto.")
print(f"File salvati in: {OUTPUT_DIR.resolve()}")
print("\nMissing values prima dell'imputazione:")
print(report_before)