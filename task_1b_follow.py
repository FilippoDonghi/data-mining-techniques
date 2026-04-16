import re
import numpy as np
import pandas as pd


# =========================
# INPUT / OUTPUT
# =========================
df = pd.read_csv("output_task1b/clean_no_impute.csv")
# se vuoi partire dall'xlsx originale, meglio ancora,
# ma questo script lavora direttamente sul tuo clean_no_impute.csv


# =========================
# HELPER
# =========================
def clean_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    if s == "":
        return np.nan
    return s


def normalize_binary(x):
    x = clean_text(x)
    if pd.isna(x):
        return np.nan

    s = x.lower()

    yes_values = {"yes", "ja", "y", "true", "1"}
    no_values = {"no", "nee", "n", "false", "0"}

    missing_values = {
        "unknown",
        "not willing to say",
        "not willing to answer",
        "prefer-not-to-say",
        "nan",
        "none"
    }

    if s in yes_values:
        return "yes"
    if s in no_values:
        return "no"
    if s in missing_values:
        return np.nan

    # codici sporchi che non vuoi tenere come categorie
    if s in {"mu", "sigma"}:
        return np.nan

    return np.nan


def normalize_programme(x):
    x = clean_text(x)
    if pd.isna(x):
        return np.nan

    s = x.lower()

    if any(k in s for k in ["ai", "artificial intelligence"]):
        return "Artificial Intelligence"

    if "computer science" in s or s == "cs":
        return "Computer Science"

    if "business analytics" in s or s == "ba":
        return "Business Analytics"

    if "computational science" in s:
        return "Computational Science"

    if "bioinformatics" in s or "systems biology" in s:
        return "Bioinformatics and Systems Biology"

    if "econometrics" in s or "data science" in s:
        return "Econometrics / Data Science"

    if "finance" in s or "fintech" in s or "quant" in s:
        return "Finance / FinTech"

    if "data mining" in s or "engineering" in s:
        return "Data / Engineering"

    if "computer security" in s or "security" in s:
        return "Computer Security"

    if s in {"msc", "master", "masters", "msc.", "m"}:
        return "Other"

    if s in {"mpa", "philosophy"}:
        return "Other"

    return "Other"


def normalize_gender(x):
    x = clean_text(x)
    if pd.isna(x):
        return np.nan

    s = x.lower()

    if s == "male":
        return "male"
    if s == "female":
        return "female"
    if s in {"non-binary", "non binary"}:
        return "non-binary"
    if s in {"gender-fluid", "gender fluid"}:
        return "gender-fluid"
    if s in {"prefer-not-to-say", "not willing to say", "not willing to answer"}:
        return "prefer-not-to-say"

    return np.nan


def clean_birth_year(x):
    if pd.isna(x):
        return np.nan

    try:
        year = float(x)
    except:
        return np.nan

    # range ampio ma plausibile per studenti adulti / casi eccezionali
    if 1970 <= year <= 2010:
        return year

    return np.nan


def clean_room_estimate(x):
    if pd.isna(x):
        return np.nan

    try:
        val = float(x)
    except:
        return np.nan

    if val < 0:
        return np.nan

    # se vuoi essere conservativo, mantieni tutto il plausibile
    if 0 <= val <= 1000:
        return val

    return np.nan


def clean_stress(x):
    if pd.isna(x):
        return np.nan

    try:
        val = float(x)
    except:
        return np.nan

    if 0 <= val <= 100:
        return val

    return np.nan


def clean_sports_hours(x):
    if pd.isna(x):
        return np.nan

    try:
        val = float(x)
    except:
        return np.nan

    # qui puoi essere un po' più severo: ore a settimana plausibili
    if 0 <= val <= 40:
        return val

    return np.nan


def clean_bedtime_hours(x):
    if pd.isna(x):
        return np.nan

    try:
        val = float(x)
    except:
        return np.nan

    if 0 <= val < 24:
        return val

    return np.nan


def clean_random_number(x):
    if pd.isna(x):
        return np.nan

    try:
        val = float(x)
    except:
        return np.nan

    # scelta consigliata: rendere missing valori chiaramente ingestibili
    # ma senza cercare di "normalizzare" davvero un numero casuale
    if np.isinf(val):
        return np.nan

    # soglia pratica per tagliare scherzi / numeri inutilizzabili
    if abs(val) > 1_000_000:
        return np.nan

    return val


def collapse_rare_categories(series, min_count=5, other_label="Other"):
    counts = series.value_counts(dropna=True)
    rare = counts[counts < min_count].index
    return series.apply(lambda x: other_label if x in rare else x)


# =========================
# CLEAN CATEGORICALS
# =========================
binary_cols = ["ml_course", "ir_course", "stats_course", "db_course", "used_llm"]

for col in binary_cols:
    df[col] = df[col].apply(normalize_binary)

df["programme"] = df["programme"].apply(normalize_programme)
df["gender"] = df["gender"].apply(normalize_gender)

# opzionale: accorpa programmi troppo rari
df["programme"] = collapse_rare_categories(df["programme"], min_count=5, other_label="Other")


# =========================
# CLEAN NUMERICALS
# =========================
df["birth_year"] = df["birth_year"].apply(clean_birth_year)
df["room_estimate"] = df["room_estimate"].apply(clean_room_estimate)
df["stress"] = df["stress"].apply(clean_stress)
df["sports_hours"] = df["sports_hours"].apply(clean_sports_hours)
df["bedtime_hours"] = df["bedtime_hours"].apply(clean_bedtime_hours)
df["random_number"] = df["random_number"].apply(clean_random_number)


# =========================
# OPTIONAL: DROP VERY NOISY COLUMNS
# =========================
# Se NON vuoi usare testo libero e random_number nei modelli:
# df = df.drop(columns=["good_day_1", "good_day_2", "random_number"])

# Se vuoi tenerli nel dataset cleaned ma non nei modelli, lasciali pure.


# =========================
# CHECK
# =========================
print("\n=== UNIQUE VALUES (categorical) ===")
for col in ["programme", "ml_course", "ir_course", "stats_course", "db_course", "gender", "used_llm"]:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False))

print("\n=== MISSING VALUES ===")
print(df.isna().sum().sort_values(ascending=False))

print("\n=== NUMERIC SUMMARY ===")
print(df[["birth_year", "room_estimate", "stress", "sports_hours", "random_number", "bedtime_hours"]].describe())


# =========================
# SAVE
# =========================
df.to_csv("clean_no_impute_fixed.csv", index=False)
print("\nFile salvato come: clean_no_impute_fixed.csv")