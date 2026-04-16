import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── load data ────────────────────────────────────────────────────────────────

df = pd.read_csv("ODI-2026.csv", sep=";")

df.columns = [
    "timestamp", "programme", "ml_course", "ir_course", "stats_course",
    "db_course", "gender", "llm_used", "birthday", "student_estimate",
    "stress_level", "sport_hours", "random_number", "bedtime",
    "good_day_1", "good_day_2"
]
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")


# ── light cleaning for EDA (original data untouched) ────────────────────────

def extract_first_number(series):
    """Pull the first numeric value out of a free-text column."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"([-]?\d+(?:\.\d+)?)")[0],
        errors="coerce"
    )

stress  = extract_first_number(df["stress_level"])
sport   = extract_first_number(df["sport_hours"])
estim   = extract_first_number(df["student_estimate"])
randnum = extract_first_number(df["random_number"])

# Keep only logically plausible values (outlier flagging, not removal yet)
stress_valid  = stress[stress.between(0, 100)]
sport_valid   = sport[sport.between(0, 40)]
randnum_valid = randnum[randnum.between(1, 10)]

# Normalise binary course columns (ja/nee → yes/no)
def normalise_binary(col):
    mapping = {"ja": "yes", "1": "yes", "nee": "no", "0": "no"}
    s = df[col].astype(str).str.strip().str.lower().replace(mapping)
    return s.where(s.isin(["yes", "no", "not willing to say"]), other="unknown")

ml  = normalise_binary("ml_course")
ir  = normalise_binary("ir_course")
db  = normalise_binary("db_course")
llm = normalise_binary("llm_used")

# Group free-text programmes into broad categories
def group_programme(p):
    p = str(p).lower()
    if any(x in p for x in ["ai", "artificial intelligence"]):
        return "AI"
    if "business analytics" in p or p.strip() in ["ba", "mba"]:
        return "Business Analytics"
    if "computer science" in p or p.strip() in ["cs"]:
        return "Computer Science"
    if "computational science" in p:
        return "Computational Science"
    if "bioinformatics" in p or "system biology" in p:
        return "Bioinformatics"
    if any(x in p for x in ["econometrics", "finance", "fintech", "quant"]):
        return "Finance / Econometrics"
    return "Other"

df["programme_group"] = df["programme"].apply(group_programme)


# ── summary table ────────────────────────────────────────────────────────────

print("=" * 60)
print("NUMERIC ATTRIBUTES")
print("=" * 60)

summary = pd.DataFrame({
    "Attribute":     ["Stress (0-100)", "Sport hrs/wk", "Students est.", "Random (1-10)"],
    "N valid":       [len(stress_valid), len(sport_valid), len(estim.dropna()), len(randnum_valid)],
    "Outlier/Miss":  [
        len(stress) - len(stress_valid),
        len(sport)  - len(sport_valid),
        estim.isna().sum(),
        len(randnum) - len(randnum_valid),
    ],
    "Mean":   [stress_valid.mean(),   sport_valid.mean(),   estim.dropna().mean(),   randnum_valid.mean()],
    "Std":    [stress_valid.std(),    sport_valid.std(),    estim.dropna().std(),    randnum_valid.std()],
    "Median": [stress_valid.median(), sport_valid.median(), estim.dropna().median(), randnum_valid.median()],
    "Min":    [stress_valid.min(),    sport_valid.min(),    estim.dropna().min(),    randnum_valid.min()],
    "Max":    [stress_valid.max(),    sport_valid.max(),    estim.dropna().max(),    randnum_valid.max()],
}).round(2)

print(summary.to_string(index=False))

print("\n" + "=" * 60)
print("UNKNOWN / AMBIGUOUS VALUES (categorical columns)")
print("=" * 60)
for name, s in [("ML course", ml), ("IR course", ir), ("Databases", db), ("LLM used", llm)]:
    print(f"  {name:15s}: {(s == 'unknown').sum()} unknown out of {len(s)}")


# ── FIGURE 1: programme distribution + stress + sport + course background ────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("ODI-2026: Exploratory Data Analysis", fontsize=14, y=1.01)

# programme
prog_counts = df["programme_group"].value_counts().sort_values()
axes[0, 0].barh(prog_counts.index, prog_counts.values, color="steelblue")
axes[0, 0].set_title("Students per programme")
axes[0, 0].set_xlabel("Count")
for i, v in enumerate(prog_counts.values):
    axes[0, 0].text(v + 0.3, i, str(v), va="center", fontsize=9)

# stress level
axes[0, 1].hist(stress_valid, bins=20, color="coral", edgecolor="white")
axes[0, 1].set_title(
    f"Stress level distribution\n(n={len(stress_valid)}, {len(stress)-len(stress_valid)} outliers >100 excluded)"
)
axes[0, 1].set_xlabel("Stress (0–100)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].axvline(stress_valid.median(), color="black", linestyle="--",
                   linewidth=1, label=f"Median = {stress_valid.median():.0f}")
axes[0, 1].legend(fontsize=9)

# sport hours
axes[1, 0].hist(sport_valid, bins=15, color="mediumseagreen", edgecolor="white")
axes[1, 0].set_title(f"Sport hours per week (n={len(sport_valid)})")
axes[1, 0].set_xlabel("Hours / week")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].axvline(sport_valid.median(), color="black", linestyle="--",
                   linewidth=1, label=f"Median = {sport_valid.median():.0f}")
axes[1, 0].legend(fontsize=9)

# course background
courses = {"ML": ml, "IR": ir, "DB": db, "LLM": llm}
x = np.arange(len(courses))
width = 0.25
answer_colors = {"yes": "steelblue", "no": "coral", "unknown": "lightgray"}

for i, (answer, color) in enumerate(answer_colors.items()):
    counts = [s.value_counts().get(answer, 0) for s in courses.values()]
    axes[1, 1].bar(x + i * width, counts, width, label=answer,
                   color=color, edgecolor="white")

axes[1, 1].set_xticks(x + width)
axes[1, 1].set_xticklabels(courses.keys())
axes[1, 1].set_title("Academic background (courses taken)")
axes[1, 1].set_ylabel("Students")
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("eda_figure1.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_figure1.png")


# ── FIGURE 2: stress vs sport hours, coloured by gender ─────────────────────

fig2, ax = plt.subplots(figsize=(8, 5))

gender_col = df["gender"].astype(str).str.strip().str.lower()
for g, color in {"male": "steelblue", "female": "coral"}.items():
    mask = gender_col.str.contains(g) & stress.between(0, 100) & sport.between(0, 40)
    ax.scatter(sport[mask], stress[mask], alpha=0.5, color=color, label=g, s=40)
    if mask.sum() > 2:
        z = np.polyfit(sport[mask], stress[mask], 1)
        xs = np.linspace(sport[mask].min(), sport[mask].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), color=color, linewidth=1.5)

ax.set_xlabel("Sport hours / week")
ax.set_ylabel("Stress level (0–100)")
ax.set_title("Stress vs Sport hours by gender (linear regression)")
ax.legend()
plt.tight_layout()
plt.savefig("eda_figure2.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_figure2.png")


# ── FIGURE 3: students estimate distribution ─────────────────────────────────

fig3, ax = plt.subplots(figsize=(7, 4))
ax.hist(estim.dropna(), bins=20, color="mediumpurple", edgecolor="white")
ax.axvline(estim.median(), color="black", linestyle="--", linewidth=1.2,
           label=f"Median = {estim.median():.0f}")
ax.set_title("Estimated number of students in the room\n(actual value was ~500)")
ax.set_xlabel("Estimate")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("eda_figure3.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_figure3.png")


# ── noteworthy finding: random number column ─────────────────────────────────

print("\n" + "=" * 60)
print("NOTE: random number column (expected range 1–10)")
print("=" * 60)
n_valid   = randnum_valid.count()
n_total   = len(df)
print(f"  In range [1, 10]: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
print(f"  Out of range / non-numeric: {n_total - n_valid}/{n_total} ({100*(n_total-n_valid)/n_total:.1f}%)")
print(f"  Example outliers: {df['random_number'].sample(5, random_state=42).tolist()}")