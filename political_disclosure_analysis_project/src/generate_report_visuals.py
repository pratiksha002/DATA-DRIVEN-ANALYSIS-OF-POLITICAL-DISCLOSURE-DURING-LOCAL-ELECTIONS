"""
Generates static, report-ready visuals (PNG) covering all three analysis components:
candidate disclosures, win-prediction model, and voter perception.

Run: python3 generate_report_visuals.py
Outputs saved to ./report_visuals/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from win_prediction_model import load_and_engineer_features, build_pipeline, run_cross_validation, fit_final_model_and_get_coefficients

sns.set_theme(style="whitegrid", font_scale=1.05)
OUT = "../report_visuals"
os.makedirs(OUT, exist_ok=True)

candidates_df = pd.read_csv("../data/synthetic_bmc_candidates.csv")
perception_df = pd.read_csv("../data/synthetic_voter_perception.csv")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}/{name}.png")


# ============================================================
# 1. Candidate disclosure descriptives
# ============================================================

# Criminal case rate by party
crime_by_party = candidates_df.groupby("party").agg(
    n=("candidate_id", "count"),
    with_case=("criminal_cases", lambda x: (x > 0).sum())
).reset_index()
crime_by_party["rate"] = crime_by_party["with_case"] / crime_by_party["n"]
crime_by_party = crime_by_party.sort_values("rate", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=crime_by_party, x="rate", y="party", hue="party", legend=False,
            palette="Reds_r", ax=ax)
ax.set_xlabel("Share of candidates with a declared criminal case")
ax.set_ylabel("Party")
ax.set_title("Criminal Case Disclosure Rate by Party")
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
save(fig, "01_criminal_cases_by_party")

# Asset distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(candidates_df["total_assets_inr"] / 1e7, bins=25, kde=True, ax=ax, color="#2a6f97")
ax.axvline(1.0, color="crimson", linestyle="--", label="Crorepati threshold (₹1 Cr)")
ax.set_xlabel("Total declared assets (₹ Crore)")
ax.set_ylabel("Number of candidates")
ax.set_title("Distribution of Declared Assets")
ax.legend()
save(fig, "02_asset_distribution")

# Education breakdown
fig, ax = plt.subplots(figsize=(7, 6))
edu_counts = candidates_df["education"].value_counts()
ax.pie(edu_counts.values, labels=edu_counts.index, autopct="%1.0f%%",
       colors=sns.color_palette("Set2"), startangle=90)
ax.set_title("Candidate Education Breakdown")
save(fig, "03_education_breakdown")

# Ward reservation categories
fig, ax = plt.subplots(figsize=(8, 5))
ward_res = candidates_df.drop_duplicates("ward")["ward_reservation"].value_counts()
sns.barplot(x=ward_res.index, y=ward_res.values, hue=ward_res.index, legend=False,
            palette="Blues_d", ax=ax)
ax.set_xlabel("Reservation category")
ax.set_ylabel("Number of wards")
ax.set_title("Ward Reservation Categories (Wards 8-28)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
save(fig, "04_ward_reservation")

# ============================================================
# 2. Win-prediction model visuals
# ============================================================
X, y, _ = load_and_engineer_features()

# Cross-validation metrics with error bars
cv_results = run_cross_validation(X, y, n_splits=5)
metrics = ["accuracy", "roc_auc", "precision", "recall", "f1"]
means = [cv_results[f"test_{m}"].mean() for m in metrics]
stds = [cv_results[f"test_{m}"].std() for m in metrics]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics, means, yerr=stds, capsize=6, color="#457b9d", alpha=0.85)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Win-Prediction Model: 5-Fold Cross-Validated Performance")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.03, f"{m:.2f}",
            ha="center", fontsize=10)
save(fig, "05_cv_performance")

# Feature importance
_, coefficients = fit_final_model_and_get_coefficients(X, y)
coef_df = coefficients.reset_index()
coef_df.columns = ["feature", "coefficient"]
coef_df["color"] = np.where(coef_df["coefficient"] > 0, "#2a9d8f", "#e76f51")

fig, ax = plt.subplots(figsize=(9, 6))
coef_df_sorted = coef_df.sort_values("coefficient")
ax.barh(coef_df_sorted["feature"], coef_df_sorted["coefficient"], color=coef_df_sorted["color"])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Logistic regression coefficient (log-odds)")
ax.set_title("Feature Importance — Win Prediction Model")
save(fig, "06_feature_importance")

# ROC curve
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
pipeline = build_pipeline()
pipeline.fit(X_train, y_train)
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#264653", linewidth=2.5, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Win Prediction Model")
ax.legend(loc="lower right")
save(fig, "07_roc_curve")

# ============================================================
# 3. Voter perception visuals
# ============================================================

# Before/after favorability
melted = perception_df.melt(
    id_vars=["respondent_id"],
    value_vars=["baseline_favorability", "post_exposure_favorability"],
    var_name="stage", value_name="favorability"
)
melted["stage"] = melted["stage"].map({
    "baseline_favorability": "Before exposure", "post_exposure_favorability": "After exposure"
})

fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=melted, x="stage", y="favorability", hue="stage", legend=False,
            palette=["#8ecae6", "#fb8500"], ax=ax)
sns.stripplot(data=melted, x="stage", y="favorability", color="black", alpha=0.15, ax=ax)
ax.set_title("Favorability Before vs After Disclosure Exposure")
ax.set_ylabel("Favorability score (1-10)")
save(fig, "08_favorability_before_after")

# Awareness moderation effect (serious-case candidates)
serious = perception_df[perception_df["candidate_serious_case"] == True]
by_awareness = serious.groupby("already_aware_pre_exposure")["favorability_change"].agg(["mean", "sem"]).reset_index()
by_awareness["already_aware_pre_exposure"] = by_awareness["already_aware_pre_exposure"].map(
    {True: "Already aware", False: "Not aware\n(new information)"}
)

fig, ax = plt.subplots(figsize=(7, 6))
bars = ax.bar(by_awareness["already_aware_pre_exposure"], by_awareness["mean"],
              yerr=by_awareness["sem"], capsize=8, color=["#6a994e", "#bc4749"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Mean favorability change")
ax.set_title("Awareness Moderates the Impact of Serious Criminal Disclosures")
for bar, v in zip(bars, by_awareness["mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2, v - 0.3 if v < 0 else v + 0.1,
            f"{v:+.2f}", ha="center", fontsize=11, fontweight="bold")
save(fig, "09_awareness_moderation")

# Awareness rate pie
fig, ax = plt.subplots(figsize=(6, 6))
awareness_counts = perception_df["already_aware_pre_exposure"].value_counts()
labels = ["Unaware", "Already aware"] if False in awareness_counts.index else awareness_counts.index
ax.pie(awareness_counts.values,
       labels=["Unaware before exposure" if not i else "Already aware" for i in awareness_counts.index],
       autopct="%1.0f%%", colors=["#e63946", "#a8dadc"], startangle=90)
ax.set_title("Voter Awareness of Disclosures — Pre-Exposure")
save(fig, "10_awareness_rate")

print(f"\nAll visuals saved to ./{OUT}/")
