"""
Win-Prediction Model: Does a candidate's disclosure profile predict electoral outcome?
=========================================================================================

PROJECT: Data-Driven Analysis of Political Disclosure — BMC 2026 (Borivali + Kandivali)
DATA: synthetic_bmc_candidates.csv (SIMULATED data, calibrated against real ADR-published
      aggregate statistics for BMC 2026 winners -- see generate_synthetic_dataset.py for
      calibration sources and methodology).

RESEARCH QUESTION:
    Can we predict whether a candidate won their ward using only information from their
    public nomination affidavit (disclosure profile) -- party, assets, education, criminal
    record -- without any information about campaign activity, popularity, or local context?

MODEL CHOICE: Logistic Regression
    Chosen over more complex models (random forest, gradient boosting) for three reasons:
    1. Interpretability -- for a disclosure-analysis project, WHICH features matter and in
       WHICH direction is more valuable than raw predictive accuracy. Logistic regression
       coefficients are directly interpretable as log-odds effects.
    2. Sample size -- with ~145 candidates across 21 wards, this is a small dataset. Complex
       models with many parameters risk overfitting; logistic regression's few parameters
       are appropriate here.
    3. This is a methodology demonstration on simulated data (see project documentation) --
       a transparent, well-understood model is the right choice for showing the pipeline
       works, rather than optimizing for accuracy on data with a designed ground truth.

FEATURES USED (all derived from the affidavit's "Summary" section, Section 14):
    - party            (categorical, one-hot encoded, BJP as reference-adjacent baseline)
    - log_assets        = log(1 + total declared assets in INR)
                          Log-transformed because raw asset values are heavily right-skewed
                          (a small number of very wealthy candidates would otherwise dominate)
    - log_income        = log(1 + annual declared income in INR)
    - higher_ed         = 1 if Graduate or Post Graduate, else 0 (binary simplification of
                          the 6-level education variable, chosen for interpretability)
    - criminal_severity = ordinal variable replacing the original two binary flags
                          (has_criminal_case, serious_criminal_case). Coded as:
                              0 = no criminal case
                              1 = minor case(s) only
                              2 = at least one serious case
                          CHANGE FROM EARLIER VERSION: the original two-binary-flag design
                          caused multicollinearity (every "serious" case is also a "has
                          case", so the flags overlap almost entirely), which made the
                          logistic regression coefficients unstable -- serious_criminal_case
                          flipped sign between the single train/test split and the full-data
                          fit. This single ordinal variable removes that redundancy and
                          gives a cleanly interpretable, stable coefficient.
    - age               (continuous, standardized)

WHY THESE FEATURES AND NOT OTHERS:
    - Liabilities were excluded: highly correlated with assets in this dataset (both scale
      together), and assets already captures the wealth signal.
    - Gender was excluded from the model deliberately: including it as a predictor of
      winning would conflate the effect of ward-level reservation (which mechanically
      determines gender in many wards) with any independent effect of gender itself --
      not a clean causal question this model design can answer.

EVALUATION METHODOLOGY:
    Given the small sample size (145 candidates, ~21 positive cases -- one winner per ward),
    a single train/test split gives unstable metrics that vary a lot depending on which
    candidates land in the test set. We use Stratified 5-Fold Cross-Validation instead:
    the data is split into 5 folds preserving the win/loss ratio in each fold, the model is
    trained on 4 folds and evaluated on the 5th, five times, and metrics are averaged.
    This gives a more reliable estimate of how well the model generalizes than any single
    split would.

    class_weight='balanced' is used throughout because winners are a minority class
    (~14% of candidates, since only 1 of ~7 candidates per ward wins) -- without this,
    a model could achieve high accuracy by simply predicting "lost" for everyone.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def load_and_engineer_features(path="../data/synthetic_bmc_candidates.csv"):
    """Load candidate data and construct the model-ready feature set."""
    df = pd.read_csv(path)

    df["log_assets"] = np.log1p(df["total_assets_inr"])
    df["log_income"] = np.log1p(df["annual_income_inr"])
    df["higher_ed"] = df["education"].isin(["Graduate", "Post Graduate"]).astype(int)
    df["criminal_severity"] = np.where(
        df["serious_criminal_case"] == 1, 2,
        np.where(df["criminal_cases"] > 0, 1, 0)
    )

    feature_cols = ["log_assets", "log_income", "higher_ed", "criminal_severity",
                     "age", "party"]
    X = df[feature_cols].copy()
    y = df["won"]
    return X, y, df


def build_pipeline():
    """
    Build a scikit-learn Pipeline that handles preprocessing and modeling together.
    Using a Pipeline (rather than manually scaling/encoding before the split) prevents
    data leakage: scaling parameters are fit ONLY on each fold's training data during
    cross-validation, never on data the model will be evaluated against.
    """
    numeric_features = ["log_assets", "log_income", "age"]
    categorical_features = ["party"]
    # higher_ed (0/1) and criminal_severity (0/1/2, ordinal) pass through unscaled --
    # criminal_severity is deliberately NOT one-hot encoded: treating it as ordinal
    # (rather than 3 separate categories) encodes the assumption that "serious" is a
    # more severe version of "minor", not an unrelated category -- consistent with how
    # the affidavit itself defines severity as a threshold, not a type.

    from sklearn.preprocessing import OneHotEncoder
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("party_ohe", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ], remainder="passthrough")  # passes through higher_ed, criminal_severity

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return pipeline


def run_cross_validation(X, y, n_splits=5):
    """
    Stratified K-Fold cross-validation. Returns per-fold and averaged metrics.
    Stratification ensures each fold has a similar win/loss ratio to the full dataset --
    important here because winners are rare (~1 per ward).
    """
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    return results


def fit_final_model_and_get_coefficients(X, y):
    """
    Fit the model on the FULL dataset (after cross-validation has already told us how
    well it generalizes) to extract final, most-stable coefficient estimates for
    interpretation in the report.
    """
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    ohe = pipeline.named_steps["preprocess"].named_transformers_["party_ohe"]
    party_names = ohe.get_feature_names_out(["party"])
    numeric_names = ["log_assets", "log_income", "age"]
    passthrough_names = ["higher_ed", "criminal_severity"]

    all_feature_names = numeric_names + list(party_names) + passthrough_names
    coefficients = pipeline.named_steps["classifier"].coef_[0]

    coef_series = pd.Series(coefficients, index=all_feature_names).sort_values(
        key=abs, ascending=False
    )
    return pipeline, coef_series


if __name__ == "__main__":
    X, y, df = load_and_engineer_features()

    print(f"Dataset: {len(X)} candidates, {y.sum()} winners ({y.mean():.1%})\n")

    print("=" * 60)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 60)
    cv_results = run_cross_validation(X, y, n_splits=5)

    for metric in ["accuracy", "roc_auc", "precision", "recall", "f1"]:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric.upper():12s}: {scores.mean():.3f} (+/- {scores.std():.3f})  "
              f"[folds: {np.round(scores, 2)}]")

    print("\n" + "=" * 60)
    print("FINAL MODEL (fit on full dataset) -- FEATURE COEFFICIENTS")
    print("=" * 60)
    print("(Positive = increases win probability, Negative = decreases it)\n")
    final_pipeline, coefficients = fit_final_model_and_get_coefficients(X, y)
    print(coefficients.to_string())

    print("\nNOTE: Coefficients are in standardized/log space for numeric features and")
    print("log-odds space throughout -- report relative magnitude and direction, not")
    print("raw values, when writing this up.")
