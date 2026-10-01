"""
Political Disclosure Analysis Dashboard — BMC 2026 (Borivali + Kandivali)
===========================================================================
Run locally with:  streamlit run dashboard.py

Requires the following files in the same folder:
  - synthetic_bmc_candidates.csv
  - synthetic_voter_perception.csv
  - win_prediction_model.py   (imported for model training functions)

NOTE: All underlying data is SIMULATED, calibrated against real published
ADR statistics for BMC 2026. This is stated explicitly in the dashboard's
"About" section — do not remove that disclosure when presenting this.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from win_prediction_model import load_and_engineer_features, build_pipeline, run_cross_validation, fit_final_model_and_get_coefficients

st.set_page_config(page_title="BMC Disclosure Analysis", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    candidates = pd.read_csv("../data/synthetic_bmc_candidates.csv")
    perception = pd.read_csv("../data/synthetic_voter_perception.csv")
    return candidates, perception

candidates_df, perception_df = load_data()

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Candidate Disclosures",
    "Win Prediction Model",
    "Voter Perception",
    "Methodology & Limitations",
])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data-Driven Analysis of Political Disclosure — BMC 2026 Election\n\n"
    "Wards 8–28 (Borivali + Kandivali)"
)

# ============================================================
# PAGE: OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Political Disclosure Analysis — BMC 2026")
    st.caption("Borivali (R/Central) + Kandivali (R/South) — Wards 8–28")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Candidates analyzed", len(candidates_df))
    col2.metric("Wards covered", candidates_df["ward"].nunique())
    col3.metric("Have a criminal case", f"{(candidates_df['criminal_cases'] > 0).mean():.0%}")
    col4.metric("Crorepati candidates", f"{(candidates_df['total_assets_inr'] >= 1e7).mean():.0%}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("What this project analyzes")
        st.write(
            "This dashboard combines two linked studies:\n\n"
            "1. **Candidate disclosure analysis** — what candidates declared in their "
            "nomination affidavits (assets, criminal record, education), and whether "
            "that profile predicts who wins.\n\n"
            "2. **Voter perception study** — how aware voters are of these disclosures, "
            "and whether seeing them shifts favorability toward a candidate."
        )

    with c2:
        st.subheader("Headline finding")
        unaware_pct = (~perception_df["already_aware_pre_exposure"]).mean()
        st.metric("Voters unaware of disclosures before exposure", f"{unaware_pct:.0%}")
        st.write(
            "Exposure to disclosure details significantly shifts favorability "
            f"(mean change: {perception_df['favorability_change'].mean():+.2f} points, "
            "p < 0.001), and the shift is larger for voters who weren't already aware."
        )

# ============================================================
# PAGE: CANDIDATE DISCLOSURES
# ============================================================
elif page == "Candidate Disclosures":
    st.title("Candidate Disclosure Landscape")

    ward_filter = st.multiselect(
        "Filter by ward", sorted(candidates_df["ward"].unique()),
        default=sorted(candidates_df["ward"].unique())
    )
    df = candidates_df[candidates_df["ward"].isin(ward_filter)]

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Criminal cases by party")
        crime_by_party = df.groupby("party").agg(
            candidates=("candidate_id", "count"),
            with_case=("criminal_cases", lambda x: (x > 0).sum())
        ).reset_index()
        crime_by_party["rate"] = crime_by_party["with_case"] / crime_by_party["candidates"]
        fig = px.bar(crime_by_party.sort_values("rate", ascending=False),
                     x="party", y="rate", title="Share of candidates with a criminal case",
                     labels={"rate": "Share with criminal case"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Asset distribution")
        fig2 = px.histogram(df, x="total_assets_inr", nbins=30,
                             title="Distribution of declared assets (INR)",
                             labels={"total_assets_inr": "Total assets (INR)"})
        fig2.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Education breakdown")
        edu_counts = df["education"].value_counts().reset_index()
        edu_counts.columns = ["education", "count"]
        fig3 = px.pie(edu_counts, names="education", values="count")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Ward reservation categories")
        ward_res = df.drop_duplicates("ward")["ward_reservation"].value_counts().reset_index()
        ward_res.columns = ["reservation", "wards"]
        fig4 = px.bar(ward_res, x="reservation", y="wards", title="Wards by reservation category")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Full candidate table")
    st.dataframe(df, use_container_width=True)

# ============================================================
# PAGE: WIN PREDICTION MODEL
# ============================================================
elif page == "Win Prediction Model":
    st.title("Win Prediction from Disclosure Profile")
    st.write(
        "Logistic regression predicting electoral outcome from disclosure-profile "
        "features only (party, assets, education, criminal record) — no campaign "
        "or popularity data. See *Methodology & Limitations* for why this model "
        "was chosen and how it was validated."
    )

    X, y, df_full = load_and_engineer_features()

    with st.spinner("Running cross-validation..."):
        cv_results = run_cross_validation(X, y, n_splits=5)

    st.subheader("Cross-validated performance (5-fold)")
    metrics_summary = pd.DataFrame({
        "Metric": ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1"],
        "Mean": [cv_results[f"test_{m}"].mean() for m in
                 ["accuracy", "roc_auc", "precision", "recall", "f1"]],
        "Std Dev": [cv_results[f"test_{m}"].std() for m in
                    ["accuracy", "roc_auc", "precision", "recall", "f1"]],
    })
    st.dataframe(metrics_summary.style.format({"Mean": "{:.3f}", "Std Dev": "{:.3f}"}),
                 use_container_width=True)
    st.caption(
        "Precision is intentionally traded off for recall (class_weight='balanced') "
        "because winners are a minority class (~14.5% of candidates)."
    )

    st.subheader("Feature importance (final model, fit on full data)")
    final_pipeline, coefficients = fit_final_model_and_get_coefficients(X, y)
    coef_df = coefficients.reset_index()
    coef_df.columns = ["feature", "coefficient"]
    coef_df["direction"] = np.where(coef_df["coefficient"] > 0, "Increases win probability",
                                     "Decreases win probability")
    fig5 = px.bar(coef_df.sort_values("coefficient"), x="coefficient", y="feature",
                  color="direction", orientation="h",
                  title="Logistic regression coefficients (log-odds scale)")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("ROC Curve")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC curve (AUC = {roc_auc_val:.3f})"))
    fig6.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random baseline"))
    fig6.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig6, use_container_width=True)

# ============================================================
# PAGE: VOTER PERCEPTION
# ============================================================
elif page == "Voter Perception":
    st.title("Voter Awareness & Perception Shift")

    col1, col2, col3 = st.columns(3)
    col1.metric("Respondents", len(perception_df))
    col2.metric("Unaware pre-exposure",
                f"{(~perception_df['already_aware_pre_exposure']).mean():.0%}")
    col3.metric("Mean favorability change",
                f"{perception_df['favorability_change'].mean():+.2f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Favorability: before vs after exposure")
        melted = perception_df.melt(
            id_vars=["respondent_id"],
            value_vars=["baseline_favorability", "post_exposure_favorability"],
            var_name="stage", value_name="favorability"
        )
        melted["stage"] = melted["stage"].map({
            "baseline_favorability": "Before", "post_exposure_favorability": "After"
        })
        fig7 = px.box(melted, x="stage", y="favorability", title="Favorability distribution")
        st.plotly_chart(fig7, use_container_width=True)

    with c2:
        st.subheader("Effect of prior awareness (serious-case candidates only)")
        serious = perception_df[perception_df["candidate_serious_case"] == True]
        by_awareness = serious.groupby("already_aware_pre_exposure")["favorability_change"].mean().reset_index()
        by_awareness["already_aware_pre_exposure"] = by_awareness["already_aware_pre_exposure"].map(
            {True: "Already aware", False: "Not aware (new info)"}
        )
        fig8 = px.bar(by_awareness, x="already_aware_pre_exposure", y="favorability_change",
                      title="Mean favorability change by prior awareness")
        st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Raw response data")
    st.dataframe(perception_df, use_container_width=True)

# ============================================================
# PAGE: METHODOLOGY
# ============================================================
elif page == "Methodology & Limitations":
    st.title("Methodology & Limitations")
    st.warning(
        "**All candidate and respondent data in this dashboard is SIMULATED**, "
        "not real BMC 2026 records. This was explicitly approved as the project's "
        "basis by the project guide. See below for how it was constructed."
    )

    st.subheader("Candidate data calibration")
    st.write(
        "- Party seat-share weights based on real 2026 BMC results (227 seats)\n"
        "- Criminal case rate, crorepati rate, and education breakdown calibrated "
        "against ADR/Maharashtra Election Watch's published analysis of real BMC "
        "2026 winning candidates\n"
        "- Ward reservation categories (Women/OBC/SC/General) are real, for wards 8–28"
    )

    st.subheader("Voter perception data design")
    st.write(
        "- Before/after paired design: baseline favorability, simulated exposure to "
        "a matched candidate's real (simulated) disclosure, post-exposure favorability\n"
        "- The shift is governed by a designed model (not random): serious criminal "
        "cases lower favorability more than minor ones, the effect is larger for "
        "voters who weren't already aware, and wealth has a small, deliberately "
        "ambiguous effect\n"
        "- Gender was excluded from the win-prediction model to avoid confounding "
        "with ward-level gender reservation"
    )

    st.subheader("Model validation")
    st.write(
        "- Stratified 5-fold cross-validation (not a single train/test split) given "
        "the small, imbalanced sample (145 candidates, ~14.5% winners)\n"
        "- class_weight='balanced' used throughout to address class imbalance\n"
        "- criminal_severity coded as a single ordinal variable (not two overlapping "
        "binary flags) to avoid multicollinearity"
    )

    st.subheader("Key limitations to state in the report")
    st.write(
        "- Small sample size (145 candidates, 210 respondents) limits statistical "
        "power and the stability of classification metrics\n"
        "- Findings describe the designed simulation, not real Mumbai voter behavior "
        "or real candidate outcomes — frame all results as methodology demonstration\n"
        "- Real survey data collection was not performed; this was a deliberate "
        "scope decision approved by the project guide"
    )
