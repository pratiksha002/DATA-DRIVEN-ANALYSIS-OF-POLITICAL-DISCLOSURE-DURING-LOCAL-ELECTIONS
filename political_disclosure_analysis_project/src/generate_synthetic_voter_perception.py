"""
Synthetic voter awareness & perception-change dataset generator.

IMPORTANT: This generates SIMULATED respondents, not real survey data. This must
be clearly labeled as synthetic/simulated in your report methodology -- do not
present outputs as real voter survey results.

Design: paired before/after. Each simulated respondent is matched to one real
candidate from synthetic_bmc_candidates.csv (their own ward). We generate a
baseline favorability score, then simulate "exposure" to that candidate's actual
disclosure (criminal record, assets) and compute a post-exposure favorability
score. The shift is NOT random -- it follows a designed model:

  - Serious criminal case disclosure lowers favorability more than a minor one
  - The drop is LARGER when the respondent was not already aware (new information
    moves opinion more than confirming what's already known) -- this is the
    "awareness moderates disclosure impact" effect your project is centered on
  - Wealth/assets have a small, deliberately ambiguous effect (can read as
    "capable" or "corrupt" -- reflects genuine mixed findings in real research,
    not resolved either way here)
  - Higher education respondents react somewhat more strongly to criminal
    disclosures (education as a moderator)

Run generate_synthetic_dataset.py FIRST -- this script reads its output.
"""

import numpy as np
import pandas as pd

np.random.seed(7)

CANDIDATES_FILE = "../data/synthetic_bmc_candidates.csv"
RESPONDENTS_PER_WARD = 10

EDUCATION_LEVELS = ["Below 10th", "10th-12th pass", "Graduate", "Post Graduate"]
EDUCATION_WEIGHTS = [0.10, 0.35, 0.40, 0.15]
HIGHER_ED = {"Graduate", "Post Graduate"}


def generate_respondents(candidates_df):
    rows = []
    resp_id = 1
    for ward, group in candidates_df.groupby("ward"):
        for _ in range(RESPONDENTS_PER_WARD):
            candidate = group.sample(1).iloc[0]

            age = int(np.clip(np.random.normal(38, 13), 18, 80))
            gender = np.random.choice(["Man", "Woman", "Other"], p=[0.52, 0.47, 0.01])
            education = np.random.choice(EDUCATION_LEVELS, p=EDUCATION_WEIGHTS)

            # Baseline awareness of THIS candidate's disclosures -- most voters aren't
            # already aware; slightly more likely if the case is serious (more newsworthy)
            base_awareness_p = 0.25 + (0.15 if candidate["serious_criminal_case"] else 0)
            already_aware = np.random.random() < base_awareness_p

            baseline_favorability = float(np.clip(np.random.normal(5.5, 1.8), 0, 10))

            # --- Designed exposure effect ---
            if candidate["serious_criminal_case"]:
                criminal_effect = -2.2
            elif candidate["criminal_cases"] > 0:
                criminal_effect = -0.9
            else:
                criminal_effect = 0.0

            awareness_multiplier = 0.5 if already_aware else 1.3  # new info moves opinion more
            education_multiplier = 1.25 if education in HIGHER_ED else 1.0

            wealth_effect = 0.12 * np.log1p(candidate["total_assets_inr"] / 1e7)
            wealth_effect *= np.random.choice([1, -1], p=[0.55, 0.45])  # deliberately ambiguous

            noise = np.random.normal(0, 0.7)

            delta = (criminal_effect * awareness_multiplier * education_multiplier
                     + wealth_effect + noise)

            post_favorability = float(np.clip(baseline_favorability + delta, 0, 10))

            rows.append({
                "respondent_id": resp_id,
                "ward": ward,
                "matched_candidate_id": candidate["candidate_id"],
                "respondent_age": age,
                "respondent_gender": gender,
                "respondent_education": education,
                "already_aware_pre_exposure": already_aware,
                "candidate_criminal_cases": candidate["criminal_cases"],
                "candidate_serious_case": bool(candidate["serious_criminal_case"]),
                "candidate_total_assets_inr": candidate["total_assets_inr"],
                "baseline_favorability": round(baseline_favorability, 2),
                "post_exposure_favorability": round(post_favorability, 2),
                "favorability_change": round(post_favorability - baseline_favorability, 2),
            })
            resp_id += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    candidates_df = pd.read_csv(CANDIDATES_FILE)
    df = generate_respondents(candidates_df)
    df.to_csv("../data/synthetic_voter_perception.csv", index=False)

    print(f"Generated {len(df)} synthetic respondents across "
          f"{candidates_df['ward'].nunique()} wards\n")
    print(f"Already aware of disclosures pre-exposure: "
          f"{df['already_aware_pre_exposure'].mean():.1%}")
    print(f"Average favorability change (all): {df['favorability_change'].mean():+.2f}")
    print(f"Average change when candidate has serious case: "
          f"{df[df['candidate_serious_case']]['favorability_change'].mean():+.2f}")
    print(f"Average change when NOT already aware vs already aware "
          f"(serious case only):")
    serious = df[df["candidate_serious_case"]]
    print(f"  Not aware before: "
          f"{serious[~serious['already_aware_pre_exposure']]['favorability_change'].mean():+.2f}")
    print(f"  Already aware:    "
          f"{serious[serious['already_aware_pre_exposure']]['favorability_change'].mean():+.2f}")
    print(f"\nSaved to synthetic_voter_perception.csv")
