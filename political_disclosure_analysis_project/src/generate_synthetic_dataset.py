"""
Synthetic candidate disclosure dataset generator.

IMPORTANT: This generates SIMULATED data. No row represents a real person or
real candidate. Names are generated placeholders. This must be clearly labeled
as synthetic/simulated in your report methodology section -- do not present
outputs as real BMC candidate records.

Calibration sources (real, published, cited):
  - ADR / Maharashtra Election Watch, "Analysis of Criminal Background, Financial,
    Education, Gender and other details of Winning Candidates in BMC Elections 2026"
    (published Feb 2026): used for realistic marginal distributions among winners
    (criminal case rate, crorepati rate, average assets, education split, age split,
    gender split).
  - Real ward reservation categories for wards 8-28 (R/Central, R/South) from
    public ward statistics.
  - Real party vote/seat shares from the 2026 BMC election (227 seats: BJP 89,
    SS(UBT) 65, Shiv Sena 29, INC 24, AIMIM 8, MNS 6, NCP 3, SP 2, NCP-SP 1).

The win/loss outcome is generated from a designed logistic relationship between
features (so your ML model has real signal to recover) plus random noise --
NOT pure randomness. This makes the "does disclosure profile predict winning"
question meaningful as a methodology exercise, even though the underlying
candidates are fictional.
"""

import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ---- Real ward metadata (wards 8-28: R/Central + R/South, Borivali + Kandivali) ----
WARDS = {
    8: "Women", 9: "General", 10: "OBC", 11: "OBC Women", 12: "OBC Women",
    13: "OBC Women", 14: "Women", 15: "Women", 16: "Women", 17: "Women",
    18: "OBC Women", 19: "OBC Women", 20: "General", 21: "Women", 22: "General",
    23: "General", 24: "Women", 25: "General", 26: "SC", 27: "OBC Women", 28: "Women",
}
WARD_LIST = list(WARDS.keys())

# Real party seat shares (2026 BMC, all 227 seats) -> used as party selection weights
PARTIES = {
    "BJP": 89, "SS (UBT)": 65, "Shiv Sena": 29, "INC": 24,
    "AIMIM": 8, "MNS": 6, "NCP": 3, "SP": 2, "NCP-SP": 1,
    "Independent": 15,  # approx buffer for non-major candidates
}
party_names = list(PARTIES.keys())
party_weights = np.array(list(PARTIES.values()), dtype=float)
party_weights /= party_weights.sum()

EDUCATION_LEVELS = ["Illiterate", "Literate", "5th-12th pass", "Diploma", "Graduate", "Post Graduate"]
# Calibrated roughly to ADR's winner breakdown: ~47% mid-level, ~50% graduate+, few literate/illiterate
EDUCATION_WEIGHTS = [0.01, 0.01, 0.40, 0.08, 0.30, 0.20]

FIRST_NAMES_M = ["Rajesh", "Suresh", "Amit", "Vijay", "Sanjay", "Prakash", "Nilesh", "Mahesh", "Ganesh", "Ashok"]
FIRST_NAMES_F = ["Sunita", "Anita", "Priya", "Rekha", "Vandana", "Kavita", "Meena", "Sarika", "Neha", "Poonam"]
SURNAMES = ["Patil", "Shinde", "Jadhav", "Kadam", "Pawar", "More", "Deshmukh", "Salvi", "Naik", "Rane"]


def synthetic_name(gender):
    first = random.choice(FIRST_NAMES_F if gender == "Woman" else FIRST_NAMES_M)
    return f"{first} {random.choice(SURNAMES)}"


def assign_gender(reservation):
    """Ward reservation forces gender for at least one seat; approximate accordingly."""
    if "Women" in reservation:
        return "Woman"
    return random.choices(["Man", "Woman"], weights=[0.65, 0.35])[0]


def generate_candidate(cand_id, ward):
    reservation = WARDS[ward]
    gender = assign_gender(reservation)
    age = int(np.clip(np.random.normal(46, 11), 25, 78))
    education = random.choices(EDUCATION_LEVELS, weights=EDUCATION_WEIGHTS)[0]
    party = random.choices(party_names, weights=party_weights)[0]

    # Criminal cases: ~24% have >=1 case, ~13% of those are "serious" (roughly matches ADR)
    has_case = random.random() < 0.24
    criminal_cases = np.random.poisson(1.5) + 1 if has_case else 0
    serious_case = 1 if (has_case and random.random() < 0.55) else 0

    # Assets: log-normal, calibrated so ~79% end up "crorepati" (>=1,00,00,000),
    # matching ADR's reported winner crorepati rate for BMC 2026
    total_assets = float(np.random.lognormal(mean=17.0, sigma=1.1))
    liabilities = float(total_assets * np.random.uniform(0, 0.25))
    annual_income = float(np.random.lognormal(mean=12.8, sigma=0.9))
    declared_expenditure = float(np.random.uniform(50000, 500000))  # ward-level ceiling range

    return {
        "candidate_id": cand_id,
        "ward": ward,
        "ward_reservation": reservation,
        "name": synthetic_name(gender),
        "gender": gender,
        "age": age,
        "education": education,
        "party": party,
        "criminal_cases": criminal_cases,
        "serious_criminal_case": serious_case,
        "total_assets_inr": round(total_assets, 2),
        "liabilities_inr": round(liabilities, 2),
        "annual_income_inr": round(annual_income, 2),
        "declared_expenditure_inr": round(declared_expenditure, 2),
    }


def assign_outcomes(df):
    """
    Designed logistic relationship for win probability, so the ML model has
    real signal to recover -- not pure randomness. Party strength dominates
    (matches real seat-share pattern), with smaller effects from assets,
    criminal record (negative, mild), and education.
    """
    party_strength = {p: PARTIES[p] / 89 for p in party_names}  # normalize to BJP=1.0

    df["log_assets"] = np.log1p(df["total_assets_inr"])
    z = (
        1.6 * df["party"].map(party_strength)
        + 0.35 * (df["log_assets"] - df["log_assets"].mean()) / df["log_assets"].std()
        - 0.25 * df["serious_criminal_case"]
        + 0.10 * (df["education"].isin(["Graduate", "Post Graduate"])).astype(int)
        + np.random.normal(0, 0.9, size=len(df))  # noise
    )
    prob = 1 / (1 + np.exp(-z))

    # Enforce exactly one winner per ward (mirrors real single-member ward structure)
    df["_win_score"] = prob
    df["won"] = 0
    for ward, group in df.groupby("ward"):
        winner_idx = group["_win_score"].idxmax()
        df.loc[winner_idx, "won"] = 1
    df.drop(columns=["_win_score", "log_assets"], inplace=True)
    return df


def generate_dataset(candidates_per_ward=(4, 9)):
    rows = []
    cand_id = 1
    for ward in WARD_LIST:
        n_candidates = random.randint(*candidates_per_ward)
        for _ in range(n_candidates):
            rows.append(generate_candidate(cand_id, ward))
            cand_id += 1
    df = pd.DataFrame(rows)
    df = assign_outcomes(df)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("../data/synthetic_bmc_candidates.csv", index=False)
    print(f"Generated {len(df)} synthetic candidates across {len(WARD_LIST)} wards")
    print(f"Winners: {df['won'].sum()} (should equal number of wards: {len(WARD_LIST)})")
    print(f"\nCriminal cases rate: {(df['criminal_cases'] > 0).mean():.1%}")
    print(f"Crorepati rate: {(df['total_assets_inr'] >= 1e7).mean():.1%}")
    print(f"Women candidates: {(df['gender'] == 'Woman').mean():.1%}")
    print(f"\nSaved to synthetic_bmc_candidates.csv")
