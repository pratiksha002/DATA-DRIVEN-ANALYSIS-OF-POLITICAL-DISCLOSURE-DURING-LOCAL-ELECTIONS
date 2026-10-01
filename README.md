# Data-Driven Analysis of Political Disclosure — BMC 2026 Election

**Scope:** Borivali (R/Central, wards 8–17) + Kandivali (R/South, wards 18–28)

> ⚠️ **All data in this project is SIMULATED**, not real candidate/voter records.
> It is calibrated against real published ADR statistics and real BMC 2026 seat
> shares (see `src/generate_synthetic_dataset.py` docstring for sources). This was
> explicitly approved as the project's basis by the project guide. State this
> clearly in any presentation of results.

## Project structure

```
political_disclosure_analysis/
├── data/                                  # Generated datasets (CSV)
│   ├── synthetic_bmc_candidates.csv          145 candidates, 21 wards
│   ├── synthetic_voter_perception.csv        210 simulated survey respondents
│   └── voter_survey_form_responses.csv       same data, Google Form export format
│
├── src/                                   # Pipeline code
│   ├── generate_synthetic_dataset.py         Candidate data generator
│   ├── generate_synthetic_voter_perception.py Voter perception data generator
│   ├── win_prediction_model.py               ML model: logistic regression + CV
│   └── generate_report_visuals.py            Generates all report charts
│
├── dashboard/
│   └── dashboard.py                       Interactive Streamlit dashboard
│
├── report_visuals/                        10 PNG charts for the written report
│
├── scraper/
│   └── sec_affidavit_scraper.py           SEC portal scraper (exploratory,
│                                            not used in final pipeline — kept
│                                            for documentation of the real-data
│                                            approach that was attempted first)
│
├── report/
│   └── Political_Disclosure_Analysis_Report.docx
│
├── requirements.txt
└── README.md   (this file)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run order (if regenerating from scratch)

```bash
cd src
python3 generate_synthetic_dataset.py           # -> ../data/synthetic_bmc_candidates.csv
python3 generate_synthetic_voter_perception.py  # -> ../data/synthetic_voter_perception.csv
python3 win_prediction_model.py                 # prints CV results + coefficients
python3 generate_report_visuals.py              # -> ../report_visuals/*.png
```

## Run the dashboard

```bash
cd dashboard
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. Requires `data/` to be one level up (already
the case in this project structure) and `src/win_prediction_model.py` importable
(handled automatically via `sys.path` in `dashboard.py`).

## Methodology summary

1. **Candidate disclosure data** — simulated candidates across 21 real wards, with
   realistic distributions (party seat-share, criminal case rate, asset levels)
   calibrated against ADR's published BMC 2026 statistics.
2. **Win-prediction model** — logistic regression predicting electoral outcome from
   disclosure-profile features only. Validated with stratified 5-fold cross-validation
   (ROC-AUC ≈ 0.81). See `src/win_prediction_model.py` docstring for full methodology
   and design rationale.
3. **Voter perception study** — simulated before/after survey measuring favorability
   shift after exposure to a candidate's disclosures, with prior awareness as a
   designed moderating variable.
4. **Dashboard & visuals** — all charts and the dashboard read live from the CSVs
   and the trained model, so re-running the pipeline updates everything downstream.

See `report/Political_Disclosure_Analysis_Report.docx` for the full write-up,
findings, and limitations.
