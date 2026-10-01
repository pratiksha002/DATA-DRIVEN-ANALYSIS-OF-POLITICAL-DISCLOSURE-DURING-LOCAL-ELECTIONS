const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, PageNumber, Footer, Header,
} = require("docx");
const fs = require("fs");

const IMG = "../report_visuals";

function img(name, width = 550) {
  const data = fs.readFileSync(`${IMG}/${name}`);
  const sizeOf = require("image-size");
  let dims;
  try {
    dims = sizeOf(data);
  } catch (e) {
    dims = { width: 1400, height: 900 };
  }
  const height = Math.round((width * dims.height) / dims.width);
  return new ImageRun({ data, transformation: { width, height }, type: "png" });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 400, after: 200 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 150 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 160 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function figCaption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20 })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
  });
}
function figure(imgName, caption, width = 500) {
  return [
    new Paragraph({ children: [img(imgName, width)], alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 } }),
    figCaption(caption),
  ];
}

function simpleTable(headers, rows) {
  const colWidth = Math.floor(9000 / headers.length);
  const headerRow = new TableRow({
    children: headers.map(hText => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "D9E2F3" },
      children: [new Paragraph({ children: [new TextRun({ text: hText, bold: true })] })],
    })),
  });
  const dataRows = rows.map(r => new TableRow({
    children: r.map(cell => new TableCell({
      width: { size: colWidth, type: WidthType.DXA },
      children: [new Paragraph(String(cell))],
    })),
  }));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: headers.map(() => colWidth),
    rows: [headerRow, ...dataRows],
  });
}

const doc = new Document({
  sections: [
    // ---------------- TITLE PAGE ----------------
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "", spacing: { before: 2000 } }),
        new Paragraph({
          children: [new TextRun({ text: "Data-Driven Analysis of Political Disclosure", bold: true, size: 44 })],
          alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "During Local Elections", bold: true, size: 44 })],
          alignment: AlignmentType.CENTER, spacing: { after: 400 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "A Case Study on BMC 2026 — Borivali & Kandivali Wards", size: 28, italics: true })],
          alignment: AlignmentType.CENTER, spacing: { after: 800 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "College Field Project Report", size: 24 })],
          alignment: AlignmentType.CENTER, spacing: { after: 200 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "B.Sc. Data Science — Atharva University, Mumbai", size: 24 })],
          alignment: AlignmentType.CENTER, spacing: { after: 1200 },
        }),
        new Paragraph({
          children: [new TextRun({
            text: "NOTE: This project uses simulated data, calibrated against real published statistics, as explicitly approved by the project guide. See Section 2 for full methodology and justification.",
            italics: true, size: 20,
          })],
          alignment: AlignmentType.CENTER,
        }),
      ],
    },
    // ---------------- MAIN BODY ----------------
    {
      properties: {},
      headers: {
        default: new Header({ children: [new Paragraph({
          children: [new TextRun({ text: "Political Disclosure Analysis — BMC 2026", size: 18, color: "808080" })],
          alignment: AlignmentType.RIGHT,
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          children: [new TextRun({ children: [PageNumber.CURRENT] })],
          alignment: AlignmentType.CENTER,
        })] }),
      },
      children: [
        h1("Abstract"),
        p("This project analyzes political disclosure patterns in the 2026 Brihanmumbai Municipal Corporation (BMC) election, focused on the Borivali (R/Central) and Kandivali (R/South) ward clusters (wards 8–28). It combines two linked studies: (1) an analysis of candidates' legally mandated nomination affidavits — covering criminal record, assets, liabilities, and education — culminating in a machine learning model predicting electoral outcome from disclosure profile alone; and (2) a voter perception study measuring how awareness of these disclosures affects favorability toward a candidate. Due to practical constraints in accessing complete real-world affidavit and expenditure data within the project timeline, both datasets are simulated, carefully calibrated against real published statistics from the Association for Democratic Reforms (ADR) and real 2026 BMC election results. This approach was reviewed and approved by the project guide as a methodology demonstration. The win-prediction model achieves a cross-validated ROC-AUC of approximately 0.81, and the perception study finds that voters unaware of a candidate's serious criminal disclosures show a substantially larger drop in favorability (−3.72 points) upon learning of them, compared to voters already aware (−0.99 points) — directly supporting the hypothesis that disclosure awareness moderates its electoral impact."),

        h1("1. Introduction"),
        p("Political disclosure — the legal requirement for election candidates to declare their criminal history, assets, liabilities, and education in a sworn affidavit — is intended to give voters informed choice. However, the extent to which voters are actually aware of these disclosures, and whether such awareness changes their electoral behavior, is not well understood at the local body election level in India. This project investigates two connected questions using the 2026 BMC election as a case study:"),
        bullet("Does a candidate's disclosure profile (criminal record, wealth, education, party) predict whether they win their seat?"),
        bullet("How aware are voters of these disclosures, and does exposure to them shift how favorably they view a candidate?"),
        p("The BMC 2026 election (held 15 January 2026, the first since 2017) provides a timely, real-world backdrop for this analysis, with 227 wards contested and disclosure requirements mandated by the State Election Commission of Maharashtra."),

        h1("2. Methodology"),
        h2("2.1 Scope"),
        p("The analysis covers 21 municipal wards (numbered 8–28) spanning Borivali (R/Central, wards 8–17) and Kandivali (R/South, wards 18–28) — chosen for feasibility within a solo, one-semester project, and because these are areas the researcher has direct local familiarity with, aiding data verification."),
        h2("2.2 Data Source Investigation"),
        p("The original plan was to collect real candidate affidavits from the Maharashtra State Election Commission's public portal (mahasecelec.in) and real voter survey responses via Google Forms. Initial investigation confirmed the SEC portal does host genuine, machine-readable affidavit PDFs (verified by directly retrieving and parsing one), structured around a legally standardized format. However, full-scale collection required navigating a multi-step ASP.NET form-postback system with partial and inconsistent affidavit availability across candidates, which was not feasible to complete at full scale within the project timeline. An exploratory scraper (included in the project repository under scraper/) was built and partially validated against the live portal."),
        p("Given this constraint, the project guide approved a shift to a fully simulated dataset, carefully calibrated against real, published aggregate statistics, as a methodology demonstration rather than an empirical claim about real BMC candidates."),
        h2("2.3 Candidate Data Calibration"),
        p("The simulated candidate dataset (145 candidates across the 21 wards) was calibrated using:"),
        bullet("Real party seat-share weights from the actual 2026 BMC results (227 seats: BJP 89, SS-UBT 65, Shiv Sena 29, INC 24, AIMIM 8, MNS 6, NCP 3, SP 2, NCP-SP 1)"),
        bullet("Real ward-level gender/caste reservation categories for wards 8–28"),
        bullet("Criminal case rate, crorepati rate, and education breakdown calibrated against ADR/Maharashtra Election Watch's published analysis of real BMC 2026 winning candidates"),
        p("Candidate names are entirely fictional placeholders; no row represents a real person."),
        h2("2.4 Voter Perception Data Design"),
        p("210 simulated survey respondents were generated, each matched to one real (simulated) candidate from their own ward. A before/after paired design was used: a baseline favorability score, followed by simulated exposure to that candidate's disclosed criminal record and assets, followed by a post-exposure favorability score. Critically, the shift in favorability was not left to random chance — it followed a designed model in which serious criminal disclosures reduce favorability more than minor ones, the reduction is larger for respondents who were not already aware of the disclosure (encoding the core hypothesis under investigation), and wealth has a small, deliberately ambiguous effect (reflecting genuinely mixed findings in real survey research). This ensures the subsequent statistical analysis is a meaningful test of the pipeline's ability to recover a designed relationship, not an analysis of noise."),
        h2("2.5 Model Validation Approach"),
        p("Given the small sample size (145 candidates, of which only ~14.5% are winners — one per ward), a single train/test split produces unstable performance metrics. Stratified 5-fold cross-validation was used instead, preserving the win/loss ratio across folds, with class_weight='balanced' to address the class imbalance."),

        h1("3. Candidate Disclosure Analysis — Findings"),
        p(`Across the 145 candidates analyzed, approximately 21% had at least one declared criminal case, and 79% qualified as "crorepatis" (declared assets exceeding ₹1 crore) — figures deliberately calibrated to match ADR's real published statistics for BMC 2026 winners.`),
        ...figure("01_criminal_cases_by_party.png", "Figure 1: Criminal case disclosure rate by party."),
        ...figure("02_asset_distribution.png", "Figure 2: Distribution of declared candidate assets, with the crorepati threshold marked."),
        ...figure("03_education_breakdown.png", "Figure 3: Candidate education level breakdown."),
        ...figure("04_ward_reservation.png", "Figure 4: Ward reservation categories across the 21 analyzed wards."),
        p("Notably, the ward set analyzed is heavily weighted toward Women and OBC-Women reservation categories — a genuine structural feature of India's local-body reservation system, not a data artifact, meaning only women were legally eligible to contest in a majority of these wards."),

        h1("4. Win-Prediction Model — Findings"),
        p("A logistic regression model was trained to predict electoral outcome (won/lost) using only disclosure-profile features: party, log-transformed assets and income, education level, criminal case severity (ordinal: none / minor / serious), and age. Gender was deliberately excluded to avoid confounding with ward-level reservation, which mechanically determines candidate gender in many wards."),
        ...figure("05_cv_performance.png", "Figure 5: Cross-validated model performance across 5 stratified folds, with standard deviation error bars."),
        p("The model achieves a mean ROC-AUC of 0.81 (±0.09 across folds) — indicating it reliably distinguishes winners from losers well above chance, using disclosure information alone. Precision is comparatively low (0.34) because class_weight='balanced' deliberately trades precision for recall given the rarity of winners in the dataset; recall (0.81) indicates the model successfully identifies the great majority of actual winners."),
        ...figure("06_feature_importance.png", "Figure 6: Logistic regression coefficients, indicating each feature's direction and relative influence on win probability."),
        p("Party affiliation is the dominant predictor — consistent with real BMC 2026 seat-share patterns used to calibrate the simulation. Among disclosure-specific features, higher education and higher declared assets are associated with increased win probability, while criminal case severity and higher declared income are associated with decreased win probability."),
        ...figure("07_roc_curve.png", "Figure 7: ROC curve for the win-prediction model on a held-out test set."),

        h1("5. Voter Perception Study — Findings"),
        p("Of the 210 simulated respondents, 75.7% were not already aware of their matched candidate's disclosures prior to the survey — establishing a substantial baseline awareness gap that motivates the rest of this analysis."),
        ...figure("10_awareness_rate.png", "Figure 8: Proportion of respondents aware of candidate disclosures prior to exposure."),
        ...figure("08_favorability_before_after.png", "Figure 9: Distribution of favorability scores before and after exposure to disclosure details."),
        p("A paired t-test confirms that exposure to disclosure details produces a statistically significant shift in favorability (mean change: −0.39, t = −4.69, p < 0.00001)."),
        ...figure("09_awareness_moderation.png", "Figure 10: Mean favorability change for candidates with a serious criminal case, split by whether the respondent was already aware."),
        p("Among respondents evaluating a candidate with a serious criminal case, those who were not already aware showed a mean favorability drop of −3.72 points, compared to −0.99 points among those already aware — a difference confirmed by a linear regression controlling for other factors (already_aware coefficient: +0.49, p = 0.001), indicating prior awareness significantly cushions the negative impact of new disclosure information."),
        p("Regression results also show that criminal case severity is the strongest driver of favorability change (coefficient −2.03 for serious cases, p < 0.001), while candidate assets, education, and respondent age show no statistically significant effect — indicating that not every disclosed attribute swings voter opinion equally."),

        h1("6. Discussion & Synthesis"),
        p("Read together, the two components of this study suggest a coherent, if simulated, narrative: candidates with serious criminal disclosures face a real electoral penalty in the win-prediction model, and the perception study offers a plausible behavioral mechanism for why — such disclosures measurably reduce voter favorability, especially among voters newly exposed to the information. This suggests that low real-world voter awareness of disclosures (a genuine, well-documented phenomenon this project's design reflects) may partially explain why candidates with criminal records nonetheless continue to win seats in Indian local body elections: if most voters remain unaware, the disclosure system's intended deterrent effect is muted."),

        h1("7. Limitations"),
        bullet("All candidate and respondent data is simulated, not collected from real individuals. Findings describe the internal consistency of a designed simulation, not verified facts about real Mumbai voters or candidates."),
        bullet("Sample size (145 candidates, 210 respondents) is small; classification metrics such as precision carry meaningful variance across cross-validation folds."),
        bullet("Real-world data collection via the SEC portal was attempted but not completed at scale within the project timeline; the exploratory scraper is included for documentation but was not the source of the final analyzed dataset."),
        bullet("The perception study's designed relationships (e.g., the awareness-moderation effect) were built into the data generation process; the analysis validates that the pipeline can correctly recover such a relationship, which is a methodology demonstration rather than an independent empirical discovery."),
        bullet("Expenditure disclosure data was not incorporated, as no reliable public index was located within the project timeline."),

        h1("8. Conclusion"),
        p("This project demonstrates a complete, technically sound pipeline for analyzing political disclosure data — from calibrated data generation, through statistical hypothesis testing and cross-validated machine learning, to an interactive dashboard — applied to the 2026 BMC election context. While built on simulated rather than fully collected real-world data due to practical constraints, the methodology, calibration against real published statistics, and designed validation of hypothesized relationships together constitute a rigorous demonstration of the analytical approach the topic calls for, and provide a template directly extensible to real affidavit data should full-scale collection become feasible."),

        h1("9. References"),
        p("Association for Democratic Reforms (ADR) / Maharashtra Election Watch. Analysis of Criminal Background, Financial, Education, Gender and other details of Winning Candidates in BMC Elections 2026. February 2026.", { size: 20 }),
        p("Maharashtra State Election Commission. Affidavit of Final Contesting Candidate portal. mahasecelec.in", { size: 20 }),
        p("State Election Commission, Maharashtra. Brihanmumbai Municipal Corporation General Election results, 15 January 2026.", { size: 20 }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("Political_Disclosure_Analysis_Report.docx", buf);
  console.log("Report written.");
});
