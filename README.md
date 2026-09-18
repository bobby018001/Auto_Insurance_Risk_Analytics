# Auto Insurance Risk Analytics

An end-to-end machine-learning project for automobile insurance pricing and risk assessment. The pipeline predicts three actuarial outcomes:

- **LC** — loss cost per exposure unit
- **HALC** — historically adjusted loss cost
- **CS** — binary claim status

The project combines leakage-aware preprocessing, Tweedie regression, imbalanced classification, model comparison, SHAP interpretation, and business-risk analysis across 39,928 anonymized training policies.

## Results from the original project run

| Target | Selected model | Validation metric |
| --- | --- | --- |
| LC | LightGBM with Tweedie objective | MSE 254,710 |
| HALC | LightGBM with Tweedie objective | MSE 996,779 |
| CS | Weighted XGBoost classifier | ROC-AUC 0.857 |

These are historical holdout results from the supplied project materials. They should be reproduced before reuse and do not establish performance on another insurer, period, or jurisdiction.

## Pipeline

1. Clean and validate anonymized policy data.
2. Learn outlier bounds from training data only.
3. Engineer policy-tenure, renewal-timing, cancellation, premium, vehicle-age, and power-to-weight features.
4. Exclude target components that would leak future information.
5. Compare Tweedie GLM, XGBoost, LightGBM, and a boosting ensemble for LC and HALC.
6. Compare logistic regression, random forest, and weighted XGBoost for claim status.
7. Retrain selected models and explain predictions with SHAP.
8. Produce aligned LC, HALC, and CS predictions for the test portfolio.

## Repository structure

```text
Auto_Insurance_Risk_Analytics/
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_claim_status_classification.ipynb
│   ├── 03_loss_cost_regression.ipynb
│   └── 04_shap_and_predictions.ipynb
├── figures/
│   ├── eda_overview.png
│   ├── shap_CS_summary.png
│   ├── shap_HALC_summary.png
│   ├── shap_LC_HALC_bar.png
│   └── shap_LC_summary.png
├── poster/
│   └── insurance_loss_analytics_poster.html
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Obtain the authorized course data and place it according to [data/README.md](data/README.md). Run the notebooks in numeric order. Stored outputs and Colab execution metadata were removed from the public copies, and environment-specific Drive paths were replaced with repository-relative paths.

## Interpretation

The supplied SHAP analysis identifies cancellation behavior, policy tenure, renewal timing, and net premium as important signals across the targets. These findings support pricing and retention analysis, but SHAP values describe model behavior rather than causal effects.

## Limitations and responsible use

- The raw feature names are anonymized, limiting independent semantic validation.
- The 88.85% zero-claim rate creates severe class imbalance and a mixed outcome distribution.
- MSE is sensitive to heavy-tailed losses and should be paired with segment-level diagnostics.
- A single random holdout may overstate stability when insurance data shifts over time.
- Age, vehicle, payment, or behavioral variables may be regulated or act as proxies for protected characteristics.
- Production use requires fairness testing, calibration, temporal validation, drift monitoring, and actuarial/legal review.
- The models are specific to automobile insurance and should not be transferred directly to life or health insurance.

## Contribution scope

This was a team academic project. The repository owner reports responsibility for the complete code implementation, including preprocessing, feature engineering, regression and classification pipelines, model selection, prediction generation, and SHAP analysis. The report, poster, and business interpretation were collaborative team deliverables.

## Data availability

The course datasets and prediction submissions are not redistributed because the supplied archive does not include a public redistribution license. The code and schema instructions remain available so authorized users can reconstruct the analysis.

No standalone license is attached to the project-authored materials; copyright remains with the original contributors.
