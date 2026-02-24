# Ad Creative Fatigue Detector

A machine learning web application that detects whether a digital ad creative is fatiguing — and explains exactly why — using XGBoost and SHAP.

Built as a marketing ML portfolio project to demonstrate how AI can help marketers make smarter, faster decisions about when to refresh ad creatives before performance tanks.

---

## The Problem

Most marketing teams only realize an ad has fatigued after performance has already collapsed. By then, budget has been wasted and results have suffered. This tool acts as an **early warning system** — flagging creatives that are showing fatigue signals before it's too late to act.

---

## How It Works

The model analyzes campaign performance metrics and compares them against campaign-level benchmarks to detect decay patterns. An ad is flagged as fatiguing when at least two of three conditions are met:

- CTR has dropped below 90% of the campaign's own average
- Engagement has dropped below 90% of the campaign's own average  
- ROI falls in the bottom 35th percentile

This rule-based label was used to train an XGBoost classifier on 300,000 ad campaign records, achieving a **ROC-AUC of 0.99** on the test set.

---

## Features

- Real-time fatigue prediction with probability score
- SHAP waterfall chart explaining which features drove the prediction
- Input summary dashboard showing key computed metrics
- Clean dark-themed UI built with Streamlit
- Handles class imbalance (6% fatigue rate) via `scale_pos_weight` and a calibrated 0.3 decision threshold

---

## Tech Stack

| Layer | Tools |
|---|---|
| Model | XGBoost, scikit-learn |
| Explainability | SHAP |
| Frontend | Streamlit |
| Data Processing | pandas, numpy |
| Visualization | matplotlib |

---

## Project Structure

```
ad-creative-fatigue-detector/
│
├── data/
│   └── raw/
│       └── Social_Media_Advertising.csv
├── notebooks/
│   └── eda.ipynb              # EDA, feature engineering, model training
├── src/
│   ├── features.py            # Feature engineering logic
│   ├── train.py               # Model training pipeline
│   └── predict.py             # Prediction and SHAP explanation logic
├── model/
│   └── xgb_model.pkl          # Trained XGBoost model
├── app.py                     # Streamlit application
└── requirements.txt
```

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.9925 |
| Recall (Fatigued) | 0.98 |
| Precision (Fatigued) | 0.52 |
| Decision Threshold | 0.30 |

High recall was prioritized over precision — in a marketing context, missing a fatigued ad is more costly than over-flagging a healthy one.

---

## Key ML Concepts Demonstrated

**Feature Engineering** — CTR decay, engagement decay, and spend efficiency were engineered from raw campaign metrics rather than used directly, capturing relative performance trends rather than absolute values.

**Data Leakage Prevention** — ROI and spend efficiency were excluded from the final feature set after being identified as leakage sources (they were used to construct the fatigue label).

**Class Imbalance Handling** — With only 6% positive cases, `scale_pos_weight` was used during training and the decision threshold was lowered to 0.3 to improve recall on the minority class.

**Explainability** — SHAP waterfall plots provide per-prediction feature attribution, making the model's reasoning transparent to non-technical marketing stakeholders.

---

## Dataset

**Social Media Advertising Dataset** — 300,000 fictional ad campaign records across Facebook, Instagram, Pinterest, and Twitter. Includes impressions, clicks, engagement scores, conversion rates, acquisition costs, ROI, and demographic targeting data.

---

## Installation

```bash
git clone https://github.com/yourusername/ad-creative-fatigue-detector.git
cd ad-creative-fatigue-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## Usage

1. Enter your campaign's performance metrics in the left panel
2. Set CTR vs Campaign Average and Engagement vs Campaign Average using the sliders — values below 0.85 signal potential decay
3. Select your campaign details
4. Click **Analyze Campaign**
5. View the fatigue probability, result card, and SHAP explanation on the right

---

*Built by Aryamani Boruah | Marketing ML Portfolio*
