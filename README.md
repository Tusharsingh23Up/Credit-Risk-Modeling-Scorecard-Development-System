# 🏦 Credit Risk Modeling & Scorecard Development System

![Python](https://img.shields.io/badge/Python-Risk%20Analytics-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange?style=for-the-badge)
![Banking](https://img.shields.io/badge/Banking-Credit%20Risk-0A66C2?style=for-the-badge)
![Scorecard](https://img.shields.io/badge/Scorecard-WoE%20%7C%20IV-2E8B57?style=for-the-badge)

> **A credit risk analytics project for building a loan default prediction model, developing a scorecard, segmenting customer risk, and supporting lending decisions.**

---

## 🚀 Project Overview

This project focuses on **credit risk modeling**, a key analytics use case in banking, fintech, NBFCs, and lending institutions.

The system uses borrower-level data to estimate default probability and create a structured credit scorecard using feature engineering, Weight of Evidence, Information Value, logistic regression, risk segmentation, and model performance evaluation.

---

## 🎯 Business Problem

Lenders need to evaluate whether a borrower is likely to repay a loan. Poor credit decisions can increase default losses, while overly strict decisions can reduce business growth.

This project supports credit decisioning by predicting default probability, classifying customers by risk level, improving loan approval decisions, and creating transparent scorecard logic.

---

## ✅ Key Features

| Feature | Description |
|---|---|
| Data Cleaning | Handles missing values, outliers, and inconsistent records |
| Feature Engineering | Creates borrower-level risk variables |
| WoE Transformation | Converts variables into risk-friendly bins |
| IV Calculation | Measures predictor strength |
| Logistic Regression | Predicts default probability |
| Scorecard Development | Converts model output into business-friendly scores |
| Risk Segmentation | Groups borrowers into low, medium, and high risk |
| Model Evaluation | Uses ROC-AUC, KS statistic, confusion matrix, and accuracy |

---

## 📊 Credit Risk Workflow

```text
Borrower Data → Data Cleaning → Feature Engineering → WoE / IV Analysis → Logistic Regression → Probability of Default → Credit Scorecard → Risk Segmentation
```

---

## 📌 Key Metrics

- Probability of Default
- ROC-AUC Score
- KS Statistic
- Accuracy
- Precision
- Recall
- F1 Score
- Information Value
- Credit Score
- Risk Band

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Modeling | Logistic Regression |
| Risk Analytics | WoE, IV, Scorecard |
| Visualization | Matplotlib, Plotly |
| Notebook | Jupyter Notebook |

---

## 📁 Project Structure

```text
credit-risk-scorecard-model/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── credit_risk_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── woe_iv.py
│   ├── modeling.py
│   └── scorecard.py
├── reports/
│   └── model_summary.pdf
├── assets/
│   └── screenshots/
├── requirements.txt
└── README.md
```

---

## 📸 Screenshots

```markdown
![Model Performance](assets/screenshots/model-performance.png)
![Scorecard Output](assets/screenshots/scorecard-output.png)
![Risk Segmentation](assets/screenshots/risk-segmentation.png)
```

---

## 📈 Example Output

| Customer ID | Probability of Default | Credit Score | Risk Segment |
|---|---:|---:|---|
| CUST001 | 0.08 | 742 | Low Risk |
| CUST002 | 0.31 | 615 | Medium Risk |
| CUST003 | 0.67 | 488 | High Risk |

---

## 💼 Business Impact

This project demonstrates how data analytics can improve lending decisions by reducing default risk, standardizing credit evaluation, creating explainable risk scores, and supporting portfolio monitoring.

---

## 🚀 How to Run

```bash
git clone https://github.com/Tusharsingh23Up/Credit-Risk-Modeling-Scorecard-Development-System.git
cd Credit-Risk-Modeling-Scorecard-Development-System
pip install -r requirements.txt
```

---

## 🔮 Future Enhancements

- XGBoost and Random Forest comparison
- Reject inference
- Model monitoring dashboard
- PSI stability tracking
- Automated credit policy rules
- Streamlit deployment

---

## 👤 Author

**Tushar Singh**  
Financial Analyst Aspirant | Finance + Data Analytics + AI  
GitHub: [Tusharsingh23Up](https://github.com/Tusharsingh23Up)

