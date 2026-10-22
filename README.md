# Olist E-Commerce Analytics & Delay Prediction

An end-to-end **Data Analytics** and **Predictive Machine Learning** project analyzing delivery logistics and customer satisfaction for Brazilian e-commerce platform **Olist** (~96,000 orders).

> 🇮🇹 **Note for Italian speakers:** The repository documentation (README) is written in English for international accessibility. The complete Jupyter Notebook (`olist_delay_prediction.ipynb`) and code comments are written in Italian.

---

## Executive Summary
In e-commerce logistics, delivery delays are the primary driver of customer dissatisfaction. Analyzing over 96,000 orders reveals that delivery delays account for **6.66% of total orders (~6,380 orders)**, but cause a **47% drop in customer review scores** (from **4.29 to 2.27 stars**).

This project develops an end-to-end Machine Learning pipeline to **predict delayed orders at checkout**, enabling proactive intervention before dispatch.

---

## Key Business Findings
* **Global Delay Rate:** 6.66% of total transactions.
* **Customer Rating Impact:** Ratings plummet from 4.29 stars (on-time) to 2.27 stars (delayed).
* **Geographical Disparity:** Northern and Northeastern Brazilian states (e.g., Alagoas, Maranhão) record delay rates exceeding 15–20%, compared to 4.4% in São Paulo (SP).
* **Primary Drivers:** Order financial value (`totale_pagamento`), estimated delivery window (`stima_giorni_consegna`), and seasonality are the strongest predictors of delivery bottlenecks.

---

## Tech Stack & Methodology
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn, XGBoost (`XGBClassifier`, `RandomForestClassifier`, `LogisticRegression`)
* **Database & Querying:** SQLite, SQL

---

## Machine Learning Model Comparison

Due to severe class imbalance (6.66% positive class `1`), models were evaluated using **ROC-AUC**, **Recall**, **Precision**, and **F1-Score**:

| Model | Setup / Class Weight | Accuracy | Recall (Class 1) | Precision (Class 1) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | `class_weight='balanced'` | 65% | 61% | 11% | 0.19 | 0.6794 |
| **Random Forest Classifier** | Threshold = 0.15 | 85% | 36% | 18% | 0.24 | 0.6993 |
| **XGBoost Classifier (Winner)** | `scale_pos_weight=ratio` | **77%** | **62%** | **17%** | **0.26** | **0.7600** |

> **Champion Model:** **XGBoost** achieved the highest discriminative power (**ROC-AUC 0.7600**), capturing **62% of real delays** at checkout with an optimal F1-score of **0.26**.

---

## Strategic Business Recommendations
1. **High-Value Order Pre-Alerting:** Implement automated tracking flags for high-value orders (`totale_pagamento`) bound for remote or historically delayed states (e.g., RJ and Northeast region).
2. **Dynamic Seasonal Buffers:** Automatically extend estimated delivery dates (`stima_giorni_consegna`) by 2–3 days during peak sales months (e.g., Black Friday period) to align customer expectations and protect review scores.
3. **Regional SLA Renegotiation:** Review carrier Service Level Agreements (SLAs) and local fulfillment nodes in critical regions like Rio de Janeiro (RJ).

---

## Repository Structure
```text
olist-delay-prediction/
├── data/                       # Raw and processed datasets (.csv)
│   └── ecommerce_clean.csv
├── notebooks/                  # Jupyter notebooks containing full EDA & ML pipeline
│   └── olist_delay_prediction.ipynb
├── .gitignore                  # Git ignore rules for data files and checkpoints
├── README.md                   # Executive project summary
└── requirements.txt            # Python dependencies