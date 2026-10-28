# Olist E-Commerce Analytics & Delay Prediction App

An end-to-end **Data Analytics**, **Predictive Machine Learning**, and **Streamlit Web Application** project analyzing delivery logistics and customer satisfaction for Brazilian e-commerce platform **Olist** (~99,400 orders).

> 🇮🇹 **Note for Italian speakers:** The repository documentation (README) is written in English for international accessibility. The complete Jupyter Notebook (`olist_delay_prediction.ipynb`), Streamlit App (`app.py`), and code comments are written in Italian.

---

## Interactive Streamlit Web App

The project includes an interactive web dashboard built with **Streamlit** that allows business stakeholders to analyze regional logistics risks and predict delivery delay probabilities at checkout in real time.

### Key Web App Features:
1. **Executive KPI Dashboard:** Real-time business metrics comparing overall delay rate (6.66%) vs. review scores (4.29 ⭐ vs 2.27 ⭐).
2. **Interactive Logistics Map:** A dynamic South American map powered by **Plotly Express** visualizing delay rates across all 27 Brazilian states (highlighting high-risk regions like Alagoas at 20.81% and Maranhão at 17.13%).
3. **Real-Time Delay Risk Predictor:** An interactive form allowing users to input order details (financial value, estimated delivery days, destination state, purchase month, and day of week) to receive an instant risk assessment (Low, Moderate, High) powered by our trained **XGBoost** model.

### Application Screenshots
| Executive KPIs & Delay Map | Delay Risk Predictor |
| :---: | :---: |
| ![KPIs and Map](path/to/foto1.png) | ![Predictor](path/to/foto2.png) |

---

## Executive Summary
In e-commerce logistics, delivery delays are the primary driver of customer dissatisfaction. Analyzing over 99,000 orders reveals that delivery delays account for **6.66% of total orders (~6,380 orders)**, but cause a **47% drop in customer review scores** (from **4.29 to 2.27 stars**).

This project develops an end-to-end Machine Learning pipeline to **predict delayed orders at checkout**, enabling proactive intervention before dispatch.

---

## Key Business Findings
* **Global Delay Rate:** 6.66% of total transactions.
* **Customer Rating Impact:** Ratings plummet from 4.29 stars (on-time) to 2.27 stars (delayed).
* **Geographical Disparity:** Northern and Northeastern Brazilian states (e.g., Alagoas, Maranhão, Sergipe) record delay rates exceeding 14–20%, compared to 4.43% in São Paulo (SP).
* **Primary Drivers:** Order financial value (`totale_pagamento`), estimated delivery window (`stima_giorni_consegna`), destination state, and seasonality are the strongest predictors of delivery bottlenecks.

---

## Tech Stack & Methodology
* **Language:** Python 3.10+
* **Web Framework:** Streamlit
* **Data Processing & Analytics:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn, XGBoost (`XGBClassifier`, `RandomForestClassifier`, `LogisticRegression`), Joblib
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

## How to Run the App Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/olist-delay-prediction.git](https://github.com/YOUR_USERNAME/olist-delay-prediction.git)
   cd olist-delay-prediction