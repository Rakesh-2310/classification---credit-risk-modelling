# 💳 Credit Risk Prediction System (ML + Deployment)

<p align="center">
  <b>End-to-End Machine Learning System to Predict Loan Default & Generate Credit Scores</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python"/>
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange"/>
  <img src="https://img.shields.io/badge/Model-XGBoost-green"/>
  <img src="https://img.shields.io/badge/Tuning-Optuna-red"/>
  <img src="https://img.shields.io/badge/Deployment-Streamlit-ff4b4b?logo=streamlit"/>
  <img src="https://img.shields.io/badge/Imbalance-SMOTE--Tomek-yellow"/>
</p>

---

## 🚀 Overview

This project builds a **production-ready credit risk system** to predict whether a customer will **default on a loan (0/1)** and extends the prediction to generate **credit scores (300–900)** and **risk ratings (Poor → Excellent)**.

> 💡 Designed to simulate a real-world **NBFC (Non-Banking Financial Company)** credit decisioning system.

---

## 🎯 Problem Statement

Financial institutions must:

* Identify **high-risk borrowers**
* Minimize **default losses**
* Maintain **approval rates**

👉 This system helps automate **risk assessment** using machine learning.

---

## 🧠 Key Highlights

* ⚡ **94% Recall on Defaulters** (high-risk detection)
* ⚖️ Handles imbalance using **SMOTE-Tomek**
* 📊 Industry metrics: **AUC, KS, Gini**
* 💳 Converts predictions → **Credit Score + Rating**
* 🖥️ Real-time **Streamlit Deployment**
* 🔒 Prevents **Data Leakage (proper pipeline design)**

---

## 🏗️ System Architecture

```text
User Input (Streamlit UI)
        ↓
Feature Engineering
        ↓
Scaling (MinMaxScaler)
        ↓
Trained Model (Logistic / XGBoost)
        ↓
Default Probability
        ↓
Credit Score (300–900)
        ↓
Risk Category (Poor → Excellent)
```

---

## ⚙️ ML Pipeline

### 🔹 1. Data Processing

* Missing value handling
* Outlier treatment
* Business rule validation

### 🔹 2. Feature Engineering

* Loan-to-Income Ratio
* Delinquency Ratio
* Credit Utilization
* Avg DPD per Delinquency

### 🔹 3. Data Leakage Prevention

* Train-test split **before preprocessing**
* Ensured no leakage in transformations

### 🔹 4. Feature Selection

* Correlation filtering
* **VIF (Variance Inflation Factor)**

### 🔹 5. Imbalanced Data Handling

* **SMOTE-Tomek** (oversampling + cleaning)

### 🔹 6. Model Training

* Logistic Regression (interpretable)
* XGBoost (high performance)

### 🔹 7. Hyperparameter Tuning

* RandomizedSearchCV
* **Optuna (efficient search)**

### 🔹 8. Evaluation Metrics

* ROC-AUC
* KS Statistic
* Gini Coefficient

---

## 💳 Credit Score Logic

* Probability → Score mapping:

  ```
  Score = 300 + (1 - PD) × 600
  ```
* Rating categories:

  * Poor
  * Average
  * Good
  * Excellent

---

## 📂 Project Structure

```bash
├── credit_risk_model.ipynb     # Model development
├── main.py                    # Streamlit UI
├── prediction_helper.py       # Preprocessing + inference logic
├── model_data.joblib          # Model + scaler + features
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```bash
git clone
pip install -r requirements.txt
streamlit run main.py
```

---

## 📊 Model Performance

| Metric           | Value             |
| ---------------- | ----------------- |
| Recall (Default) | **94%**           |
| AUC Score        | High              |
| KS Statistic     | Strong separation |
| Gini             | High              |

---

## ⚠️ Limitations

* Manual preprocessing (no sklearn pipeline)
* No experiment tracking (MLflow)
* Limited handling of unseen categories

---

## 🔮 Future Improvements

* 🔄 End-to-end **sklearn Pipeline**
* 📈 **SHAP Explainability**
* ⚙️ **MLflow tracking**
* 🚀 FastAPI deployment
* 🐳 Dockerization