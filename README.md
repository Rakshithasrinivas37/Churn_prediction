# 📉 Customer Churn Prediction

A machine learning project that predicts customer churn using Logistic Regression,
served via a Flask REST API, containerized with Docker, and automated with a
GitHub Actions CI/CD pipeline.

---

## 📌 Project Overview

Customer churn refers to when a customer stops using a company's service.
This project builds a binary classification model to predict whether a
customer is likely to churn based on their demographic and service usage data.

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML | scikit-learn, numpy, pandas |
| API | Flask, Gunicorn |
| Containerization | Docker, GitHub Container Registry |
| CI/CD | GitHub Actions |

---

## 🔄 CI/CD Pipeline

Every push to `main` automatically triggers:
```
Push Code
    │
    ▼
✅ Train Model        (python src/train.py)
    │
    ▼
✅ Run Tests          (python tests/test.py)
    │
    ▼
✅ Build Docker Image
    │
    ▼
🐳 Push to ghcr.io
```

---

## 📈 Results

| Metric | Value |
|---|---|
| ROC-AUC Score | 0.86 |
| Validation Strategy | 5-Fold Stratified CV |
| Churn Rate in Dataset | 26.5% |

---