# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project that detects fraudulent credit card transactions using **XGBoost**, handles highly imbalanced data, and explains model predictions using **SHAP**.

Built with Python, FastAPI and Streamlit, with a deployed backend and interactive dashboard.

## 🚀 Live Demo

| Component            | Link                                                                           |
| -------------------- | ------------------------------------------------------------------------------ |
| 🌐 Live Dashboard    | [Streamlit App](https://credit-card-fraud-detection-dhruv.streamlit.app/)      |
| ⚡ Backend API        | [FastAPI](https://credit-card-fraud-detection-pq0d.onrender.com/)              |
| 📘 API Documentation | [Swagger UI](https://credit-card-fraud-detection-pq0d.onrender.com/docs)       |
| 💻 Source Code       | [GitHub Repository](https://github.com/Dhruvb0101/credit-card-fraud-detection) |

## 📌 Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a very small percentage of total transactions.

This project builds a machine learning system to classify transactions into:

* **Legitimate (0):** Normal transactions.
* **Fraudulent (1):** Potentially fraudulent transactions.

The main focus is handling class imbalance, improving fraud detection, evaluating model performance, and providing explainable predictions.

## ✨ Key Features

* End-to-end machine learning pipeline.
* Exploratory Data Analysis (EDA).
* Handling highly imbalanced transaction data.
* XGBoost-based fraud classification.
* Optimized classification threshold.
* SHAP-based model explainability.
* FastAPI backend with single and batch prediction endpoints.
* Interactive Streamlit dashboard.
* Deployed frontend and backend.

## 🛠️ Tech Stack

| Category         | Technologies                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------- |
| Language         | [Python](https://www.python.org/)                                                                                   |
| Data Processing  | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)                                                   |
| Machine Learning | [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)                               |
| Imbalanced Data  | [Imbalanced-learn](https://imbalanced-learn.org/)                                                                   |
| Explainability   | [SHAP](https://shap.readthedocs.io/)                                                                                |
| Visualization    | [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| Frontend         | [Streamlit](https://streamlit.io/)                                                                                  |
| Backend          | [FastAPI](https://fastapi.tiangolo.com/)                                                                            |
| HTTP Requests    | [Requests](https://requests.readthedocs.io/)                                                                        |
| Model Storage    | [Joblib](https://joblib.readthedocs.io/)                                                                            |
| Deployment       | [Render](https://render.com/), [Streamlit Community Cloud](https://streamlit.io/cloud)                              |
| Version Control  | [Git](https://git-scm.com/), [GitHub](https://github.com/)                                                          |

## 📂 Dataset

The project uses the Credit Card Fraud Detection dataset from European cardholders.

**Dataset:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Original dataset statistics:

* Total transactions: 284,807
* Legitimate transactions: 284,315
* Fraudulent transactions: 492
* Fraud percentage: Approximately 0.17%

For model development, a stratified sample of 120,000 transactions was selected. After removing duplicates, the final dataset contained 119,798 transactions.

| Class      | Transactions |
| ---------- | -----------: |
| Legitimate |      119,596 |
| Fraudulent |          202 |
| Total      |      119,798 |

**Features:** `Time`, `V1`–`V28`, `Amount`, and `Class`.

The raw dataset is not included in the repository because of its large size.

## 🤖 Model & Performance

The project uses **XGBoost (Extreme Gradient Boosting)** for fraud classification.

The model was selected for its ability to learn complex patterns in structured data.

### Evaluation Results

| Metric                   |  Result |
| ------------------------ | ------: |
| Model                    | XGBoost |
| PR-AUC                   |  0.8696 |
| Precision                |  94.44% |
| Recall                   |  85.00% |
| Classification Threshold |  0.9939 |

### Evaluation Metrics

* **Precision:** Measures how many transactions predicted as fraud were actually fraudulent.
* **Recall:** Measures how many actual fraudulent transactions were detected.
* **PR-AUC:** Evaluates performance across different precision-recall thresholds.
* **Confusion Matrix:** Shows correct and incorrect classifications.

PR-AUC, precision and recall are particularly useful for evaluating highly imbalanced classification problems.

## 🔍 Model Explainability – SHAP

The project integrates SHAP (SHapley Additive exPlanations) to understand model predictions.

It helps identify how different transaction features influence the model's output.

Implemented explainability features include:

* Global feature importance.
* Feature contribution analysis.
* Individual prediction explanations.
* SHAP visualizations.

## ⚡ FastAPI Backend

The trained model is served through a FastAPI REST API.

**Available endpoints:**

| Method | Endpoint         | Purpose                       |
| ------ | ---------------- | ----------------------------- |
| GET    | `/`              | API welcome message           |
| GET    | `/health`        | Health check                  |
| POST   | `/predict`       | Single transaction prediction |
| POST   | `/predict_batch` | Batch transaction prediction  |

**API Documentation:** [Open Swagger UI](https://credit-card-fraud-detection-pq0d.onrender.com/docs)

## 📊 Streamlit Dashboard

The interactive dashboard allows users to explore the dataset and test fraud predictions.

Features include:

* Dataset overview and class distribution.
* Exploratory data visualizations.
* Model performance metrics.
* SHAP explainability.
* Sample transaction prediction.

The Streamlit frontend communicates with the deployed FastAPI backend using HTTP requests.

**Live Application:** [Open Dashboard](https://credit-card-fraud-detection-dhruv.streamlit.app/)

## 🏗️ System Architecture

```text
          User
           |
           v
    Streamlit Dashboard
           |
           | HTTP Request
           v
      FastAPI Backend
           |
           v
      XGBoost Model
           |
           v
    Fraud Prediction
           |
           v
    API Response
           |
           v
    Streamlit Dashboard
```

## 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── shap/
│
├── dashboard/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── sample/
│
├── src/
├── tests/
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Dhruvb0101/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside:

```text
data/raw/creditcard.csv
```

### 5. Train the Model

```bash
python app.py train --data-path data/raw/creditcard.csv --sample-size 120000 --disable-smoteenn --disable-mlflow
```

### 6. Run Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## 🔮 Future Improvements

* Experiment with additional classification algorithms.
* Improve fraud detection performance.
* Implement real-time transaction monitoring.
* Add model drift detection.
* Introduce automated model retraining.
* Improve API monitoring and security.

## 👨‍💻 Author

**Dhruv Agarwal**

B.Tech – Computer Science and Engineering

* GitHub: [Dhruvb0101](https://github.com/Dhruvb0101)
* Project: [Credit Card Fraud Detection](https://github.com/Dhruvb0101/credit-card-fraud-detection)

---

⭐ If you find this project interesting, consider giving the repository a star!
