# AI-IDS-Project: ML/DL-Based Network Intrusion Detection System

An intrusion detection system trained on the CICIDS2017 dataset, comparing classical machine learning, deep learning, and unsupervised anomaly detection approaches, with SHAP-based explainability to interpret model decisions.

## Pipeline

The project follows a full ML workflow, organized as sequential notebooks:

1. **Data Exploration** — analysis of label distribution and class imbalance (benign vs. attack traffic)
2. **Preprocessing** — feature scaling, label encoding, and cleaning
3. **Classical ML** — KNN, Logistic Regression, Random Forest, XGBoost
4. **Deep Learning** — 1D-CNN and Dense Neural Network architectures
5. **Autoencoder** — unsupervised anomaly detection via reconstruction error
6. **SHAP Explainability** — feature importance and per-prediction interpretability across models

## Results — Classical ML Models

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 99.87% | 99.88% | 99.87% | 99.87% |
| Random Forest | 99.37% | 99.78% | 99.37% | 99.55% |
| KNN | 98.53% | 98.68% | 98.53% | 98.60% |
| Logistic Regression | 97.76% | 98.05% | 97.76% | 97.81% |

XGBoost achieved the best overall performance, with Random Forest close behind. Full confusion matrices and training curves for all models (including 1D-CNN, Dense NN, and the autoencoder) are available in `/reports`.

## Explainability

SHAP (SHapley Additive exPlanations) was used to interpret model predictions, identifying which network traffic features most influence the classification of an event as benign or malicious — critical for trust and adoption of ML-based detection in real SOC workflows.

## Anomaly Detection

An autoencoder was trained separately to detect intrusions via reconstruction error, providing an unsupervised approach that doesn't rely on labeled attack data — useful for detecting novel attack patterns not seen during training.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — a labeled network intrusion dataset covering common attack types (DoS, DDoS, brute force, infiltration, botnet, and web attacks) alongside benign traffic.

## Tech Stack

Python, scikit-learn, Keras/TensorFlow, XGBoost, SHAP, pandas

## Project Structure

```
├── app/                    # Flask dashboard for model results
├── data/                   # Processed data, encoders, scalers
├── notebooks/              # Sequential analysis pipeline (01-06)
└── reports/                # Visualizations: confusion matrices, training history, comparisons
```
