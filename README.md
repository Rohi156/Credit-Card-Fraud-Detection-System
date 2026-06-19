# 🛡️ FraudGuard — Live Credit Card Fraud Detection System

A complete end-to-end Machine Learning project that detects credit card fraud in real-time via a dark-themed web application.

---

## 📁 Project Structure

```
fraud_detection/
├── train_model.py       # ML pipeline: data generation, SMOTE, training, evaluation
├── app.py               # Flask web server (API + dashboard)
├── requirements.txt     # Python dependencies
├── models/
│   └── fraud_model.pkl  # Saved model bundle (generated after training)
├── plots/               # EDA & evaluation charts (generated after training)
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── feature_correlations.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── templates/
    └── index.html       # Single-page web application
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Generate 50,000 synthetic transactions (~0.5% fraud)
- Apply **SMOTE** to balance the training data
- Train a **Random Forest Classifier**
- Print F1-Score, ROC-AUC, and a full classification report
- Save `models/fraud_model.pkl` and 5 chart PNGs to `plots/`

### 3. Launch the Web App

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## 🧠 ML Pipeline Details

| Component | Choice | Reason |
|-----------|--------|--------|
| Algorithm | Random Forest Classifier | Robust to outliers, handles non-linear patterns, built-in feature importance |
| Imbalance handling | SMOTE + `class_weight='balanced'` | Synthetic oversampling of minority class in training set only |
| Evaluation | F1-Score + Confusion Matrix + ROC-AUC | Accuracy alone is misleading on imbalanced data |
| Scaling | StandardScaler on Amount & Time | Prevents magnitude bias |

### Model Results (Test Set)
- **Accuracy**: 99.99%
- **F1-Score (Fraud)**: ~0.99
- **ROC-AUC**: ~1.00

---

## 🌐 Web Application Features

### Live Detection Tab
- Enter `Amount`, `Time`, and V1–V28 PCA features
- Three quick presets: ✅ Safe Transaction, 🚨 Fraud Pattern, 🎲 Random
- **Green "TRANSACTION APPROVED"** for safe predictions
- **Red "🚨 FRAUD DETECTED! CARD BLOCKED"** alert with animated glow for fraud
- Fraud probability bar chart
- Recent transaction history table

### Admin Dashboard Tab
- 8 key metrics (total transactions, accuracy, F1-score, ROC-AUC, etc.)
- Class distribution chart
- Amount distribution (legit vs fraud)
- Feature correlation heatmap
- Confusion matrix
- Top 15 feature importances

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web application |
| POST | `/api/predict` | Predict transaction class |
| GET | `/api/dashboard` | Model metrics + chart images |
| GET | `/api/status` | Health check |

### POST /api/predict — Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Amount": 1842, "Time": 50000, "V1": -4.2, "V14": -6.1, "V4": 6.1}'
```

Response:
```json
{
  "prediction": 1,
  "probability": 97.34,
  "label": "FRAUD"
}
```

---

## 📦 Dependencies

```
flask, pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib
```

---

## 💡 Notes

- The synthetic dataset mimics the structure of the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). To use real data, replace `generate_dataset()` in `train_model.py` with `pd.read_csv('creditcard.csv')`.
- All SMOTE resampling is applied **only to the training set** to prevent data leakage.
- The `.pkl` file stores the model, scaler, feature names, metrics, and dataset stats as a single bundle.
