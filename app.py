"""
Credit Card Fraud Detection - Flask Web Application
=====================================================
Serves the prediction API and admin dashboard.
Run: python app.py
"""

import os
import json
import base64
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model bundle ──
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'fraud_model.pkl')
bundle = None

def load_model():
    global bundle
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        print(f"[APP] Model loaded from {MODEL_PATH}")
    else:
        print("[APP] ⚠  Model not found. Run train_model.py first.")

load_model()


def img_to_b64(path):
    """Convert a plot PNG to base64 string for embedding in JSON."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main SPA."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Body: { "Time": float, "Amount": float, "V1": float, ..., "V28": float }
    Returns: { "prediction": 0|1, "probability": float, "label": str }
    """
    if bundle is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body received.'}), 400

    try:
        model        = bundle['model']
        scaler       = bundle['scaler']
        feature_cols = bundle['feature_cols']

        # Build feature vector in correct column order
        row = {}
        for col in feature_cols:
            val = data.get(col, 0.0)
            row[col] = float(val)

        import pandas as pd
        df_row = pd.DataFrame([row])

        # Scale Amount and Time (same as training)
        df_row[['Amount', 'Time']] = scaler.transform(df_row[['Amount', 'Time']])

        prediction  = int(model.predict(df_row)[0])
        probability = float(model.predict_proba(df_row)[0][1])

        return jsonify({
            'prediction':  prediction,
            'probability': round(probability * 100, 2),
            'label':       'FRAUD' if prediction == 1 else 'SAFE',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dashboard')
def dashboard():
    """
    GET /api/dashboard
    Returns model metrics, dataset stats, and base64-encoded chart images.
    """
    if bundle is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
    charts = {
        'class_distribution':  img_to_b64(f'{plot_dir}/class_distribution.png'),
        'amount_distribution': img_to_b64(f'{plot_dir}/amount_distribution.png'),
        'feature_correlations':img_to_b64(f'{plot_dir}/feature_correlations.png'),
        'confusion_matrix':    img_to_b64(f'{plot_dir}/confusion_matrix.png'),
        'feature_importance':  img_to_b64(f'{plot_dir}/feature_importance.png'),
    }

    return jsonify({
        'metrics': bundle.get('metrics', {}),
        'stats':   bundle.get('stats', {}),
        'charts':  charts,
    })


@app.route('/api/status')
def status():
    """Health-check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': bundle is not None})


if __name__ == '__main__':
    print("\n🚀  Starting Fraud Detection Server on http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
