# ============================================================
# CYBER ATTACK DETECTION SYSTEM — Flask Web App
# ============================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────
BASE_PATH   = r"C:\Users\hp\Desktop\Projets\cyber-ids-project"
MODELS_PATH = os.path.join(BASE_PATH, "models")
DATA_PATH   = os.path.join(BASE_PATH, "data")

# ── Load all models and artifacts on startup ──────────────────
print("Loading models...")

le           = joblib.load(os.path.join(DATA_PATH,   'label_encoder.pkl'))
scaler       = joblib.load(os.path.join(DATA_PATH,   'scaler.pkl'))
feature_cols = joblib.load(os.path.join(DATA_PATH,   'feature_columns.pkl'))
shap_imp     = joblib.load(os.path.join(MODELS_PATH, 'shap_importance.pkl'))
ae_threshold = joblib.load(os.path.join(MODELS_PATH, 'autoencoder_threshold.pkl'))

# Classical ML
knn_model = joblib.load(os.path.join(MODELS_PATH, 'knn_model.pkl'))
lr_model  = joblib.load(os.path.join(MODELS_PATH, 'lr_model.pkl'))
rf_model  = joblib.load(os.path.join(MODELS_PATH, 'rf_model.pkl'))
xgb_model = joblib.load(os.path.join(MODELS_PATH, 'xgb_model.pkl'))

# Deep learning
dense_model = keras.models.load_model(os.path.join(MODELS_PATH, 'dense_model.keras'))
cnn_model   = keras.models.load_model(os.path.join(MODELS_PATH, 'cnn_model.keras'))
autoencoder = keras.models.load_model(os.path.join(MODELS_PATH, 'autoencoder.keras'))

print("All models loaded!")

# ── Model registry ────────────────────────────────────────────
MODELS = [
    {'key': 'xgboost', 'name': 'XGBoost',             'type': 'sklearn', 'f1': 99.87, 'accuracy': 99.87},
    {'key': 'rf',      'name': 'Random Forest',        'type': 'sklearn', 'f1': 99.55, 'accuracy': 99.37},
    {'key': 'dense',   'name': 'Dense Neural Network', 'type': 'keras',   'f1': 99.66, 'accuracy': 99.70},
    {'key': 'cnn',     'name': '1D-CNN',               'type': 'keras',   'f1': 98.49, 'accuracy': 98.59},
    {'key': 'knn',     'name': 'KNN',                  'type': 'sklearn', 'f1': 98.60, 'accuracy': 98.53},
    {'key': 'lr',      'name': 'Logistic Regression',  'type': 'sklearn', 'f1': 97.81, 'accuracy': 97.76},
]

_model_map = {
    'xgboost': xgb_model, 'rf': rf_model, 'knn': knn_model,
    'lr': lr_model, 'dense': dense_model, 'cnn': cnn_model,
}
for m in MODELS:
    m['model'] = _model_map[m['key']]

# ── Store recent live results ─────────────────────────────────
live_results = []

# ── Overall model performance table ──────────────────────────
MODEL_RESULTS = [
    {'name': 'KNN',                 'accuracy': 98.53, 'precision': 98.68, 'recall': 98.53, 'f1': 98.60},
    {'name': 'Logistic Regression', 'accuracy': 97.76, 'precision': 98.05, 'recall': 97.76, 'f1': 97.81},
    {'name': 'Random Forest',       'accuracy': 99.37, 'precision': 99.78, 'recall': 99.37, 'f1': 99.55},
    {'name': 'XGBoost',             'accuracy': 99.87, 'precision': 99.88, 'recall': 99.87, 'f1': 99.87},
    {'name': 'Dense Neural Network','accuracy': 99.70, 'precision': 99.70, 'recall': 99.70, 'f1': 99.66},
    {'name': '1D-CNN',              'accuracy': 98.59, 'precision': 98.57, 'recall': 98.59, 'f1': 98.49},
]

ATTACK_CLASSES = le.classes_.tolist()


# ── Helper: run one model on a batch ─────────────────────────
def predict_with(meta, X_scaled):
    """Returns list of (predicted_label, confidence%) tuples."""
    m = meta['model']

    if meta['type'] == 'sklearn':
        preds = m.predict(X_scaled)
        probs = m.predict_proba(X_scaled)
        return [
            (le.classes_[pred], round(float(prob[pred]) * 100, 2))
            for pred, prob in zip(preds, probs)
        ]
    else:
        X_in = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1) \
               if meta['key'] == 'cnn' else X_scaled
        raw = m.predict(X_in, verbose=0)
        return [
            (le.classes_[int(np.argmax(row))], round(float(np.max(row)) * 100, 2))
            for row in raw
        ]


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def dashboard():
    top_features = shap_imp.head(10).to_dict('records')
    return render_template('dashboard.html',
                           model_results=MODEL_RESULTS,
                           top_features=top_features,
                           attack_classes=ATTACK_CLASSES)

@app.route('/compare')
def compare_page():
    return render_template('compare.html', attack_classes=ATTACK_CLASSES)

@app.route('/simulate')
def simulate_page():
    return render_template('simulate.html', attack_classes=ATTACK_CLASSES)


# ============================================================
# API ENDPOINTS
# ============================================================

# ── Model Comparison API ──────────────────────────────────────
@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        data        = request.get_json()
        attack_type = data.get('attack_type', 'DDoS')
        n_samples   = int(data.get('n_samples', 30))

        df = pd.read_parquet(os.path.join(DATA_PATH, 'clean_data.parquet'))

        # ── Sample rows ──────────────────────────────────────
        if attack_type == 'MIXED':
            classes   = df['Label'].unique()
            per_class = max(2, n_samples // len(classes))
            sampled   = pd.concat([
                df[df['Label'] == c].sample(
                    min(per_class, int((df['Label'] == c).sum())),
                    random_state=42
                )
                for c in classes
            ]).sample(frac=1, random_state=42)
        else:
            if attack_type == 'BENIGN':
                mask = df['Label'] == 0
            else:
                attack_idx = le.transform([attack_type])[0]
                mask       = df['Label'] == attack_idx
            sampled = df[mask].sample(
                min(n_samples, int(mask.sum())), random_state=42
            )

        true_labels = [le.classes_[int(lbl)] for lbl in sampled['Label'].values]
        # clean_data.parquet is already scaled — do NOT re-apply scaler
        X_scaled    = sampled.drop('Label', axis=1)[feature_cols].values

        # Debug: print to Flask console to verify labels are correct
        print(f'[compare] attack={attack_type} n={len(true_labels)}')
        print(f'[compare] true_labels sample: {true_labels[:5]}')
        print(f'[compare] X_scaled[0][:5]: {X_scaled[0][:5]}')

        # ── Run all 6 models ─────────────────────────────────
        model_results = []
        for meta in MODELS:
            preds = predict_with(meta, X_scaled)

            correct = 0
            attacks_found = 0
            per_class_stats = {}

            for (pred_label, conf), true_label in zip(preds, true_labels):
                if pred_label == true_label:
                    correct += 1
                if pred_label != 'BENIGN':
                    attacks_found += 1
                pc = per_class_stats.setdefault(true_label, {'correct': 0, 'total': 0})
                pc['total'] += 1
                if pred_label == true_label:
                    pc['correct'] += 1

            total    = len(true_labels)
            accuracy = round(correct / total * 100, 2)
            avg_conf = round(sum(c for _, c in preds) / len(preds), 2)

            class_breakdown = [
                {
                    'label':    lbl,
                    'correct':  s['correct'],
                    'total':    s['total'],
                    'accuracy': round(s['correct'] / s['total'] * 100, 1),
                }
                for lbl, s in sorted(per_class_stats.items())
            ]

            model_results.append({
                'key':            meta['key'],
                'name':           meta['name'],
                'correct':        correct,
                'total':          total,
                'accuracy':       accuracy,
                'avg_confidence': avg_conf,
                'attacks_found':  attacks_found,
                'baseline_f1':    meta['f1'],
                'class_breakdown': class_breakdown,
            })

        model_results.sort(key=lambda x: x['accuracy'], reverse=True)

        from collections import Counter
        class_dist = [
            {'label': k, 'count': v}
            for k, v in Counter(true_labels).most_common()
        ]

        return jsonify({
            'success':     True,
            'n_samples':   len(true_labels),
            'attack_type': attack_type,
            'models':      model_results,
            'class_dist':  class_dist,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e),
                        'trace': traceback.format_exc()})


# ── Simulation API ────────────────────────────────────────────
@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    try:
        data        = request.get_json()
        attack_type = data.get('attack_type', 'DDoS')
        n_samples   = int(data.get('n_samples', 20))

        df = pd.read_parquet(os.path.join(DATA_PATH, 'clean_data.parquet'))

        if attack_type == 'BENIGN':
            mask = df['Label'] == 0
        else:
            attack_idx = le.transform([attack_type])[0]
            mask       = df['Label'] == attack_idx

        attack_df = df[mask].sample(
            min(n_samples, int(mask.sum())), random_state=42
        )
        X_attack = attack_df.drop('Label', axis=1)[feature_cols]

        results = []
        for i, (_, row) in enumerate(X_attack.iterrows()):
            feature_array   = np.array([row.values])
            prediction      = xgb_model.predict(feature_array)[0]
            probabilities   = xgb_model.predict_proba(feature_array)[0]
            predicted_label = le.classes_[prediction]
            confidence      = float(probabilities[prediction]) * 100

            reconstruction = autoencoder.predict(feature_array, verbose=0)
            ae_error = float(np.mean(np.power(feature_array - reconstruction, 2)))

            results.append({
                'index':           i + 1,
                'true_label':      attack_type,
                'predicted_label': predicted_label,
                'confidence':      round(confidence, 2),
                'is_correct':      predicted_label == attack_type,
                'is_attack':       predicted_label != 'BENIGN',
                'anomaly_score':   round(ae_error, 4),
            })

        correct          = sum(1 for r in results if r['is_correct'])
        attacks_detected = sum(1 for r in results if r['is_attack'])

        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total':            len(results),
                'correct':          correct,
                'accuracy':         round(correct / len(results) * 100, 2),
                'attacks_detected': attacks_detected,
                'attack_type':      attack_type,
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── Model stats API ───────────────────────────────────────────
@app.route('/api/models')
def api_models():
    return jsonify(MODEL_RESULTS)


# ── Live detection API ────────────────────────────────────────
@app.route('/api/live', methods=['POST'])
def api_live():
    try:
        data       = request.get_json()
        features   = np.array([data['features']])
        flow_index = data.get('flow_index', 0)

        prediction      = xgb_model.predict(features)[0]
        probabilities   = xgb_model.predict_proba(features)[0]
        predicted_label = le.classes_[prediction]
        confidence      = float(probabilities[prediction]) * 100

        result = {
            'flow_index':    flow_index,
            'prediction':    predicted_label,
            'confidence':    round(confidence, 2),
            'is_attack':     predicted_label != 'BENIGN',
            'is_anomaly':    False,
            'anomaly_score': 0,
        }

        live_results.append(result)
        if len(live_results) > 100:
            live_results.pop(0)

        return jsonify({'success': True, **result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/live/results')
def api_live_results():
    return jsonify({'results': live_results})


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)