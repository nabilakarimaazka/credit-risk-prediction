"""Versi ringan dari predict.py untuk deploy (tanpa catboost dan scikit-learn).

- Model CatBoost dibaca dari src/cb_model.py (hasil export format="python").
- StandardScaler dihitung manual dari models/scaler_params.json: (x - mean) / scale.
Hasilnya sama dengan predict.py, dibuktikan di notebooks/07_export_for_deploy.ipynb.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cb_model import apply_catboost_model

# Lokasi folder model, dihitung dari posisi file ini
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

preprocess_config = joblib.load(MODEL_DIR / "preprocess_config.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
scoring_config = joblib.load(MODEL_DIR / "scoring_config.pkl")

# Pengganti scaler.pkl: rata-rata dan standar deviasi tiap kolom
with open(MODEL_DIR / "scaler_params.json") as f:
    scaler_params = json.load(f)

SCALE_COLS = scaler_params["columns"]
SCALE_MEAN = np.array(scaler_params["mean"])
SCALE_STD = np.array(scaler_params["scale"])

TIER_ACTIONS = {
    "Low Risk": "Proses otomatis / jalur cepat",
    "Medium Risk": "Verifikasi standar: slip gaji, status pekerjaan, rekening",
    "High Risk": "Verifikasi ketat, tawarkan plafon lebih kecil",
    "Very High Risk": "Tolak, atau setujui hanya dengan jaminan"
}

def preprocess(df):
    """Mengubah data peminjam mentah menjadi format yang dipahami model"""
    df = df.copy()
    for col, median_value in preprocess_config["medians"].items():
        df[col] = df[col].fillna(median_value)

    for col, mapping in preprocess_config["binary_map"].items():
        df[col] = df[col].map(mapping)

    df = pd.get_dummies(df, columns=preprocess_config["one_hot_columns"], dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Rumus StandardScaler ditulis manual
    df[SCALE_COLS] = (df[SCALE_COLS] - SCALE_MEAN) / SCALE_STD
    return df

def assign_tier(score):
    """Menentukan risk tier dari satu skor"""
    bins = scoring_config["tier_bins"]
    labels = scoring_config["tier_labels"]

    for upper, label in zip(bins[1:], labels):
        if score < upper:
            return label
    return labels[-1]

def predict_risk(df):
    """Dari data peminjam mentah menjadi skor, tier, dan keputusan"""
    X = preprocess(df)

    # 1. Skor mentah (log-odds) per baris, lalu sigmoid jadi peluang 0-1.
    #    CatBoost membaca fitur sebagai float32, jadi kita samakan presisinya
    #    supaya angka yang pas di batas pohon tidak jatuh ke cabang yang salah.
    rows = X.values.astype(np.float32).astype(np.float64).tolist()
    raw = np.array([apply_catboost_model(row) for row in rows])
    score = 1 / (1 + np.exp(-raw))

    # 2. Koreksi bobot kelas supaya bisa dibaca sebagai peluang
    W = scoring_config["class_weight_W"]
    prob = score / (score + W * (1 - score))

    # 3. Susun hasilnya dalam satu tabel
    result = pd.DataFrame(index=df.index)
    result["skor_risiko"] = score.round(3)
    result["perkiraan_peluang"] = prob.round(3)
    result["risk_tier"] = [assign_tier(s) for s in score]
    result["keputusan"] = [
        "Tinjau" if s >= scoring_config["threshold"] else "Proses" for s in score
    ]
    result["tindakan"] = result["risk_tier"].map(TIER_ACTIONS)

    return result
