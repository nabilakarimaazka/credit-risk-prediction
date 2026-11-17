from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier

# Lokasi folder model, dihitung dari posisi file ini
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

model = CatBoostClassifier()
model.load_model(MODEL_DIR / "cb_model.cbm")

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
preprocess_config = joblib.load(MODEL_DIR / "preprocess_config.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
scoring_config = joblib.load(MODEL_DIR / "scoring_config.pkl")

TIER_ACTIONS = {
    "Low Risk": "Proses otomatis / jalur cepat",
    "Medium Risk": "Verifikasi standar: slip gaji, status pekerjaan, rekening",
    "High Risk": "Verifikasi ketat, tawarkan plafon lebih kecil",
    "Very High Risk": "Tolak, atau setujui hanya dengan jaminan"
}

def preprocess(df):
    """Mengubah data peminjam mentah menjadi format yang dipahami moddel"""
    df = df.copy()
    for col, median_value in preprocess_config["medians"].items():
        df[col] = df[col].fillna(median_value)
        
    for col, mapping in preprocess_config["binary_map"].items():
        df[col] = df[col].map(mapping)
    
    df = pd.get_dummies(df, columns=preprocess_config["one_hot_columns"], dtype=int)
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    scale_cols = preprocess_config["scale_columns"]
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df

def assign_tier(score):
    """Menentukan risk tier dari satu skow"""
    bins = scoring_config["tier_bins"]
    labels = scoring_config["tier_labels"]
    
    for upper, label in zip(bins[1:], labels):
        if score < upper:
            return label
    return labels[-1]
    
def predict_risk(df):
    """Dari data peminjam mentah menjadi skor, tier, dan keputusan"""
    X = preprocess(df)
    
    # 1. Score resiko dari mode (kolom 1 = peluang  gagal bayar)
    score = model.predict_proba(X)[:,1]
    
    # 2. Koreksi bobot kelas supaya bisa dibaca sebagai peluang
    W = scoring_config["class_weight_W"]
    prob = score / (score + W * (1 - score))
    
    # 3. Susun hasilnya dalam satu tabel
    result = pd.DataFrame(index=df.index)
    result ["skor_risiko"] = score.round(3)
    result["perkiraan_peluang"] = prob.round(3)
    result["risk_tier"] = [assign_tier(s) for s in score]
    result["keputusan"] = [
        "Tinjau" if s >= scoring_config["threshold"] else "Proses" for s in score
    ]
    result["tindakan"] = result["risk_tier"].map(TIER_ACTIONS)
    
    return result