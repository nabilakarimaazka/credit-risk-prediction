"""Demo online credit risk scoring (Streamlit Community Cloud).

Memakai pipeline yang sama dengan API: src/predict.py
"""
import pandas as pd
import streamlit as st

from src.predict import predict_risk, scoring_config

st.set_page_config(page_title="Credit Risk Scoring", page_icon="📊", layout="wide")

TIER_COLOR = {
    "Low Risk": "#16a34a",
    "Medium Risk": "#d97706",
    "High Risk": "#ea580c",
    "Very High Risk": "#dc2626",
}

st.title("Credit Risk Scoring")
st.caption(
    "Perkirakan risiko gagal bayar peminjam dan dapatkan rekomendasi tindakan. "
    "Model: CatBoost (ROC-AUC 0,943). Threshold keputusan: "
    f"{scoring_config['threshold']:.2f}."
)

kiri, kanan = st.columns([1.4, 1], gap="large")

with kiri:
    with st.form("formulir"):
        st.subheader("Profil peminjam")
        a, b = st.columns(2)
        person_age = a.number_input("Usia (tahun)", min_value=18, max_value=100, value=30)
        person_income = b.number_input("Pendapatan per tahun", min_value=1, value=50000, step=1000)
        person_home_ownership = a.selectbox("Status tempat tinggal", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp_kosong = b.checkbox("Lama bekerja tidak diketahui")
        person_emp_length = b.number_input("Lama bekerja (tahun)", min_value=0.0, max_value=60.0, value=3.0, step=0.5)

        st.subheader("Detail pinjaman")
        c, d = st.columns(2)
        loan_intent = c.selectbox(
            "Tujuan pinjaman",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        loan_grade = d.selectbox("Grade pinjaman", list("ABCDEFG"), index=2)
        loan_amnt = c.number_input("Jumlah pinjaman", min_value=1, value=10000, step=500)
        loan_int_rate = d.number_input("Suku bunga (%)", min_value=0.1, max_value=40.0, value=13.5, step=0.1)

        st.subheader("Riwayat kredit")
        e, f = st.columns(2)
        default_label = e.selectbox("Pernah gagal bayar sebelumnya?", ["Tidak", "Ya"])
        cred_hist = f.number_input("Panjang riwayat kredit (tahun)", min_value=0, max_value=60, value=4)

        kirim = st.form_submit_button("Hitung risiko", type="primary", use_container_width=True)

with kanan:
    if not kirim:
        st.info("Isi formulir lalu klik **Hitung risiko**. Hasil penilaian akan muncul di sini.")
    else:
        data = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": None if emp_kosong else person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": round(loan_amnt / person_income, 2),
            "cb_person_default_on_file": "Y" if default_label == "Ya" else "N",
            "cb_person_cred_hist_length": cred_hist,
        }])
        hasil = predict_risk(data).iloc[0]
        tier = hasil["risk_tier"]
        warna = TIER_COLOR[tier]

        st.markdown(
            f"<span style='background:{warna};color:white;padding:6px 14px;"
            f"border-radius:999px;font-weight:600'>{tier}</span>",
            unsafe_allow_html=True,
        )
        st.metric("Skor risiko (0 = aman, 1 = sangat berisiko)", f"{hasil['skor_risiko']:.2f}")
        st.progress(float(min(max(hasil["skor_risiko"], 0), 1)))

        m1, m2 = st.columns(2)
        m1.metric("Keputusan", hasil["keputusan"])
        m2.metric("Perkiraan peluang gagal bayar", f"{hasil['perkiraan_peluang'] * 100:.1f}%")

        st.markdown(f"**Tindakan yang disarankan:** {hasil['tindakan']}")
        st.caption(
            "Batas tier: Low < 0,24 ≤ Medium < 0,50 ≤ High < 0,80 ≤ Very High. "
            "Threshold dipilih untuk meminimalkan biaya kesalahan (asumsi FN $500, FP $50)."
        )

st.divider()
st.caption("Credit Risk Prediction · Nabila Karima Azka · Data: Kaggle Credit Risk Dataset (laotse)")
