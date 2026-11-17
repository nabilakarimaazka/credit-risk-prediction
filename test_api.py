import requests

URL ="http://127.0.0.1:8000"

def cek(nama, data, status_harapan):
    response = requests.post(f"{URL}/predict", json=data)
    lolos = response.status_code == status_harapan
    
    print(f"[{"LOLOS" if lolos else "GAGAGL"}] {nama}")
    print(f"    status: {response.status_code} (harapan {status_harapan})")
    print(f"    jawaban: {response.json()}")
    print()
    return lolos

peminjam_aman = {
    "person_age": 40,
    "person_income": 120000,
    "person_home_ownership": "OWN",
    "person_emp_length": 10,
    "loan_intent": "VENTURE",
    "loan_grade": "A",
    "loan_amnt": 8000,
    "loan_int_rate": 7.5,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 12,
}

peminjam_berisiko = {
    "person_age": 24,
    "person_income": 25000,
    "person_home_ownership": "RENT",
    "person_emp_length": 1,
    "loan_intent": "DEBTCONSOLIDATION",
    "loan_grade": "E",
    "loan_amnt": 15000,
    "loan_int_rate": 18.5,
    "cb_person_default_on_file": "Y",
    "cb_person_cred_hist_length": 2,
}

data_salah = dict(peminjam_aman, person_age=-5, loan_grade="H")

if __name__ == "__main__":
    print("Health:", requests.get(f"{URL}/health").json())
    print()
    
    hasil = [
        cek("Peminjam aman", peminjam_aman, 200),
        cek("Peminjam berisiko", peminjam_berisiko, 200),
        cek("Data salah ditolak", data_salah, 422)
    ]
    print(f"{sum(hasil)} dari {len(hasil)} tes lolos")