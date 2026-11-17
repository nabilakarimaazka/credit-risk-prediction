from typing import Literal, Optional

from pathlib import Path
from fastapi.responses import FileResponse
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict_lite import predict_risk, scoring_config

app = FastAPI(title="Credit Risk Scoring API")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

class LoanApplication(BaseModel):
    person_age : int =  Field(..., ge=17, le=100, description="Usia peminjam (tahun)")
    person_income : float = Field(..., gt=0, description="Pendapatan per tahun")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: Optional[float] = Field(None, ge=0, le=60, description="Lama bekerja (tahun), boleh kosong")
    loan_intent : Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade : Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt : float = Field(...,gt=0, description="Jumlah Pinjaman")
    loan_int_rate: Optional[float] = Field(None, gt=0, le=40, description="Suku bunga (%), boleh kosong")
    cb_person_default_on_file : Literal["Y", "N"]
    cb_person_cred_hist_length : int = Field(..., ge=0, le=60, description="Panjang riwayat kredit (tahun)")


@app.get("/", include_in_schema=False)
def halaman_utama():
    return FileResponse(WEB_DIR / "index.html")

@app.get("/health")
def health():
    return {
        "status": "oke",
        "threshold": scoring_config["threshold"],
        "tier_labels": scoring_config["tier_labels"],
    }

@app.post("/predict")
def predict(application : LoanApplication):
    data = application.model_dump()
    data["loan_percent_income"] = round(data["loan_amnt"] / data["person_income"], 2)
    
    df = pd.DataFrame([data])
    result = predict_risk(df)
    
    return result.iloc[0].to_dict()