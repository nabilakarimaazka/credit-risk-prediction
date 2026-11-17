# Resep untuk menjalankan API credit risk di server (Hugging Face Spaces)
FROM python:3.12-slim

WORKDIR /app

# 1. Pasang library dulu (supaya build berikutnya lebih cepat)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Salin kode, halaman web, dan model
COPY src/ src/
COPY api/ api/
COPY web/ web/
COPY models/ models/

# 3. Jalankan API di port 7860 (port bawaan Hugging Face Spaces)
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
