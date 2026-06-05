# ══════════════════════════════════════════════════════
#  Dockerfile — InsectScan (Random Forest Classifier)
#  Platform : Hugging Face Spaces (Docker SDK)
#  Python   : 3.11-slim
# ══════════════════════════════════════════════════════

FROM python:3.11-slim

# Metadata
LABEL maintainer="InsectScan"
LABEL description="Insect classification using Random Forest + HOG + PCA"

# ── System dependencies (wajib untuk OpenCV & scikit-image) ───────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (layer cache) ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy seluruh project ──────────────────────────────────────────────────────
COPY . .

# ── Buat folder uploads agar app tidak error ─────────────────────────────────
RUN mkdir -p static/uploads

# ── Hugging Face Spaces menggunakan PORT 7860 ─────────────────────────────────
EXPOSE 7860

# ── Jalankan dengan Gunicorn (production WSGI server) ─────────────────────────
# -w 1 : 1 worker (HF free tier hanya 2 vCPU, RF prediction berat)
# --timeout 120 : beri waktu ekstra untuk prediksi pertama (model loading)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "120", \
     "--preload", \
     "app:app"]
