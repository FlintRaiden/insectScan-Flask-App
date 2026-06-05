# 🔬 InsectScan — Klasifikasi Serangga dengan Random Forest

Aplikasi web Flask untuk mengklasifikasikan gambar serangga ke dalam 5 kelas:
**Butterfly, Dragonfly, Grasshopper, Ladybird, Mosquito**

---

## 📁 Struktur Folder

```
flask-app/
├── app.py                          # ← Aplikasi Flask utama
├── requirements.txt
├── model/
│   ├── model_klasifikasi_serangga.pkl
│   ├── pca.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── uploads/                   # Gambar yang diunggah user (auto)
└── templates/
    └── index.html
```

---

## ⚙️ Setup & Menjalankan Aplikasi

### 1. Aktifkan virtual environment

```bash
# Masuk ke folder project
cd path/ke/flask-app

# Aktifkan venv yang sama dengan yang digunakan untuk training
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

### 3. Pastikan file model ada di folder `model/`

Salin file-file ini ke dalam folder `model/`:
- `model_klasifikasi_serangga.pkl`
- `pca.pkl`
- `scaler.pkl`
- `label_encoder.pkl`

### 4. Jalankan Flask

```bash
python app.py
```

### 5. Buka di browser

```
http://localhost:5000
```

---

## ⚠️ PENTING: Verifikasi Preprocessing

Fungsi `extract_features()` di `app.py` **HARUS IDENTIK** dengan yang digunakan
saat pelatihan model. Aplikasi akan otomatis memverifikasi ini saat startup:

```
✅ Verifikasi fitur : 487 fitur ✓
```

Jika muncul error seperti ini:
```
❌ Jumlah fitur tidak cocok! extract_features() menghasilkan X fitur,
   tapi model butuh 487 fitur.
```

Salin fungsi ekstraksi fitur dari notebook pelatihan Anda ke dalam `app.py`.

### Pipeline Fitur yang Digunakan (487 fitur total)

| Fitur            | Jumlah | Keterangan                              |
|-----------------|--------|-----------------------------------------|
| HOG             | 324    | ppc=(16,16), cpb=(2,2), orient=9        |
| HSV Histogram   | 96     | 32 bins × 3 channel (H, S, V)           |
| LBP             | 10     | P=8, R=1, method='uniform'              |
| Hu Moments      | 7      | Log-transformed                         |
| Color Stats     | 50     | 10 stats × 5 channel (R, G, B, H, S)   |
| **TOTAL**       | **487**|                                         |

---

## 📊 Performa Model

| Metrik           | Nilai   |
|-----------------|---------|
| Akurasi Test    | 55.96%  |
| Jumlah Data     | 4,449 gambar |
| Decision Trees  | 500     |
| Max Depth       | 11      |
| Min Samples Leaf| 8       |
| PCA Components  | 150     |

### AUC per Kelas
- 🦟 Mosquito:    0.903 (Terbaik)
- 🦋 Butterfly:   0.840
- 🪲 Dragonfly:   0.833
- 🐞 Ladybird:    0.812
- 🦗 Grasshopper: 0.794

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError` | Jalankan `pip install -r requirements.txt` |
| `FileNotFoundError: .pkl` | Pastikan file pkl ada di folder `model/` |
| Fitur tidak cocok | Sesuaikan `extract_features()` dengan notebook |
| Port sudah dipakai | Ganti port di `app.py`: `app.run(port=5001)` |

---
