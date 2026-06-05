"""
Aplikasi Flask untuk Klasifikasi Serangga menggunakan Random Forest

Model yang digunakan:
- Random Forest (sklearn)
- PCA untuk reduksi dimensi
- StandardScaler untuk normalisasi fitur
- HOG + fitur tambahan untuk ekstraksi fitur
"""

import os
import uuid
import traceback

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from skimage.feature import hog, local_binary_pattern
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
#  Konfigurasi Aplikasi
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Maks 16 MB
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

# Nama kelas serangga beserta info tambahannya
CLASS_INFO = {
    "Butterfly": {
        "nama_id": "Kupu-kupu",
        "deskripsi": (
            "Serangga bersayap indah dari ordo Lepidoptera. "
            "Dikenal dengan metamorfosis sempurnanya dari ulat menjadi kupu-kupu."
        ),
        "ikon": "🦋",
        "warna": "#8B5CF6",
    },
    "Dragonfly": {
        "nama_id": "Capung",
        "deskripsi": (
            "Serangga predator yang gesit dengan dua pasang sayap transparan. "
            "Mampu terbang ke segala arah termasuk mundur dan melayang di tempat."
        ),
        "ikon": "🪲",
        "warna": "#06B6D4",
    },
    "Grasshopper": {
        "nama_id": "Belalang",
        "deskripsi": (
            "Serangga herbivora dengan kaki belakang yang kuat untuk melompat. "
            "Termasuk ordo Orthoptera, beberapa spesies dapat membentuk kawanan besar."
        ),
        "ikon": "🦗",
        "warna": "#22C55E",
    },
    "Ladybird": {
        "nama_id": "Kumbang Kepik",
        "deskripsi": (
            "Kumbang kecil berwarna cerah dari famili Coccinellidae. "
            "Sering dijadikan simbol keberuntungan dan merupakan predator alami kutu daun."
        ),
        "ikon": "🐞",
        "warna": "#EF4444",
    },
    "Mosquito": {
        "nama_id": "Nyamuk",
        "deskripsi": (
            "Serangga dari famili Culicidae. Nyamuk betina membutuhkan darah "
            "untuk perkembangan telur. Beberapa spesies adalah vektor penyakit berbahaya."
        ),
        "ikon": "🦟",
        "warna": "#F59E0B",
    },
}

# ─────────────────────────────────────────────
#  Konfigurasi Ekstraksi Fitur
#  SESUAIKAN dengan notebook pelatihan!
# ─────────────────────────────────────────────
IMG_SIZE = (64, 64)          # Ukuran resize gambar
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
LBP_P = 8                    # Jumlah titik LBP
LBP_R = 1                    # Radius LBP
HSV_BINS = 32                # Bin histogram HSV per channel
EXPECTED_FEATURES = 487      # ← Jumlah fitur yang harus dihasilkan


# ─────────────────────────────────────────────
#  Fungsi Ekstraksi Fitur
#  !! SALIN DARI NOTEBOOK PELATIHAN !!
# ─────────────────────────────────────────────
def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mengekstrak fitur dari gambar BGR.

    WAJIB SESUAIKAN dengan fungsi ekstraksi fitur di notebook pelatihan!
    Fungsi ini harus menghasilkan tepat 487 fitur.

    Parameter
    ----------
    img_bgr : np.ndarray
        Gambar dalam format BGR (hasil cv2.imread atau resize).

    Return
    ------
    np.ndarray, shape (487,)
    """
    # Resize ke ukuran standar
    img = cv2.resize(img_bgr, IMG_SIZE)

    # Konversi warna
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ── 1. HOG features (324 fitur) ──────────────────────
    hog_feats = hog(
        gray,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        orientations=HOG_ORIENTATIONS,
        feature_vector=True,
    )

    # ── 2. HSV Color Histogram (96 fitur) ────────────────
    h_hist = cv2.calcHist([img_hsv], [0], None, [HSV_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([img_hsv], [1], None, [HSV_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([img_hsv], [2], None, [HSV_BINS], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist, v_hist])  # 96 fitur

    # ── 3. LBP features (10 fitur) ───────────────────────
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")
    n_bins = LBP_P + 2
    lbp_hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # ── 4. Hu Moments (7 fitur) ──────────────────────────
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # ── 5. Color Stats — SESUAIKAN jika diperlukan ───────
    # Sisa fitur untuk mencapai total 487.
    # Ganti blok ini dengan fitur dari notebook jika berbeda.
    r_stats = _channel_stats(img_rgb[:, :, 0])  # R channel
    g_stats = _channel_stats(img_rgb[:, :, 1])  # G channel
    b_stats = _channel_stats(img_rgb[:, :, 2])  # B channel
    h_stats = _channel_stats(img_hsv[:, :, 0])  # H channel
    s_stats = _channel_stats(img_hsv[:, :, 1])  # S channel
    color_stats = np.concatenate([r_stats, g_stats, b_stats, h_stats, s_stats])

    # ── Gabungkan semua fitur ─────────────────────────────
    features = np.concatenate([
        hog_feats,    # 324
        color_hist,   # 96
        lbp_hist,     # 10
        hu_moments,   # 7
        color_stats,  # 50  (10 stats × 5 channel)
    ])                # Total: 487

    return features.astype(np.float32)


def _channel_stats(channel: np.ndarray) -> np.ndarray:
    """Menghitung 10 statistik dari satu channel warna."""
    flat = channel.flatten().astype(np.float64)
    return np.array([
        flat.mean(),
        flat.std(),
        flat.min(),
        float(np.percentile(flat, 25)),
        np.median(flat),
        float(np.percentile(flat, 75)),
        flat.max(),
        float(np.percentile(flat, 75) - np.percentile(flat, 25)),  # IQR
        float(np.sum((flat - flat.mean()) ** 3) / (len(flat) * flat.std() ** 3 + 1e-10)),  # skewness
        float(np.sum((flat - flat.mean()) ** 4) / (len(flat) * flat.std() ** 4 + 1e-10)),  # kurtosis
    ])


# ─────────────────────────────────────────────
#  Load Model saat Startup
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

print("=" * 55)
print("  Memuat model klasifikasi serangga...")
print("=" * 55)

try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, "model_klasifikasi_serangga.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Verifikasi feature count
    _test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    _test_feats = extract_features(_test_img)
    assert len(_test_feats) == EXPECTED_FEATURES, (
        f"❌ Jumlah fitur tidak cocok! "
        f"extract_features() menghasilkan {len(_test_feats)} fitur, "
        f"tapi model butuh {EXPECTED_FEATURES} fitur.\n"
        f"   → Sesuaikan fungsi extract_features() di app.py dengan notebook pelatihan!"
    )

    print(f"  ✅ Model RF         : {rf_model.n_estimators} trees, depth={rf_model.max_depth}")
    print(f"  ✅ PCA              : {pca.n_components_} komponen")
    print(f"  ✅ Scaler           : {scaler.n_features_in_} fitur input")
    print(f"  ✅ Label Encoder    : {list(label_encoder.classes_)}")
    print(f"  ✅ Verifikasi fitur : {len(_test_feats)} fitur ✓")
    print("=" * 55)
    MODEL_LOADED = True

except AssertionError as e:
    print(f"\n{'!'*55}")
    print(str(e))
    print(f"{'!'*55}\n")
    MODEL_LOADED = False
    rf_model = pca = scaler = label_encoder = None

except Exception as e:
    print(f"\n❌ Gagal memuat model: {e}")
    print("   Pastikan file .pkl ada di folder 'model/'")
    traceback.print_exc()
    MODEL_LOADED = False
    rf_model = pca = scaler = label_encoder = None


# ─────────────────────────────────────────────
#  Utility Functions
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_insect(img_bgr: np.ndarray) -> dict:
    """Jalankan pipeline prediksi penuh dan kembalikan hasil."""
    # 1. Ekstraksi fitur
    features = extract_features(img_bgr)

    # 2. Normalisasi
    features_scaled = scaler.transform(features.reshape(1, -1))

    # 3. Reduksi dimensi PCA
    features_pca = pca.transform(features_scaled)

    # 4. Prediksi
    pred_class_idx = rf_model.predict(features_pca)[0]
    pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
    probabilities = rf_model.predict_proba(features_pca)[0]

    # Susun probabilitas per kelas
    class_probs = {}
    for idx, cls in enumerate(label_encoder.classes_):
        class_probs[cls] = round(float(probabilities[idx]) * 100, 2)

    return {
        "predicted_class": pred_class,
        "confidence": class_probs[pred_class],
        "class_probabilities": class_probs,
        "class_info": CLASS_INFO.get(pred_class, {}),
    }


# ─────────────────────────────────────────────
#  Route: Halaman Utama
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED)


# ─────────────────────────────────────────────
#  Route: API Prediksi
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({
            "success": False,
            "error": "Model belum dimuat. Periksa file model di folder 'model/' dan "
                     "pastikan fungsi extract_features() sesuai dengan notebook pelatihan.",
        }), 503

    # Validasi file
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file yang diunggah."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Nama file kosong."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Format file tidak didukung. Gunakan: {', '.join(ALLOWED_EXTENSIONS).upper()}",
        }), 400

    try:
        # Simpan file sementara
        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        # Baca gambar
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({"success": False, "error": "Gagal membaca gambar. Pastikan file tidak rusak."}), 400

        # Prediksi
        result = predict_insect(img)
        result["image_url"] = f"/static/uploads/{unique_filename}"
        result["success"] = True

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan saat memproses gambar: {str(e)}",
        }), 500


# ─────────────────────────────────────────────
#  Route: Health Check
# ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok" if MODEL_LOADED else "model_error",
        "model_loaded": MODEL_LOADED,
    })


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
