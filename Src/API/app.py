import os
import re
import logging
import traceback

os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.add_dll_directory(
    r"D:\SVTR-Project\ocr_env\Lib\site-packages\paddle\base"
)

from flask import Flask, request, jsonify
import cv2
import numpy as np

from Src.Preprocessing.enhancedpicture import EnhancedPicture
from Src.Model.SVTRmodel import SVTRModel

# Error ditampilan saja
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = Flask(__name__)

RAW_FOLDER       = 'Data/Raw'
PROCESSED_FOLDER = 'Data/Processed'
os.makedirs(RAW_FOLDER,       exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Inisialisasi model, ojo lali diubah debug=false buat production
ai_engine    = SVTRModel(debug=False)
preprocessor = EnhancedPicture(debug=False)


# buat filter teks seng ga penting 
def is_valid_text(text, confidence):
    if len(text.strip()) <= 2:
        return False
    if confidence < 0.50:
        return False
    if re.match(r'^[^a-zA-Z0-9]+$', text.strip()):
        return False
    clean = re.sub(r'[^a-zA-Z0-9\s/:\-\.]', '', text)
    if len(clean) < len(text) * 0.5:
        return False
    return True


# pattern recognition biar enak baca expired date nya
def extract_key_info(results):
    date_pattern = re.compile(
        r'\b(\d{1,2}[\s\-/\.]?'
        r'(?:JAN|FEB|MAR|APR|MAY|JUN|'
        r'JUL|AUG|SEP|OCT|NOV|DEC|\d{1,2})'
        r'[\s\-/\.]?\d{2,4})\b',
        re.IGNORECASE
    )
    kode_pattern = re.compile(
        r'\b([A-Z]\d+[A-Z]?\s+\d{2}:\d{2}|'
        r'LOT\s*\d+|'
        r'[A-Z]\d{3,})\b',
        re.IGNORECASE
    )

    exp_date      = None
    kode_produksi = None

    for item in results:
        text = item['text']
        if not exp_date and date_pattern.search(text):
            exp_date = {
                "text"      : text,
                "confidence": round(float(item['confidence']), 4),
                "status"    : item['status']
            }
        if not kode_produksi and kode_pattern.search(text):
            kode_produksi = {
                "text"      : text,
                "confidence": round(float(item['confidence']), 4),
                "status"    : item['status']
            }

    return exp_date, kode_produksi

@app.route('/')
def home():
    return jsonify({
        "message": "SVTR OCR API — Server aktif",
        "version": "1.0.0",
        "endpoints": {
            "health" : "GET  /cek-koneksi",
            "predict": "POST /predict"
        }
    })


@app.route('/cek-koneksi', methods=['GET'])
def cek_koneksi():
    return jsonify({
        "status" : "OK",
        "message": "API SVTR aktif dan siap menerima gambar"
    })


@app.route('/predict', methods=['POST'])
def predict():
    # Validasi file
    if 'image' not in request.files:
        return jsonify({
            "status" : "error",
            "message": "File gambar tidak ditemukan"
        }), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({
            "status" : "error",
            "message": "Nama file kosong"
        }), 400

    # Validasi ekstensi
    allowed = {'jpg', 'jpeg', 'png'}
    ext     = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({
            "status" : "error",
            "message": f"Format file tidak didukung: .{ext}. "
                       f"Gunakan jpg/jpeg/png"
        }), 400

    # Simpan file
    raw_path = os.path.join(RAW_FOLDER, file.filename)
    file.save(raw_path)

    try:
        # Prediksi dengan preprocessing
        results = ai_engine.predict_from_file(raw_path)

        # Fallback tanpa preprocessing
        if not results:
            raw_img = cv2.imread(raw_path)
            results = ai_engine.predict(raw_img)

        # Tidak ada teks terdeteksi
        if not results:
            return jsonify({
                "status" : "Re-Capture",
                "message": "Teks tidak terdeteksi, "
                           "silakan ambil foto ulang",
                "data"   : None
            }), 200

        # Filter teks sampah
        filtered = [
            r for r in results
            if is_valid_text(r['text'], r['confidence'])
        ]

        if not filtered:
            return jsonify({
                "status" : "Re-Capture",
                "message": "Teks tidak cukup jelas, "
                           "silakan ambil foto ulang",
                "data"   : None
            }), 200

        # Ekstrak informasi kunci
        exp_date, kode_produksi = extract_key_info(filtered)

        # Format semua deteksi valid
        all_detections = [{
            "text"      : r['text'],
            "confidence": round(float(r['confidence']), 4),
            "status"    : r['status']
        } for r in filtered]

        # Status keseluruhan
        best         = max(filtered, key=lambda x: x['confidence'])
        final_status = "Ok" if best['confidence'] >= 0.85 \
                       else "Re-Capture"

        return jsonify({
            "status" : final_status,
            "message": "Proses selesai",
            "data"   : {
                "exp_date"        : exp_date,
                "kode_produksi"   : kode_produksi,
                "best_text"       : best['text'],
                "confidence_score": round(
                    float(best['confidence']), 4
                ),
                "total_detected"  : len(filtered),
                "all_detections"  : all_detections,
                "processed_file"  : "processed_" + file.filename
            }
        })

    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return jsonify({
            "status" : "error",
            "message": "Terjadi kesalahan pada server"
        }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False   # False untuk production
    )