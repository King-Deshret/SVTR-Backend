import os
import re
import base64
import logging
import traceback

os.environ['FLAGS_use_onednn']                      = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

from Src.Preprocessing.enhancedpicture import EnhancedPicture
from Src.Model.SVTRmodel import SVTRModel

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")

RAW_FOLDER       = 'Data/Raw'
PROCESSED_FOLDER = 'Data/Processed'
os.makedirs(RAW_FOLDER,       exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REC_MODEL_DIR = os.path.join(BASE_DIR, 'Src', 'Model', 'Inference')

ai_engine    = SVTRModel(rec_model_dir=REC_MODEL_DIR, debug=False)
preprocessor = EnhancedPicture(debug=False)
print(f"  Model aktif: {ai_engine.model_source}")


def is_valid_text(text: str, confidence: float) -> bool:
    stripped = text.strip()
    if len(stripped) <= 2:           return False
    if confidence < 0.50:            return False
    if re.match(r'^[^a-zA-Z0-9]+$', stripped): return False
    clean = re.sub(r'[^a-zA-Z0-9\s/:\-\.]', '', stripped)
    if len(clean) < len(stripped) * 0.5: return False
    return True


_DATE_PATTERN = re.compile(
    r'\b(\d{1,2}[\s\-/\.]?'
    r'(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|\d{1,2})'
    r'[\s\-/\.]?\d{2,4})\b',
    re.IGNORECASE
)
_KODE_PATTERN = re.compile(
    r'\b([A-Z]\d+[A-Z]?\s+\d{2}:\d{2}|LOT\s*[A-Z0-9]+|[A-Z]{1,3}\d{3,}[A-Z]?)\b',
    re.IGNORECASE
)


def extract_key_info(results: list):
    exp_date = kode_produksi = None
    for item in results:
        text = item['text']
        if exp_date is None:
            m = _DATE_PATTERN.search(text)
            if m:
                exp_date = {
                    "raw_text"  : text,
                    "value"     : m.group(0),
                    "confidence": round(float(item['confidence']), 4),
                    "status"    : item['status'],
                }
        if kode_produksi is None:
            m = _KODE_PATTERN.search(text)
            if m:
                kode_produksi = {
                    "raw_text"  : text,
                    "value"     : m.group(0),
                    "confidence": round(float(item['confidence']), 4),
                    "status"    : item['status'],
                }
        if exp_date and kode_produksi:
            break
    return exp_date, kode_produksi


def build_response(filtered: list, scan_status: str):
    """
    Response JSON flat — langsung di-parse Flutter.

    {
      "scan_status"     : "OK" | "RE_CAPTURE",
      "exp_date"        : { value, confidence, status, raw_text } | null,
      "kode_produksi"   : { value, confidence, status, raw_text } | null,
      "confidence_score": float 0–1,
      "confidence_pct"  : int   0–100,
      "total_detected"  : int,
      "all_text"        : "baris1\nbaris2\n...",
      "detections"      : [ { text, confidence, status }, ... ]
    }
    """
    exp_date, kode_produksi = extract_key_info(filtered)
    best = max(filtered, key=lambda x: x['confidence'])

    return {
        "scan_status"     : scan_status,
        "exp_date"        : exp_date,
        "kode_produksi"   : kode_produksi,
        "confidence_score": round(float(best['confidence']), 4),
        "confidence_pct"  : round(float(best['confidence']) * 100),
        "total_detected"  : len(filtered),
        "all_text"        : '\n'.join(r['text'] for r in filtered),
        "detections"      : [
            {"text": r['text'], "confidence": round(float(r['confidence']), 4), "status": r['status']}
            for r in filtered
        ],
    }


def _empty_response(error_msg: str):
    return {
        "scan_status"     : "RE_CAPTURE",
        "exp_date"        : None,
        "kode_produksi"   : None,
        "confidence_score": 0.0,
        "confidence_pct"  : 0,
        "total_detected"  : 0,
        "all_text"        : "",
        "detections"      : [],
        "error"           : error_msg,
    }


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message"     : "SVTR OCR API — Server aktif",
        "version"     : "2.1.0",
        "model_source": ai_engine.model_source,
        "endpoints"   : {"health": "GET /health", "scan": "POST /api/scan"},
    })


@app.route('/health', methods=['GET'])
@app.route('/cek-koneksi', methods=['GET'])
def health():
    return jsonify({
        "status"      : "OK",
        "model_source": ai_engine.model_source,
        "message"     : "SVTR OCR API aktif dan siap menerima gambar",
    }), 200


@app.route('/api/scan', methods=['POST'])
def api_scan():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body harus JSON"}), 400
    if not data.get('image'):
        return jsonify({"error": "Field 'image' tidak ada atau kosong"}), 400

    try:
        img_bytes = base64.b64decode(data['image'])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        image     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Gambar tidak valid"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal decode gambar: {str(e)}"}), 400

    tmp_path = os.path.join(RAW_FOLDER, 'tmp_scan.jpg')
    try:
        cv2.imwrite(tmp_path, image)
    except Exception as e:
        return jsonify({"error": f"Gagal menyimpan sementara: {str(e)}"}), 500

    try:
        results = ai_engine.predict_from_file(tmp_path)
        if not results:
            results = ai_engine.predict(image)
        if not results:
            return jsonify(_empty_response(
                "Teks tidak terdeteksi. Coba foto lebih dekat."
            )), 200

        filtered = [r for r in results if is_valid_text(r['text'], r['confidence'])]
        if not filtered:
            return jsonify(_empty_response("Teks tidak jelas. Silakan foto ulang.")), 200

        best        = max(filtered, key=lambda x: x['confidence'])
        scan_status = "OK" if best['confidence'] >= 0.85 else "RE_CAPTURE"
        return jsonify(build_response(filtered, scan_status)), 200

    except Exception:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Kesalahan internal server. Cek log terminal."}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "File tidak ditemukan"}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({"status": "error", "message": "Nama file kosong"}), 400
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in {'jpg', 'jpeg', 'png'}:
        return jsonify({"status": "error", "message": f"Format tidak didukung: .{ext}"}), 400

    raw_path = os.path.join(RAW_FOLDER, file.filename)
    file.save(raw_path)

    try:
        results  = ai_engine.predict_from_file(raw_path) or ai_engine.predict(cv2.imread(raw_path))
        if not results:
            return jsonify({"status": "RE_CAPTURE", "message": "Teks tidak terdeteksi", "data": None}), 200
        filtered = [r for r in results if is_valid_text(r['text'], r['confidence'])]
        if not filtered:
            return jsonify({"status": "RE_CAPTURE", "message": "Teks tidak jelas", "data": None}), 200
        best        = max(filtered, key=lambda x: x['confidence'])
        scan_status = "OK" if best['confidence'] >= 0.85 else "RE_CAPTURE"
        return jsonify({"status": scan_status, "message": "Proses selesai",
                        "data": build_response(filtered, scan_status)}), 200
    except Exception:
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": "Kesalahan server"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 55)
    print("  SVTR OCR Backend v2.1")
    print(f"  Model: {ai_engine.model_source}")
    print("=" * 55)
    app.run(host='0.0.0.0', port=port, debug=False)