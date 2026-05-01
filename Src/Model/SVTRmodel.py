import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
import cv2
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Src.Preprocessing.enhancedpicture import EnhancedPicture


class SVTRModel:

    def __init__(self, debug=False):
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

    # ─() Versi model yang digunakan )
    # PaddleOCR  : v3.0
    # Detection  : PP-OCRv5_server_det
    # Recognition: en_PP-OCRv5_mobile_rec
    # Algorithm  : SVTR_HGNet (PP-OCRv5)
    # Input size : 48×320 px (otomatis internal)
    # Reference: github.com/PaddlePaddle/PaddleOCR

        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            device='cpu',
            enable_mkldnn=False,
            det_limit_side_len=640,
            det_limit_type='min',
        )
        self.preprocessor = EnhancedPicture(debug=debug)
        logger.debug("Model SVTR berhasil dibuat")

    def detect(self, image):
        result = self.ocr.ocr(image, rec=False)
        if not result or result[0] is None:
            logger.debug("[DETECTION] Tidak ada teks terdeteksi.")
            return []
        first = result[0]
        if isinstance(first, dict):
            boxes = first.get('dt_polys', [])
        elif isinstance(first, list):
            boxes = [line[0] for line in first]
        else:
            boxes = []
        logger.debug(f"[DETECTION] {len(boxes)} area ditemukan.")
        return boxes

    def recognize(self, image):
        result = self.ocr.ocr(image, det=False)
        if not result or result[0] is None:
            logger.debug("[RECOGNITION] Tidak ada teks terbaca.")
            return []
        first = result[0]
        recognition_results = []
        if isinstance(first, dict):
            rec_texts  = first.get('rec_texts',  [])
            rec_scores = first.get('rec_scores', [])
            for i, text in enumerate(rec_texts):
                conf = rec_scores[i] if i < len(rec_scores) else 0.0
                recognition_results.append((text, conf))
                logger.debug(
                    f"[RECOGNITION] '{text}' — {conf*100:.2f}%"
                )
        elif isinstance(first, list):
            for line in first:
                text = line[0]
                conf = line[1]
                recognition_results.append((text, conf))
                logger.debug(
                    f"[RECOGNITION] '{text}' — {conf*100:.2f}%"
                )
        return recognition_results

    def predict(self, image):
        result = self.ocr.ocr(image)
        if not result or result[0] is None:
            logger.debug("[PREDICT] Tidak ada teks terdeteksi.")
            return []
        first = result[0]

        # Format v3.x — dict
        if isinstance(first, dict):
            rec_texts  = first.get('rec_texts',  [])
            rec_scores = first.get('rec_scores', [])
            rec_polys  = first.get('rec_polys',  [])
            if not rec_texts:
                logger.debug("[PREDICT] rec_texts kosong.")
                return []
            output = []
            for i, text in enumerate(rec_texts):
                confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                box        = rec_polys[i]  if i < len(rec_polys)  else []
                if confidence >= 0.90:
                    status = "HIGH"
                elif confidence >= 0.70:
                    status = "MEDIUM"
                else:
                    status = "LOW"
                output.append({
                    'text'      : text,
                    'confidence': confidence,
                    'box'       : box,
                    'status'    : status
                })
                logger.debug(
                    f"[PREDICT] '{text}' — "
                    f"{confidence*100:.1f}% — {status}"
                )
            return output

        # Format v2.x — list
        elif isinstance(first, list):
            output = []
            for line in first:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                box        = line[0]
                text       = line[1][0]
                confidence = line[1][1]
                if confidence >= 0.90:
                    status = "HIGH"
                elif confidence >= 0.70:
                    status = "MEDIUM"
                else:
                    status = "LOW"
                output.append({
                    'text'      : text,
                    'confidence': confidence,
                    'box'       : box,
                    'status'    : status
                })
            return output

        logger.debug(f"[PREDICT] Format tidak dikenal: {type(first)}")
        return []

    def predict_from_file(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Gambar tidak ditemukan: {image_path}")
            return []

        logger.debug(f"Input: {image_path} — {image.shape}")

        processed = self.preprocessor.process(image_path)
        results = self.predict(processed)
        if not results:
            logger.debug("Fallback tanpa preprocessing...")
            results = self.predict(image)

        return results

    def visualize(self, image_path, results):
        image = cv2.imread(image_path)
        if image is None or not results:
            return
        for item in results:
            box        = item['box']
            text       = item['text']
            confidence = item['confidence']
            status     = item['status']
            if status == "HIGH":
                color = (0, 200, 0)
            elif status == "MEDIUM":
                color = (0, 165, 255)
            else:
                color = (0, 0, 220)
            pts = np.array(box, dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True,
                          color=color, thickness=2)
            label = f"{text} ({confidence*100:.1f}%)"
            x, y  = int(box[0][0]), int(box[0][1]) - 8
            cv2.putText(image, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
        cv2.imshow("SVTR Detection Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    model    = SVTRModel(debug=True)
    img_path = r"D:\SVTR-Project\Data\Raw\Ultramilk.jpeg"
    results  = model.predict_from_file(img_path)
    if not results:
        print("Tidak ada teks terdeteksi.")
    else:
        for i, item in enumerate(results, 1):
            print(f"[{i}] {item['text']} "
                  f"— {item['confidence']*100:.2f}% "
                  f"— {item['status']}")
        model.visualize(img_path, results)