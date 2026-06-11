import os
os.environ['FLAGS_use_onednn']                      = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import sys
import logging
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Src.Preprocessing.enhancedpicture import EnhancedPicture


class SVTRModel:
    """
    Wrapper PaddleOCR dengan dukungan model custom hasil fine-tuning.

    Setelah model di-export, pakai:
        model = SVTRModel(rec_model_dir='path/ke/inference/')
        # Folder inference/ harus berisi:
        #   inference.pdmodel
        #   inference.pdiparams

    Kalau rec_model_dir None atau folder tidak ada → fallback ke model default.
    """

    def __init__(self, rec_model_dir: str = None, debug: bool = False):
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

        self._build_ocr(rec_model_dir)
        self.preprocessor = EnhancedPicture(debug=debug)

    def _build_ocr(self, rec_model_dir: str):
        # API PaddleOCR 2.7.x (versi yang sama dengan training):
        #   use_angle_cls (bukan use_textline_orientation), use_gpu (bukan device)
        base_kwargs = dict(
            use_angle_cls      = True,
            lang               = 'en',
            use_gpu            = False,
            enable_mkldnn      = False,
            det_limit_side_len = 640,
            det_limit_type     = 'min',
        )

        if rec_model_dir and os.path.isdir(rec_model_dir):
            pdmodel   = os.path.join(rec_model_dir, 'inference.pdmodel')
            pdiparams = os.path.join(rec_model_dir, 'inference.pdiparams')

            if os.path.exists(pdmodel) and os.path.exists(pdiparams):
                logger.debug(f"[MODEL] Custom: {rec_model_dir}")
                # Dictionary custom WAJIB di-pass di PaddleOCR 2.7 agar
                # pemetaan index→karakter cocok dengan saat training.
                rec_dict = os.path.join(rec_model_dir, 'custom_dict.txt')
                rec_kwargs = dict(base_kwargs)
                if os.path.exists(rec_dict):
                    rec_kwargs['rec_char_dict_path'] = rec_dict
                rec_kwargs['rec_image_shape'] = '3, 48, 320'
                self.ocr          = PaddleOCR(rec_model_dir=rec_model_dir, **rec_kwargs)
                self.model_source = f"custom: {rec_model_dir}"
                return
            else:
                logger.warning(
                    f"[MODEL] inference.pdmodel/pdiparams tidak ada di '{rec_model_dir}'. "
                    f"Fallback ke default."
                )
        elif rec_model_dir:
            logger.warning(f"[MODEL] Path '{rec_model_dir}' tidak ada. Fallback ke default.")

        logger.debug("[MODEL] Menggunakan model default PaddleOCR")
        self.ocr          = PaddleOCR(**base_kwargs)
        self.model_source = "default: PaddleOCR"

    def _parse_result(self, raw_result):
        """Normalisasi output PaddleOCR v2.x dan v3.x ke format standar."""
        if not raw_result or raw_result[0] is None:
            return []

        first  = raw_result[0]
        output = []

        if isinstance(first, dict):                          # v3.x
            rec_texts  = first.get('rec_texts',  [])
            rec_scores = first.get('rec_scores', [])
            rec_polys  = first.get('rec_polys',  [])
            for i, text in enumerate(rec_texts):
                conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                box  = rec_polys[i]          if i < len(rec_polys)  else []
                output.append(self._make_item(text, conf, box))

        elif isinstance(first, list):                        # v2.x
            for line in first:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                output.append(self._make_item(line[1][0], float(line[1][1]), line[0]))

        return output

    @staticmethod
    def _make_item(text, confidence, box):
        if confidence >= 0.90:   status = "HIGH"
        elif confidence >= 0.70: status = "MEDIUM"
        else:                    status = "LOW"
        return {'text': text, 'confidence': confidence, 'box': box, 'status': status}


    def predict(self, image: np.ndarray):
        results = self._parse_result(self.ocr.ocr(image, cls=True))
        logger.debug(f"[PREDICT] {len(results)} deteksi")
        return results

    def predict_from_file(self, image_path: str):
        """
        OCR dari path file.
        1. Preprocessing (tanpa resize) → predict
        2. Fallback tanpa preprocessing jika hasilnya kosong
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Gambar tidak ditemukan: {image_path}")
            return []

        try:
            processed = self.preprocessor.process_without_resize(image_path)
            results   = self.predict(processed)
        except Exception as e:
            logger.warning(f"[FILE] Preprocessing gagal: {e}")
            results = []

        if not results:
            logger.debug("[FILE] Fallback tanpa preprocessing...")
            results = self.predict(image)

        return results

    def detect(self, image: np.ndarray):
        raw = self.ocr.ocr(image, rec=False)
        if not raw or raw[0] is None:
            return []
        first = raw[0]
        if isinstance(first, dict):  return first.get('dt_polys', [])
        elif isinstance(first, list): return [line[0] for line in first]
        return []

    def recognize(self, image: np.ndarray):
        """Hanya recognition (tanpa detection)."""
        raw = self.ocr.ocr(image, det=False)
        if not raw or raw[0] is None:
            return []
        first = raw[0]
        if isinstance(first, dict):
            texts  = first.get('rec_texts', [])
            scores = first.get('rec_scores', [])
            return [(t, float(scores[i]) if i < len(scores) else 0.0) for i, t in enumerate(texts)]
        elif isinstance(first, list):
            return [(line[0], float(line[1])) for line in first]
        return []

    def visualize(self, image_path: str, results: list):
        image = cv2.imread(image_path)
        if image is None or not results:
            return
        color_map = {"HIGH": (0,200,0), "MEDIUM": (0,165,255), "LOW": (0,0,220)}
        for item in results:
            color = color_map.get(item['status'], (200,200,200))
            pts   = np.array(item['box'], dtype=np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            label = f"{item['text']} ({item['confidence']*100:.1f}%)"
            x, y  = int(item['box'][0][0]), int(item['box'][0][1]) - 8
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        h, w    = image.shape[:2]
        scale   = min(900/w, 600/h)
        resized = cv2.resize(image, (int(w*scale), int(h*scale)))
        cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection Result", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()