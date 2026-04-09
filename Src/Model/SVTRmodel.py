import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Src.Preprocessing.enhancedpicture import EnhancedPicture


class SVTRModel:
    def __init__(self):
        self.ocr = PaddleOCR(
            use_textline_orientation=True,
            lang='en',
            device='cpu',
            enable_mkldnn=False,
            
        )

        self.preprocessor = EnhancedPicture()

        print("Model SVTR berhasil dibuat")

    def detect(self, image):
        """
        Mendeteksi lokasi teks dalam gambar.
        Input  : numpy array (BGR)
        Output : list of bounding box
                 [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ... ]
        """
        result = self.ocr.ocr(image, rec=False)  # rec=False → hanya detection

        if not result or result[0] is None:
            print("[DETECTION] Tidak ada teks terdeteksi.")
            return []

        boxes = [line[0] for line in result[0]]
        print(f"[DETECTION] {len(boxes)} area teks ditemukan.")
        return boxes

    def recognize(self, image):
        """
        Membaca teks dari gambar (sudah di-crop / sudah jelas).
        Input  : numpy array (BGR)
        Output : list of (teks, confidence)
        """
        result = self.ocr.ocr(image, det=False)  # det=False, hanya recognition

        if not result or result[0] is None:
            print("[RECOGNITION] Tidak ada teks terbaca.")
            return []

        recognition_results = []
        for line in result[0]:
            text = line[0]
            confidence = line[1]
            recognition_results.append((text, confidence))
            print(f"[RECOGNITION] '{text}' — conf: {confidence:.4f} ({confidence*100:.2f}%)")

        return recognition_results

    def predict(self, image):
        """
        Pipeline lengkap: detection → recognition.
        Input  : numpy array (BGR)
        Output : list of dict {
                    'text'      : str,
                    'confidence': float,
                    'box'       : [[x1,y1],...,[x4,y4]],
                    'status'    : 'HIGH' / 'MEDIUM' / 'LOW'
                 }
        """

        result = self.ocr.ocr(image)

        if not result or result[0] is None or len(result[0]) == 0:
            print("[PREDICT] Tidak ada teks yang terdeteksi.")
            return []

        output = []
        for line in result[0]:
            box        = line[0]  
            text       = line[1][0]  
            confidence = line[1][1]  
            if confidence >= 0.90:
                status = "HIGH"
            elif confidence >= 0.70:
                status = "MEDIUM"
            else:
                status = "LOW, perlu review manual"

            output.append({
                'text'      : text,
                'confidence': confidence,
                'box'       : box,
                'status'    : status
            })

        return output

    def predict_from_file(self, image_path):
        """
        Load gambar dari file, preprocess, lalu predict.
        Input  : path gambar (str)
        Output : sama seperti predict()
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
            return []

        print(f"\n{'='*50}")
        print(f"[INPUT] {image_path}")
        print(f"[SIZE]  {image.shape[1]}×{image.shape[0]} px")
        print(f"{'='*50}")
        print("\n[PREPROCESSING]")
        processed = self.preprocessor.process(image_path)

        print("\n[SVTR INFERENCE]")
        results = self.predict(processed)

        return results

    def visualize(self, image_path, results):
        """
        Tampilkan hasil deteksi dengan bounding box di atas gambar asli.
        """
        image = cv2.imread(image_path)
        if image is None or not results:
            return

        for item in results:
            box        = item['box']
            text       = item['text']
            confidence = item['confidence']
            status     = item['status']

            if 'HIGH' in status:
                color = (0, 200, 0)   
            elif 'MEDIUM' in status:
                color = (0, 165, 255)  #
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
    model = SVTRModel()

    img_path = "Data/Raw/kemasan.jpg"

    results = model.predict_from_file(img_path)

    print(f"\n{'='*50}")
    print(f"HASIL DETEKSI & RECOGNITION")
    print(f"{'='*50}")

    if not results:
        print("Tidak ada teks terdeteksi.")
    else:
        for i, item in enumerate(results, 1):
            print(f"\n[{i}] Teks       : {item['text']}")
            print(f"    Confidence : {item['confidence']:.4f} "
                  f"({item['confidence']*100:.2f}%)")
            print(f"    Status     : {item['status']}")

        print(f"\n{'='*50}")
        print("TEST DETECTION ONLY")
        raw_img = cv2.imread(img_path)
        boxes = model.detect(raw_img)
        print(f"Jumlah area teks: {len(boxes)}")

        print(f"\n{'='*50}")
        print("TEST RECOGNITION ONLY")
        model.recognize(raw_img)

        model.visualize(img_path, results)