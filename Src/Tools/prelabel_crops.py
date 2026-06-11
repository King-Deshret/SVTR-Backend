# -*- coding: utf-8 -*-
"""
Auto pre-label crop recognition Roboflow pakai model recognition saat ini.
Hasilnya untuk dikoreksi manual (semi-automatic annotation) -> jauh lebih cepat
daripada ketik dari nol.

Jalankan dengan venv paddleocr 2.7.3:
    D:\svtr_export_work\exp_env\Scripts\python.exe Src/Tools/prelabel_crops.py

Output: Data/Annotation/rec_roboflow_prelabeled.txt
   format: <crop_path>\t<prediksi>\t<confidence>
   -> buka di Excel/editor, koreksi kolom prediksi, simpan sebagai
      rec_roboflow_final.txt (format: path<TAB>teks)
"""
import os
os.environ['FLAGS_use_onednn'] = '0'
import sys

sys.path.insert(0, r"D:\svtr_export_work\PaddleOCR")
import cv2

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
TEMPLATE = os.path.join(PROJ, "Data", "Annotation", "rec_roboflow_template.txt")
OUT = os.path.join(PROJ, "Data", "Annotation", "rec_roboflow_prelabeled.txt")
INFER_DIR = os.path.join(PROJ, "Src", "Model", "Inference")
DICT = os.path.join(INFER_DIR, "custom_dict.txt")

from tools.infer.predict_rec import TextRecognizer
import tools.infer.utility as utility


def main():
    args = utility.parse_args()
    args.rec_model_dir = INFER_DIR
    args.rec_char_dict_path = DICT
    args.rec_image_shape = "3, 48, 320"
    args.use_space_char = True
    args.use_gpu = False
    rec = TextRecognizer(args)

    paths = []
    with open(TEMPLATE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")[0]
            paths.append(p)

    imgs, valid_paths = [], []
    for rel in paths:
        full = os.path.join(PROJ, rel)
        img = cv2.imread(full)
        if img is not None:
            imgs.append(img)
            valid_paths.append(rel)

    print(f"Pre-label {len(imgs)} crop...")
    results = []
    BATCH = 64
    for i in range(0, len(imgs), BATCH):
        res, _ = rec(imgs[i:i + BATCH])
        results.extend(res)

    with open(OUT, "w", encoding="utf-8") as f:
        for rel, (text, score) in zip(valid_paths, results):
            f.write(f"{rel}\t{text}\t{score:.3f}\n")

    print(f"Selesai -> {OUT}")
    print("Koreksi manual kolom teks, lalu simpan sbg rec_roboflow_final.txt (path<TAB>teks)")


if __name__ == "__main__":
    main()
