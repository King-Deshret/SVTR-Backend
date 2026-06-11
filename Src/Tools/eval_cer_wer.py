# -*- coding: utf-8 -*-
"""
Evaluasi CER (Character Error Rate) & WER (Word Error Rate) model recognition
pada val set BERSIH (val_clean.txt) -> angka jujur tanpa leakage.

Jalankan dengan venv yang punya paddleocr 2.7.3:
    D:\svtr_export_work\exp_env\Scripts\python.exe Src/Tools/eval_cer_wer.py
"""
import os
os.environ['FLAGS_use_onednn'] = '0'
import sys

sys.path.insert(0, r"D:\svtr_export_work\PaddleOCR")
import cv2

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
VAL = os.path.join(PROJ, "Data", "Annotation", "val_clean.txt")
INFER_DIR = os.path.join(PROJ, "Src", "Model", "Inference")
DICT = os.path.join(INFER_DIR, "custom_dict.txt")

from tools.infer.predict_rec import TextRecognizer
import tools.infer.utility as utility


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[n]


def parse(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split("\t")
            if len(p) >= 2:
                rows.append((p[0], p[1]))
    if limit:
        rows = rows[:limit]
    return rows


def main():
    args = utility.parse_args()
    args.rec_model_dir = INFER_DIR
    args.rec_char_dict_path = DICT
    args.rec_image_shape = "3, 48, 320"
    args.use_space_char = True
    args.use_gpu = False
    recognizer = TextRecognizer(args)

    rows = parse(VAL, limit=500)  # sampel 500 untuk cepat
    imgs, gts = [], []
    for relimg, gt in rows:
        path = relimg if os.path.isabs(relimg) else os.path.join(PROJ, relimg)
        img = cv2.imread(path)
        if img is not None:
            imgs.append(img)
            gts.append(gt)

    print(f"Evaluasi {len(imgs)} gambar...")
    BATCH = 64
    preds = []
    for i in range(0, len(imgs), BATCH):
        res, _ = recognizer(imgs[i:i + BATCH])
        preds.extend([t for t, _ in res])

    tot_char = tot_cerr = 0
    exact = 0
    for gt, pred in zip(gts, preds):
        tot_char += len(gt)
        tot_cerr += edit_distance(pred, gt)
        if pred.strip() == gt.strip():
            exact += 1

    cer = 100 * tot_cerr / max(1, tot_char)
    wer = 100 * (1 - exact / max(1, len(gts)))

    print("\n" + "=" * 50)
    print(f"Sampel dievaluasi : {len(gts)}")
    print(f"CER (Char Error Rate) : {cer:.2f}%")
    print(f"WER (Word Error Rate) : {wer:.2f}%")
    print(f"Exact match accuracy  : {100*exact/max(1,len(gts)):.2f}%")
    print("=" * 50)
    print("\nContoh prediksi:")
    for gt, pred in list(zip(gts, preds))[:15]:
        ok = "OK" if gt.strip() == pred.strip() else "X "
        print(f"  [{ok}] GT={gt!r:20} PRED={pred!r}")


if __name__ == "__main__":
    main()
