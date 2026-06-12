# -*- coding: utf-8 -*-
"""
Eksperimen preprocessing untuk korelasi Metode Numerik.

Membandingkan preprocessing LAMA vs V2 pada foto HP (FULL pipeline:
detection -> crop -> recognition), lalu menganalisis metrik citra.

Mengukur dampak nyata parameter preprocessing terhadap hasil OCR.

Jalankan dengan venv paddleocr 2.7.3:
  D:\svtr_export_work\exp_env\Scripts\python.exe Src/Tools/experiment_preprocessing.py
"""
import os
import sys

os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ)

import cv2
from paddleocr import PaddleOCR
from Src.Preprocessing.enhancedpicture import EnhancedPicture
from Src.Preprocessing.enhancedpicture_v2 import EnhancedPictureV2

TEST_IMAGES = [
    (r"d:\SVTR-Project\Data\Raw\07 JUL2026 SBM H28D (1).jpeg", "07 JUL2026 SBM H28D"),
    (r"d:\SVTR-Project\Data\Raw\07 JUL2026 SBM H28D (2).jpeg", "07 JUL2026 SBM H28D"),
    (r"d:\SVTR-Project\Data\Raw\07 JUL2026 SBM H28D (3).jpeg", "07 JUL2026 SBM H28D"),
    (r"d:\SVTR-Project\Data\Raw\07 JUL2026 SBM H28D (4).jpeg", "07 JUL2026 SBM H28D"),
]

INFER_DIR = os.path.join(PROJ, "Src", "Model", "Inference")
DICT = os.path.join(INFER_DIR, "custom_dict.txt")


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
            prev = tmp
    return dp[n]


def cer(pred, gt):
    return edit_distance(pred, gt) / max(1, len(gt))


def build_ocr():
    return PaddleOCR(
        rec_model_dir=INFER_DIR,
        rec_char_dict_path=DICT,
        rec_image_shape="3, 48, 320",
        use_angle_cls=True, lang='en', use_gpu=False,
        use_space_char=True,
    )


def run_full(ocr, image):
    """Full pipeline det+rec, gabung semua teks terdeteksi."""
    res = ocr.ocr(image, cls=True)
    if not res or res[0] is None:
        return ""
    lines = []
    for line in res[0]:
        try:
            lines.append(line[1][0])
        except Exception:
            pass
    return " ".join(lines)


def main():
    ocr = build_ocr()
    old = EnhancedPicture()
    new = EnhancedPictureV2()

    print("=" * 70)
    print("PERBANDINGAN PREPROCESSING (full pipeline det+rec)")
    print("CER lebih kecil = lebih baik")
    print("=" * 70)

    tot = {"raw": 0, "lama": 0, "v2": 0}
    for path, gt in TEST_IMAGES:
        if not os.path.exists(path):
            print(f"SKIP: {path}"); continue
        img = cv2.imread(img := path)
        img = cv2.imread(path)

        # analisis metrik citra (untuk laporan)
        m = new.analyze(img)

        t_raw = run_full(ocr, img)
        t_old = run_full(ocr, old.process_without_resize(img))
        t_new = run_full(ocr, new.process_without_resize(img))

        c_raw, c_old, c_new = cer(t_raw, gt), cer(t_old, gt), cer(t_new, gt)
        tot["raw"] += c_raw; tot["lama"] += c_old; tot["v2"] += c_new

        print(f"\n{os.path.basename(path)}")
        print(f"  metrik: brightness={m['brightness']:.0f} blur={m['blur_var_laplacian']:.0f} "
              f"glare={m['glare_ratio']:.3f}")
        print(f"  RAW : {t_raw!r:35} CER={c_raw:.3f}")
        print(f"  LAMA: {t_old!r:35} CER={c_old:.3f}")
        print(f"  V2  : {t_new!r:35} CER={c_new:.3f}")

    n = len([1 for p, _ in TEST_IMAGES if os.path.exists(p)])
    print("\n" + "=" * 70)
    print(f"RATA-RATA CER (n={n}):  RAW={tot['raw']/n:.3f}  "
          f"LAMA={tot['lama']/n:.3f}  V2={tot['v2']/n:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
