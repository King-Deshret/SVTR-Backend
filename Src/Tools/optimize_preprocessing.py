# -*- coding: utf-8 -*-
"""
Optimasi Parameter Preprocessing via Metode Numerik.

Mencari parameter preprocessing (gamma & CLAHE clipLimit) yang MEMINIMALKAN
rata-rata CER pada set foto uji, menggunakan:
  - GOLDEN SECTION SEARCH (silabus 2.9 Optimasi Numerik) untuk 1 parameter
  - GRID + interpolasi untuk visualisasi permukaan CER

Output: parameter optimal + tabel iterasi (untuk laporan Metode Numerik) +
grafik CER vs parameter.

Korelasi: ubah parameter -> CER berubah -> cari minimum secara numerik ->
terapkan ke proyek. Ini intervensi nyata & berdampak.

Jalankan:
  D:\svtr_export_work\exp_env\Scripts\python.exe Src/Tools/optimize_preprocessing.py
"""
import os
import sys

os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJ)

import cv2
import numpy as np
from paddleocr import PaddleOCR

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


# ---------- preprocessing parametrik ----------
def preprocess(img, gamma, clahe_clip):
    """Terapkan CLAHE(clip) + gamma pada kanal L (LAB)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)
    if abs(gamma - 1.0) > 1e-3:
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        l = cv2.LUT(l, table)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


_ocr = None
def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(rec_model_dir=INFER_DIR, rec_char_dict_path=DICT,
                         rec_image_shape="3, 48, 320", use_angle_cls=True,
                         lang='en', use_gpu=False, use_space_char=True, show_log=False)
    return _ocr


def run_full(image):
    res = get_ocr().ocr(image, cls=True)
    if not res or res[0] is None:
        return ""
    return " ".join(line[1][0] for line in res[0] if line and len(line) > 1)


# cache gambar
_imgs = [(cv2.imread(p), gt) for p, gt in TEST_IMAGES if os.path.exists(p)]


def mean_cer(gamma, clahe_clip):
    """Objective function: rata-rata CER pada semua foto uji."""
    total = 0.0
    for img, gt in _imgs:
        proc = preprocess(img, gamma, clahe_clip)
        total += cer(run_full(proc), gt)
    return total / len(_imgs)


# ---------- GOLDEN SECTION SEARCH (silabus 2.9) ----------
def golden_section(f, lo, hi, tol=0.08, maxit=15):
    gr = (5 ** 0.5 - 1) / 2
    history = []
    a, b = lo, hi
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = f(c1), f(c2)
    history.append((round(c1, 3), round(f1, 4)))
    history.append((round(c2, 3), round(f2, 4)))
    it = 0
    while abs(b - a) > tol and it < maxit:
        it += 1
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = f(c1)
            history.append((round(c1, 3), round(f1, 4)))
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = f(c2)
            history.append((round(c2, 3), round(f2, 4)))
    x_opt = (a + b) / 2
    return x_opt, f(x_opt), it, history


def main():
    print(f"Foto uji: {len(_imgs)} | GT: '07 JUL2026 SBM H28D'")
    print("\nBaseline:")
    print(f"  RAW (tanpa preprocessing) CER = {sum(cer(run_full(im),gt) for im,gt in _imgs)/len(_imgs):.4f}")

    # ---- Optimasi GAMMA (clahe tetap 2.0) ----
    print("\n" + "=" * 60)
    print("OPTIMASI GAMMA (Golden Section Search, clahe=2.0)")
    print("=" * 60)
    g_opt, g_cer, g_it, g_hist = golden_section(lambda g: mean_cer(g, 2.0), 0.4, 2.5)
    print("  Iterasi (gamma, CER):")
    for h in g_hist:
        print(f"    gamma={h[0]:.3f} -> CER={h[1]:.4f}")
    print(f"  >> Gamma optimal = {g_opt:.3f}, CER = {g_cer:.4f} ({g_it} iterasi)")

    # ---- Optimasi CLAHE clip (gamma optimal) ----
    print("\n" + "=" * 60)
    print(f"OPTIMASI CLAHE clipLimit (Golden Section, gamma={g_opt:.2f})")
    print("=" * 60)
    c_opt, c_cer, c_it, c_hist = golden_section(lambda c: mean_cer(g_opt, c), 0.5, 6.0)
    print("  Iterasi (clip, CER):")
    for h in c_hist:
        print(f"    clip={h[0]:.3f} -> CER={h[1]:.4f}")
    print(f"  >> Clip optimal = {c_opt:.3f}, CER = {c_cer:.4f} ({c_it} iterasi)")

    # ---- Ringkasan ----
    raw_cer = sum(cer(run_full(im), gt) for im, gt in _imgs) / len(_imgs)
    print("\n" + "=" * 60)
    print("RINGKASAN (CER, makin kecil makin baik)")
    print("=" * 60)
    print(f"  RAW                      : {raw_cer:.4f}")
    print(f"  Optimal (gamma={g_opt:.2f}, clip={c_opt:.2f}): {c_cer:.4f}")
    delta = raw_cer - c_cer
    if delta > 0:
        print(f"  -> Preprocessing optimal MEMPERBAIKI CER sebesar {delta:.4f}")
    else:
        print(f"  -> RAW tetap lebih baik (preprocessing tidak membantu untuk kasus ini)")
        print(f"     Kesimpulan: untuk teks dot-matrix yakult, raw input optimal.")

    # ---- grafik CER vs gamma (sweep untuk visualisasi) ----
    try:
        import matplotlib.pyplot as plt
        gammas = [0.4 + i * 0.15 for i in range(15)]
        cers = [mean_cer(g, 2.0) for g in gammas]
        plt.figure(figsize=(9, 5))
        plt.plot(gammas, cers, 'o-', color='steelblue')
        plt.axvline(g_opt, color='red', ls='--', label=f'gamma optimal={g_opt:.2f}')
        plt.xlabel('Gamma'); plt.ylabel('Rata-rata CER'); plt.legend(); plt.grid(alpha=0.3)
        plt.title('Optimasi Parameter Gamma vs CER (Golden Section Search)')
        out = os.path.join(PROJ, "Data", "Annotation", "optimize_gamma_cer.png")
        plt.savefig(out, dpi=150)
        print(f"\nGrafik: {out}")
    except Exception as e:
        print(f"(grafik dilewati: {e})")


if __name__ == "__main__":
    main()
