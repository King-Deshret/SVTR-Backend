# -*- coding: utf-8 -*-
"""
Perbandingan Galat & Analisis Numerik: Model v2 vs v3.
Dirancang mengikuti silabus Metode Numerik (Galat, Regresi, Interpolasi,
Root Finding, ODE, SPL, Diferensiasi Numerik).

Mengolah kurva loss training kedua model untuk laporan BAB III (Korelasi)
dan BAB Galat.

Usage:
  python Src/Tools/compare_models_numerik.py log_v2.txt log_v3.txt
Default:
  log_v2 = Data/Annotation/train_log_v2.txt
  log_v3 = Data/Annotation/train_log_v3.txt  (isi setelah training v3 selesai)
"""
import os
import re
import sys
import math

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")


def parse_log(path):
    """Ambil loss rata-rata per epoch dari log PaddleOCR."""
    if not os.path.exists(path):
        return [], []
    text = open(path, encoding="utf-8", errors="ignore").read()
    by_epoch = {}
    for m in re.finditer(r'epoch:\s*\[(\d+)/\d+\][^\n]*?loss:\s*([\d.]+)', text):
        e = int(m.group(1)); l = float(m.group(2))
        by_epoch.setdefault(e, []).append(l)
    epochs = sorted(by_epoch)
    losses = [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs]
    return epochs, losses


# ---------- Galat (silabus 2.1) ----------
def rmse(actual, pred):
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, pred)) / len(actual))

def mae(actual, pred):
    return sum(abs(a - p) for a, p in zip(actual, pred)) / len(actual)

def galat_relatif(actual, pred):
    """Galat relatif rata-rata (%)."""
    return 100 * sum(abs(a - p) / abs(a) for a, p in zip(actual, pred) if a != 0) / len(actual)


# ---------- SPL untuk regresi (silabus 2.7 + 2.2) ----------
def gauss_solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            if abs(M[c][c]) < 1e-12: continue
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0
    return x

def poly_fit(xs, ys, deg):
    n = deg + 1
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum((x ** i) * y for x, y in zip(xs, ys)) for i in range(n)]
    return gauss_solve(A, b)

def poly_eval(c, x):
    return sum(ci * x ** i for i, ci in enumerate(c))

def r_squared(ys, pred):
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def analyze_one(name, epochs, losses):
    print(f"\n{'='*55}\nMODEL {name}\n{'='*55}")
    if not losses:
        print("  (log tidak ditemukan/ kosong)")
        return None
    print(f"  Epoch: {epochs[0]}-{epochs[-1]} ({len(epochs)} titik)")
    print(f"  Loss awal: {losses[0]:.3f} -> akhir: {losses[-1]:.3f}")
    print(f"  Reduksi loss: {100*(losses[0]-losses[-1])/losses[0]:.1f}%")

    # regresi polinomial deg-3 + R^2 (silabus 2.2)
    c = poly_fit(epochs, losses, 3)
    pred = [poly_eval(c, x) for x in epochs]
    print(f"  Regresi deg-3 R^2 = {r_squared(losses, pred):.4f}")
    print(f"  Galat fitting: RMSE={rmse(losses,pred):.3f}, MAE={mae(losses,pred):.3f}")
    return {"epochs": epochs, "losses": losses, "coeffs": c, "pred": pred}


def main():
    log_v2 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "Data", "Annotation", "train_log_v2.txt")
    log_v3 = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJ, "Data", "Annotation", "train_log_v3.txt")

    e2, l2 = parse_log(log_v2)
    e3, l3 = parse_log(log_v3)

    m2 = analyze_one("v2 (dataset 67% umum)", e2, l2)
    m3 = analyze_one("v3 (dataset seimbang)", e3, l3)

    # ---------- PERBANDINGAN GALAT v2 vs v3 ----------
    if m2 and m3:
        print(f"\n{'='*55}\nPERBANDINGAN GALAT v2 vs v3 (BAB Galat)\n{'='*55}")
        # samakan panjang (epoch beririsan)
        n = min(len(l2), len(l3))
        a, b = l2[:n], l3[:n]
        print(f"  RMSE antar-kurva loss : {rmse(a,b):.3f}")
        print(f"  MAE antar-kurva loss  : {mae(a,b):.3f}")
        print(f"  Galat relatif rata2   : {galat_relatif(a,b):.2f}%")
        print(f"  Loss akhir v2={l2[-1]:.3f} vs v3={l3[-1]:.3f}")
        better = "v3" if l3[-1] < l2[-1] else "v2"
        print(f"  -> Model {better} konvergen ke loss lebih rendah")

    # ---------- grafik perbandingan ----------
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 6))
        if m2: plt.plot(e2, l2, label='v2 (67% umum)', color='orange')
        if m3: plt.plot(e3, l3, label='v3 (seimbang)', color='steelblue')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
        plt.title('Perbandingan Kurva Loss: Model v2 vs v3')
        out = os.path.join(PROJ, "Data", "Annotation", "compare_loss_v2_v3.png")
        plt.savefig(out, dpi=150)
        print(f"\nGrafik perbandingan: {out}")
    except Exception as e:
        print(f"(grafik dilewati: {e})")


if __name__ == "__main__":
    main()
