# -*- coding: utf-8 -*-
"""
Analisis Metode Numerik pada data training SVTR (loss & accuracy per epoch).

Mengaplikasikan materi kuliah Metode Numerik ke kurva training:
  1. Regresi (linear, polinomial, eksponensial)  -> model tren loss
  2. Interpolasi (linear, Lagrange/Newton)        -> estimasi nilai antar-epoch
  3. Root Finding (biseksi, Newton-Raphson, secant) -> cari epoch saat loss = target
  4. ODE (Euler) + Deret Taylor                   -> model peluruhan loss
  5. Galat/Error (RMSE, MAE, MAPE, error absolut) -> kualitas fitting
  6. Sistem Persamaan Linear (normal equation)    -> dasar regresi polinomial

INPUT: file log training (train_log.txt dari Kaggle) ATAU CSV epoch,loss,acc
OUTPUT: tabel + grafik + nilai numerik untuk laporan

Jalankan:
  python Src/Tools/numerical_analysis.py path/ke/train_log.txt
"""
import os
import re
import sys
import math


# ============================================================
# 0. PARSING DATA dari log PaddleOCR
# ============================================================
def parse_log(path):
    """Ambil (epoch, loss, acc) dari train_log.txt PaddleOCR."""
    epochs, losses, accs = [], [], []
    text = open(path, encoding="utf-8", errors="ignore").read()
    # baris: epoch: [12/100], ... loss: 43.59 ... acc: 0.0
    pat = re.compile(r'epoch:\s*\[(\d+)/\d+\].*?loss:\s*([\d.]+)', re.S)
    # ambil eval metric: "cur metric, acc: 0.66"
    # kita kumpulkan loss per-epoch (rata-rata) dan acc terakhir per epoch
    by_epoch_loss = {}
    for m in re.finditer(r'epoch:\s*\[(\d+)/\d+\][^\n]*?loss:\s*([\d.]+)', text):
        e = int(m.group(1)); l = float(m.group(2))
        by_epoch_loss.setdefault(e, []).append(l)
    for e in sorted(by_epoch_loss):
        epochs.append(e)
        losses.append(sum(by_epoch_loss[e]) / len(by_epoch_loss[e]))
    return epochs, losses


def load_csv(path):
    epochs, losses = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    epochs.append(float(parts[0])); losses.append(float(parts[1]))
                except ValueError:
                    pass
    return epochs, losses


# ============================================================
# 6. SISTEM PERSAMAAN LINEAR (Gaussian elimination)
#    dipakai untuk menyelesaikan normal equation regresi polinomial
# ============================================================
def solve_linear_system(A, b):
    """Eliminasi Gauss dengan partial pivoting. A: n x n, b: n."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # pivot
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(col + 1, n):
            f = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    # back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
    return x


# ============================================================
# 1. REGRESI
# ============================================================
def poly_regression(xs, ys, degree):
    """Regresi polinomial via normal equation (pakai solve_linear_system)."""
    n = degree + 1
    # matriks normal: sum x^(i+j)
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum((x ** i) * y for x, y in zip(xs, ys)) for i in range(n)]
    coeffs = solve_linear_system(A, b)  # c0 + c1 x + c2 x^2 ...
    return coeffs


def poly_eval(coeffs, x):
    return sum(c * (x ** i) for i, c in enumerate(coeffs))


def exp_regression(xs, ys):
    """Fit y = a*exp(b*x) via linearisasi ln(y)=ln(a)+b*x (loss>0)."""
    lx = [x for x, y in zip(xs, ys) if y > 0]
    ly = [math.log(y) for y in ys if y > 0]
    # regresi linear ln(y) = A + b x
    n = len(lx)
    sx = sum(lx); sy = sum(ly); sxx = sum(x * x for x in lx); sxy = sum(x * y for x, y in zip(lx, ly))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    A = (sy - b * sx) / n
    return math.exp(A), b  # a, b


# ============================================================
# 5. GALAT / ERROR
# ============================================================
def error_metrics(actual, predicted):
    n = len(actual)
    abs_err = [abs(a - p) for a, p in zip(actual, predicted)]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n)
    mape = 100 * sum(abs(a - p) / abs(a) for a, p in zip(actual, predicted) if a != 0) / n
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "max_abs_err": max(abs_err)}


# ============================================================
# 2. INTERPOLASI
# ============================================================
def linear_interp(xs, ys, xq):
    for i in range(len(xs) - 1):
        if xs[i] <= xq <= xs[i + 1]:
            t = (xq - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return None


def lagrange_interp(xs, ys, xq):
    """Interpolasi Lagrange (pakai subset titik agar stabil)."""
    total = 0.0
    n = len(xs)
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if j != i:
                term *= (xq - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


# ============================================================
# 3. ROOT FINDING - cari epoch saat loss = target
# ============================================================
def make_loss_function(coeffs, target):
    """f(x) = poly(x) - target ; akarnya = epoch saat loss=target."""
    return lambda x: poly_eval(coeffs, x) - target


def bisection(f, a, b, tol=1e-4, maxit=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None, 0
    for it in range(maxit):
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol:
            return c, it + 1
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2, maxit


def newton_raphson(f, df, x0, tol=1e-4, maxit=100):
    x = x0
    for it in range(maxit):
        fx = f(x)
        if abs(fx) < tol:
            return x, it + 1
        d = df(x)
        if abs(d) < 1e-12:
            break
        x -= fx / d
    return x, maxit


def secant(f, x0, x1, tol=1e-4, maxit=100):
    for it in range(maxit):
        f0, f1 = f(x0), f(x1)
        if abs(f1) < tol:
            return x1, it + 1
        if abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x2
    return x1, maxit


def poly_derivative(coeffs):
    return [i * c for i, c in enumerate(coeffs)][1:] or [0.0]


# ============================================================
# 4. ODE (Euler) - model peluruhan loss dL/dx = -k*L
# ============================================================
def euler_ode(L0, k, x_start, x_end, h=1.0):
    xs, Ls = [x_start], [L0]
    x, L = x_start, L0
    while x < x_end:
        L = L + h * (-k * L)   # dL/dx = -k L
        x = x + h
        xs.append(x); Ls.append(L)
    return xs, Ls


def taylor_exp_decay(L0, k, x0, x, order=3):
    """Deret Taylor untuk L(x)=L0*exp(-k(x-x0)) di sekitar x0."""
    dx = x - x0
    total = 0.0
    for n in range(order + 1):
        term = L0 * ((-k) ** n) * (dx ** n) / math.factorial(n)
        total += term
    return total


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python numerical_analysis.py <train_log.txt | data.csv>")
        return
    path = sys.argv[1]
    if path.endswith(".csv"):
        xs, ys = load_csv(path)
    else:
        xs, ys = parse_log(path)

    if len(xs) < 4:
        print("Data terlalu sedikit. Pastikan log berisi baris 'epoch: [n/N] ... loss: ...'")
        return

    print(f"Data: {len(xs)} titik (epoch {int(min(xs))}-{int(max(xs))})")
    print(f"Loss awal: {ys[0]:.3f}, Loss akhir: {ys[-1]:.3f}\n")

    # --- 1 & 6: Regresi polinomial (pakai SPL) ---
    print("=" * 55)
    print("1. REGRESI POLINOMIAL (via Sistem Persamaan Linear)")
    print("=" * 55)
    for deg in [1, 2, 3]:
        c = poly_regression(xs, ys, deg)
        pred = [poly_eval(c, x) for x in xs]
        err = error_metrics(ys, pred)
        print(f"  Derajat {deg}: RMSE={err['RMSE']:.3f}, MAPE={err['MAPE(%)']:.2f}%")
    best_coeffs = poly_regression(xs, ys, 3)
    print(f"  Koefisien deg-3: {[round(c,5) for c in best_coeffs]}")

    # --- exp regression ---
    print("\n2. REGRESI EKSPONENSIAL  y = a*exp(b*x)")
    a, b = exp_regression(xs, ys)
    pred_exp = [a * math.exp(b * x) for x in xs]
    err_exp = error_metrics(ys, pred_exp)
    print(f"  a={a:.3f}, b={b:.5f}  (b<0 = peluruhan)")
    print(f"  RMSE={err_exp['RMSE']:.3f}, MAPE={err_exp['MAPE(%)']:.2f}%")

    # --- 2: Interpolasi ---
    print("\n" + "=" * 55)
    print("3. INTERPOLASI (estimasi loss di epoch antara)")
    print("=" * 55)
    mid = (xs[0] + xs[-1]) / 2
    li = linear_interp(xs, ys, mid)
    print(f"  Loss di epoch {mid:.0f} (linear interp): {li:.3f}")

    # --- 3: Root finding (epoch saat loss=target) ---
    print("\n" + "=" * 55)
    print("4. ROOT FINDING: epoch saat loss mencapai target")
    print("=" * 55)
    target = (min(ys) + max(ys)) / 2  # target loss = tengah
    f = make_loss_function(best_coeffs, target)
    dcoeffs = poly_derivative(best_coeffs)
    df = lambda x: poly_eval(dcoeffs, x)
    print(f"  Target loss = {target:.3f}")
    rb, ib = bisection(f, xs[0], xs[-1])
    rn, iC = newton_raphson(f, df, (xs[0] + xs[-1]) / 2)
    rs, isi = secant(f, xs[0], xs[-1])
    print(f"  Biseksi       : epoch={rb:.2f} ({ib} iterasi)" if rb else "  Biseksi: tidak ada akar di interval")
    print(f"  Newton-Raphson: epoch={rn:.2f} ({iC} iterasi)")
    print(f"  Secant        : epoch={rs:.2f} ({isi} iterasi)")

    # --- 4: ODE + Taylor ---
    print("\n" + "=" * 55)
    print("5. ODE (Euler) & DERET TAYLOR: model peluruhan loss")
    print("=" * 55)
    k = -b if b < 0 else 0.01
    ex, eL = euler_ode(ys[0], k, xs[0], xs[-1], h=(xs[-1]-xs[0])/len(xs))
    print(f"  Euler: L({xs[0]:.0f})={ys[0]:.2f} -> L({xs[-1]:.0f})~={eL[-1]:.3f}")
    taylor_val = taylor_exp_decay(ys[0], k, xs[0], xs[-1], order=3)
    print(f"  Taylor orde-3 di epoch {xs[-1]:.0f}: {taylor_val:.3f}")

    # --- grafik ---
    try:
        import matplotlib.pyplot as plt
        xfit = [xs[0] + i*(xs[-1]-xs[0])/200 for i in range(201)]
        yfit = [poly_eval(best_coeffs, x) for x in xfit]
        plt.figure(figsize=(10,6))
        plt.scatter(xs, ys, s=12, label='data aktual', color='steelblue')
        plt.plot(xfit, yfit, 'r-', label='regresi poly deg-3')
        plt.plot(xs, pred_exp, 'g--', label='regresi eksponensial')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.title('Analisis Numerik: Regresi Kurva Loss Training')
        plt.grid(alpha=0.3)
        out = os.path.join(os.path.dirname(path), 'numerical_analysis.png')
        plt.savefig(out, dpi=150)
        print(f"\nGrafik disimpan: {out}")
    except Exception as e:
        print(f"\n(grafik dilewati: {e})")


if __name__ == "__main__":
    main()
