# Korelasi Metode Numerik dengan Proyek SVTR (Scene Text Recognition)

> Dokumen ringkas untuk dipelajari. Prinsip korelasi yang dipakai:
> **ada DATA + ada RUMUS + MASUK AKAL.** Bukan sekadar menempelkan istilah.

Proyek: Pengenalan teks (tanggal kadaluarsa & kode produksi) pada kemasan
pangan menggunakan model Deep Learning **SVTR**. Model dilatih (training)
dengan PaddleOCR, lalu di-deploy ke API + aplikasi Android.

Sumber data korelasi:
- `train_log_v2.txt` — kurva loss & metrik per epoch (epoch 1-100)
- `per_char_metrics.csv` — precision/recall/F1 per karakter
- Hasil evaluasi: CER, WER, accuracy di validation set
- `enhancedpicture.py` — preprocessing citra (resize, koreksi cahaya)

---

## Ringkasan: Peta Korelasi 9 Materi

| Materi Silabus | Kekuatan | Cukup 1 model? | Data yang dipakai |
|---|---|---|---|
| 2.1 Galat & Angka Signifikan | KUAT | Ya | CER/WER/loss (galat prediksi) |
| 2.2 Regresi / Curve Fitting | KUAT | Ya | titik (epoch, loss) → fit fungsi |
| 2.9 Optimasi Numerik | KUAT | Ya | kurva loss + optimizer Adam |
| 2.4 Root Finding | SEDANG | Ya | persamaan regresi loss → cari akar |
| 2.7 Sistem Persamaan Linear | SEDANG | Ya | normal equation regresi (Gauss) |
| 2.8 Diferensiasi Numerik | SEDANG | Ya | turunan numerik kurva loss |
| 2.5 Interpolasi | SEDANG | Ya | preprocessing resize/warp citra |
| 2.3 Integrasi Numerik | LEMAH | lebih baik 2 model | luas bawah kurva loss |
| 2.6 ODE | LEMAH (abstrak) | Ya | model peluruhan loss |

**Kesimpulan utama:** Sebagian besar materi bisa dikorelasikan **hanya dengan
satu model (v2)**. Perbandingan dua model (v2 vs v3) hanya perlu untuk
melengkapi materi Galat & Integrasi, bukan keharusan.

---

## Penjelasan Per Materi

### 2.1 Galat & Angka Signifikan — KORELASI PALING KUAT

**Kenapa berkorelasi:** CER (Character Error Rate) secara harfiah ADALAH galat.
Confidence score yang menolak hasil di bawah ambang = penerapan galat untuk
pengambilan keputusan.

**Data:** prediksi model vs ground truth (teks asli).

**Rumus (sesuai silabus):**
- RMSE = sqrt( (1/n) * Σ (xi_sejati − xi_pendekatan)² )
- MAE  = (1/n) * Σ |xi_sejati − xi_pendekatan|
- Galat relatif = |nilai_sejati − nilai_pendekatan| / |nilai_sejati| × 100%
- CER = (edit distance) / (jumlah karakter ground truth)

**Masuk akal:** Ini jantung proyek. Sistem "confidence-aware" menolak hasil
kalau galatnya tinggi → langsung memakai konsep galat.

---

### 2.2 Regresi / Curve Fitting — KUAT

**Kenapa berkorelasi:** (a) Kurva loss training bisa di-fit jadi fungsi
matematis. (b) Training neural network itu sendiri = regresi (meminimalkan
selisih prediksi vs aktual).

**Data:** titik (epoch, loss) dari log training.

**Rumus:**
- Regresi linear: y = a0 + a1·x
- Least squares: a1 = (nΣxy − ΣxΣy) / (nΣx² − (Σx)²)
- Koefisien determinasi: R² = 1 − (SS_res / SS_tot)

**Hasil nyata (model v2):** regresi polinomial derajat 3 menghasilkan
**R² ≈ 0.87** terhadap kurva loss → fungsi cocok menggambarkan tren.

---

### 2.9 Optimasi Numerik — KUAT (secara konsep paling inti)

**Kenapa berkorelasi:** Proses training = optimasi numerik yang meminimalkan
fungsi loss. Optimizer yang dipakai proyek ini adalah **Adam** (terlihat di
config training `svtr_finetune.yml`).

**Data:** kurva loss menurun per epoch (132.98 → 6.63, reduksi 95%).

**Rumus (sesuai silabus):**
- Gradient Descent: θ_baru = θ_lama − α · ∇L(θ)
- Adam: gabungan momentum + adaptive learning rate

**Masuk akal:** Kurva loss yang menurun adalah BUKTI VISUAL gradient descent
bekerja. α = learning rate (di config: 0.0005, dengan Cosine schedule).

---

### 2.4 Root Finding — SEDANG (didukung silabus)

**Kenapa berkorelasi:** Silabus menyatakan "pencarian akar berkaitan dengan
kondisi gradien nol (stationary point) pada optimasi loss." Jadi: cari epoch
saat turunan loss = 0 (titik konvergen/model berhenti membaik).

**Data:** persamaan regresi loss (dari materi 2.2).

**Rumus:**
- Biseksi: c = (a+b)/2, cek tanda f(a)·f(c)
- Newton-Raphson: x_{n+1} = x_n − f(x_n)/f'(x_n)
- Secant: x_{n+1} = x_n − f(x_n)·(x_n − x_{n-1})/(f(x_n) − f(x_{n-1}))

**Masuk akal:** "decode grafik loss → persamaan → cari akar turunannya =
epoch konvergen." Perbandingan iterasi: Newton-Raphson biasanya konvergen
paling cepat, Biseksi paling lambat tapi paling stabil.

---

### 2.7 Sistem Persamaan Linear — SEDANG (otomatis terpakai)

**Kenapa berkorelasi:** (a) Regresi least-squares DISELESAIKAN via SPL
(normal equation). (b) Setiap layer neural network = transformasi linear
Wx + b.

**Data:** matriks normal equation dari proses regresi.

**Rumus:**
- Bentuk umum: A·x = b
- Eliminasi Gauss: reduksi ke segitiga atas + substitusi mundur
- Dekomposisi LU: A = L·U

**Masuk akal:** Setiap kali fitting kurva loss, di belakangnya menyelesaikan
SPL. Jadi materi ini "menyatu" dengan regresi.

---

### 2.8 Diferensiasi Numerik — SEDANG (sering dilupakan, padahal bagus)

**Kenapa berkorelasi:** (a) Backpropagation = diferensiasi (chain rule).
(b) Turunan numerik kurva loss menunjukkan LAJU penurunan (kapan training
melambat).

**Data:** kurva loss per epoch.

**Rumus:**
- Beda maju: f'(x) ≈ (f(x+h) − f(x)) / h
- Beda tengah: f'(x) ≈ (f(x+h) − f(x−h)) / 2h  (akurasi lebih tinggi)
- Turunan kedua: f''(x) ≈ (f(x+h) − 2f(x) + f(x−h)) / h²

**Masuk akal:** Turunan loss mendekati 0 = model konvergen. Menghubungkan
langsung ke Root Finding (cari kapan turunan = 0).

---

### 2.5 Interpolasi — SEDANG (korelasikan ke PREPROCESSING, bukan loss)

**Kenapa berkorelasi:** Silabus menyebut interpolasi dipakai di image resizing
& warping. Proyek ini BENAR-BENAR memakainya di preprocessing:
`cv2.resize` dengan `INTER_CUBIC` dan `INTER_AREA` = interpolasi.

**Data:** citra sebelum & sesudah resize/normalisasi.

**Rumus:**
- Interpolasi linear: f(x) = f(x0) + (x − x0)·(f(x1) − f(x0))/(x1 − x0)
- Lagrange & Newton (selisih terbagi)

**Masuk akal:** Sebelum dikenali, citra teks di-resize ke ukuran standar
(tinggi 48px). Proses memperbesar/memperkecil piksel = interpolasi.
Ini korelasi paling jujur untuk interpolasi (bukan dipaksakan ke kurva loss).

---

### 2.3 Integrasi Numerik — LEMAH (pakai untuk perbandingan model)

**Kenapa berkorelasi (tipis):** "Area under loss curve" = total akumulasi
loss sepanjang training. Bisa jadi metrik efisiensi: area lebih kecil =
konvergen lebih cepat.

**Data:** kurva loss dua model (v2 vs v3).

**Rumus:**
- Trapesium: ∫ ≈ (h/2)·[f(x0) + 2f(x1) + ... + 2f(x_{n-1}) + f(xn)]
- Simpson 1/3: ∫ ≈ (h/3)·[f(x0) + 4f(x1) + 2f(x2) + ... + f(xn)]

**Masuk akal:** Gunakan untuk MEMBANDINGKAN dua model — model dengan
area-under-loss lebih kecil = belajar lebih efisien. Di sinilah perbandingan
v2 vs v3 relevan.

---

### 2.6 ODE — LEMAH/ABSTRAK (pakai bila perlu melengkapi)

**Kenapa berkorelasi (abstrak):** Silabus bilang ODE relevan dengan dinamika
pembaruan weight. Penurunan loss bisa dimodelkan sebagai peluruhan
eksponensial: dL/dt = −k·L.

**Data:** kurva loss.

**Rumus:**
- Euler: y_{i+1} = y_i + h·f(x_i, y_i)
- Heun (Euler modifikasi): rata-rata slope awal & akhir
- RK4: kombinasi 4 slope

**Masuk akal:** Modelkan kurva loss sebagai sistem peluruhan, bandingkan
solusi numerik (Euler/RK4) dengan data aktual. Framing: "pemodelan dinamika
penurunan loss". Pakai bila butuh mencakup materi ini.

---

## Alur Elegan: 1 Model → 4 Materi Sekaligus

Dari **satu** kurva loss model v2, bisa menyentuh 4 materi berurutan:

1. **Regresi (2.2)** → fit kurva loss jadi persamaan polinomial
2. **SPL (2.7)** → persamaan regresi diselesaikan via Eliminasi Gauss
3. **Diferensiasi (2.8)** → turunkan persamaan untuk dapat laju penurunan
4. **Root Finding (2.4)** → cari akar turunan = epoch konvergen

Tambah **Galat (2.1)** untuk ukur kualitas fitting (RMSE/MAE), dan
**Optimasi (2.9)** sebagai konteks (loss turun = gradient descent bekerja).

---

## Kesimpulan

1. Metode Numerik **bukan tempelan** pada proyek Deep Learning — ia fondasi
   nyata: training = optimasi, loss = galat, preprocessing = interpolasi.
2. **7 dari 9 materi** bisa dikorelasikan dengan **kuat/sedang** memakai
   data satu model saja (v2).
3. Korelasi yang jujur butuh: **data nyata** (kurva loss, CER/WER, citra) +
   **rumus silabus** + **interpretasi masuk akal** — bukan sekadar menyebut
   nama metode.
4. Perbandingan dua model (v2 vs v3) hanya nilai tambah untuk Galat &
   Integrasi, bukan syarat semua korelasi.

---
*Disusun sebagai bahan belajar. Angka (R²=0.87, reduksi loss 95%) berasal dari
log training nyata model v2 proyek SVTR.*
