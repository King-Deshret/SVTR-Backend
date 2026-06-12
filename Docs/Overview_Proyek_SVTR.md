# Overview Proyek SVTR — Scene Text Recognition pada Kemasan Pangan

> Bahan belajar untuk memahami proyek secara menyeluruh:
> definisi, tech stack, alur kerja, dan progress.

---

## 1. Definisi & Tujuan Proyek

**Apa ini?** Sistem yang membaca teks pada kemasan pangan — khususnya
**tanggal kadaluarsa** dan **kode produksi** — secara otomatis dari foto,
menggunakan Deep Learning.

**Masalah yang dipecahkan:** Teks pada kemasan sulit dibaca OCR biasa karena:
- Permukaan mengilap / foil reflektif (glare)
- Cetakan dot-matrix / emboss kecil
- Permukaan melengkung (distorsi perspektif)
- Pencahayaan rendah / blur

**Solusi:** Model **SVTR** (Single Visual model for Text Recognition) yang
di-fine-tune khusus untuk teks kemasan, dilengkapi **confidence-aware OCR**
(menolak hasil yang tidak meyakinkan + minta foto ulang).

**Target keberhasilan:**
- Akurasi kata >85% pada kondisi normal
- Akurasi >75% pada kondisi degradasi (blur/glare/low-light)
- CER & WER di bawah 15%
- Inferensi <10 detik per gambar
- Aplikasi Android terintegrasi penuh

---

## 2. Tech Stack

### Backend (AI + API)
| Komponen | Teknologi | Fungsi |
|---|---|---|
| Framework DL | **PaddlePaddle 2.6** | Mesin deep learning (seperti TensorFlow/PyTorch) |
| Toolkit OCR | **PaddleOCR 2.7.3** | Pipeline OCR + tools training |
| Arsitektur model | **SVTR_LCNet** | Recognition (baca karakter) |
| Web framework | **Flask** | REST API server |
| Image processing | **OpenCV** | Preprocessing citra (CLAHE, resize) |
| WSGI server | **Gunicorn** | Production server |

### Frontend (Mobile)
| Komponen | Teknologi | Fungsi |
|---|---|---|
| Framework | **Flutter** | Aplikasi Android |
| State management | **Provider** | Kelola state app |
| HTTP client | **http package** | Komunikasi ke API |
| Storage | **SharedPreferences** | Simpan setting & history |

### Deployment & Tools
| Komponen | Teknologi | Fungsi |
|---|---|---|
| Hosting API | **HuggingFace Spaces** (Docker) | Server cloud gratis 24/7 |
| Training | **Kaggle** (GPU) | Latih model |
| Data sintetik | **TRDG** | Generate teks buatan |
| Dataset riil | **Roboflow** | Sumber gambar kemasan |
| Version control | **Git / GitHub** | Manajemen kode |

---

## 3. Arsitektur Sistem (Alur End-to-End)

```
[HP Android - Flutter]
      |  foto kemasan (base64)
      v
[HuggingFace Spaces - Flask API]
      |
      v
  Preprocessing (OpenCV: CLAHE, gamma, denoise)
      |
      v
  Pipeline PaddleOCR:
    1. Deteksi teks (PP-OCRv5_det) -- cari lokasi teks
    2. Orientasi (textline_ori)    -- betulkan teks miring
    3. Recognition (SVTR custom)   -- BACA karakter  <-- model fine-tune
      |
      v
  Confidence filtering + ekstraksi (regex tanggal/kode)
      |
      v  JSON {exp_date, kode_produksi, confidence, ...}
[HP Android] -- tampilkan hasil + indikator keyakinan
```

**Catatan penting:** Deteksi pakai model bawaan PaddleOCR, tapi **recognition
(membaca karakter) pakai model hasil fine-tuning sendiri**.

---

## 4. Pipeline Pengembangan Model (Alur Kerja Training)

```
1. KUMPUL DATA
   - Data riil (ICDAR, kemasan)
   - Data sintetik (TRDG: kode/tanggal generated)
   - Data Roboflow (kemasan riil + anotasi manual)
      |
      v
2. PREPROCESSING & LABELING
   - Konversi format ke PaddleOCR (path <TAB> teks)
   - Anotasi crop manual (GUI labeling tool)
   - Validasi karakter vs dictionary
      |
      v
3. PEMBERSIHAN DATASET (anti data-leakage)
   - Deteksi leakage train<->val (dulu 60%!)
   - Split ulang per-label unik (leakage 0%)
   - Seimbangkan komposisi domain
      |
      v
4. TRAINING (Kaggle GPU)
   - Fine-tune SVTR_LCNet
   - Optimizer Adam, Cosine LR
   - Checkpoint tiap 10 epoch
      |
      v
5. EXPORT & EVALUASI
   - Export ke inference format
   - Hitung CER, WER, F1, confusion matrix
      |
      v
6. DEPLOY
   - Upload model ke HuggingFace
   - API live, APK pakai URL cloud
```

---

## 5. Progress Kerja (Milestone)

| Tahap | Status | Catatan |
|---|---|---|
| Setup backend Flask + API | SELESAI | Endpoint /scan, /health |
| Deploy ke cloud (HuggingFace) | SELESAI | Live 24/7, tanpa perlu PC nyala |
| Integrasi Flutter <-> API | SELESAI | URL cloud, tanpa setup WiFi |
| Training model v1 | SELESAI | Ada masalah: data leakage 60% |
| Temukan & perbaiki leakage | SELESAI | Split bersih, leakage 0% |
| Tambah data sintetik (TRDG) | SELESAI | 4000 kode/tanggal unik |
| Tambah data Roboflow + labeling | SELESAI | 378 crop riil dianotasi |
| Training model v2 (bersih) | SELESAI | CER ~20%, acc ~42% (jujur) |
| Dataset v3 (seimbang) | SELESAI | domain 33%->56% |
| Training model v3 (200 epoch) | PROSES | sedang di Kaggle |
| Analisis Metode Numerik | PROSES | korelasi untuk laporan |

---

## 6. Pelajaran Penting dari Proyek

1. **Konsistensi versi krusial.** Model dilatih PaddleOCR 2.7, harus
   di-inference dengan versi sama. Versi beda = hasil rusak (output ngaco).

2. **Data leakage = metrik palsu.** Model v1 tampak akurasi 91% tapi hasil
   nyata jelek — karena 60% data validasi "bocor" dari training (model
   menghafal, bukan belajar). Setelah dibersihkan, angka jujur ~42-66%.

3. **Dictionary harus sinkron.** Model OCR mengeluarkan index angka, lalu
   dipetakan ke karakter via dictionary. Kalau urutan dict beda 1 saja,
   semua huruf salah meski model benar.

4. **Sintetik + riil saling melengkapi.** Sintetik untuk variasi pola,
   riil untuk generalisasi kondisi nyata (glare, dot-matrix).

5. **Komposisi data menentukan fokus model.** Terlalu banyak teks umum
   (67%) membuat model "terdistraksi" dari target (kode/tanggal).

---

## 7. Struktur Folder Proyek

```
SVTR-Project/
├── Src/
│   ├── API/app.py              # Flask API (endpoint scan, health)
│   ├── Model/
│   │   ├── SVTRmodel.py        # Wrapper PaddleOCR
│   │   └── Inference/          # Model hasil training (pdmodel, dict)
│   ├── Preprocessing/
│   │   └── enhancedpicture.py  # CLAHE, gamma, denoise, resize
│   └── Tools/                  # Script dataset & analisis
│       ├── analyze_leakage.py        # deteksi data leakage
│       ├── build_balanced_dataset.py # split seimbang anti-leakage
│       ├── generate_synthetic.py     # generate data TRDG
│       ├── eval_cer_wer.py           # evaluasi CER/WER
│       ├── numerical_analysis.py     # analisis metode numerik
│       └── label_gui.py              # tool labeling interaktif
├── Data/
│   ├── Annotation/             # file label (train_v3.txt, dll)
│   └── Training/               # gambar dataset
├── Dockerfile                  # config deploy HuggingFace
├── requirements.txt            # dependency Python
└── Docs/                       # dokumentasi (file ini)
```

---

*Bahan belajar proyek SVTR. Disusun berdasarkan kondisi nyata pengembangan.*
