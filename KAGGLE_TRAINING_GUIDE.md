# Panduan Training Recognition v2 di Kaggle

Dataset v2 sudah bersih (anti-leakage, +sintetik +roboflow). Ikuti urutan ini.

## A. Siapkan data untuk upload ke Kaggle

Yang perlu di-upload sebagai **Kaggle Dataset** (buat dataset baru atau update yang lama):

1. **Gambar** (folder, bisa di-zip):
   - `Data/Training/Dataset_train_1/`
   - `Data/Training/train_data_real+icdar/`
   - `Data/Training/train_data_Real+Sintetik/`
   - `Data/Training/train_data_27_2_2025_1024/`
   - `Data/Training/train_data ICDAR/`
   - `Data/Training/synthetic/`         (4000 sintetik baru)
   - `Data/Training/roboflow_crops/`    (378 crop riil baru)

2. **Label & config**:
   - `Data/Annotation/train_v2.txt`
   - `Data/Annotation/val_v2.txt`
   - `Src/Model/Inference/custom_dict.txt`
   - `Src/Tools/svtr_finetune_v2.yml`

> Catatan path: di train_v2.txt path berbentuk `Data/Training/...`.
> Pastikan struktur folder di Kaggle dataset mempertahankan prefix itu,
> ATAU sesuaikan PROJECT_PATH di cell di bawah agar menunjuk ke root
> tempat folder `Data/` berada.

## B. Cell-cell Kaggle (urut)

### Cell 1 — Install (sama seperti training lama)
```python
!pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q
!pip install "paddleocr==2.7.3" -q
!pip install lmdb -q
print("Install selesai")
```

### Cell 2 — Clone PaddleOCR release/2.7
```python
import os
if not os.path.exists('/kaggle/working/PaddleOCR'):
    os.system('git clone https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR -q')
    os.system('git -C /kaggle/working/PaddleOCR checkout release/2.7 -q')
print("train.py ada:", os.path.exists('/kaggle/working/PaddleOCR/tools/train.py'))
```

### Cell 3 — Siapkan config (isi placeholder)
```python
import re

# SESUAIKAN dengan lokasi dataset Kaggle kamu:
DATASET_ROOT = '/kaggle/input/NAMA-DATASET-KAMU'   # root tempat folder Data/ berada
DICT  = f'{DATASET_ROOT}/custom_dict.txt'          # atau path dict
YML_SRC = f'{DATASET_ROOT}/svtr_finetune_v2.yml'
OUTPUT_DIR = '/kaggle/working/output/PP-OCRv5_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Resume otomatis jika ada checkpoint sebelumnya
ckpt = 'null'
latest = f'{OUTPUT_DIR}/latest.pdparams'
if os.path.exists(latest):
    ckpt = f'{OUTPUT_DIR}/latest'

with open(YML_SRC, encoding='utf-8') as f:
    cfg = f.read()
cfg = cfg.replace('USE_GPU_PLACEHOLDER', 'true')
cfg = cfg.replace('DICT_PATH_PLACEHOLDER', DICT)
cfg = cfg.replace('CHECKPOINT_PLACEHOLDER', ckpt)
# PROJECT_PATH = root tempat folder Data/ + file train_v2.txt berada
cfg = cfg.replace('PROJECT_PATH', DATASET_ROOT)
cfg = re.sub(r'save_model_dir:.*', f'save_model_dir: {OUTPUT_DIR}/', cfg)
cfg = re.sub(r'save_res_path:.*', f'save_res_path: {OUTPUT_DIR}/predicts.txt', cfg)

with open('/kaggle/working/train_v2.yml', 'w', encoding='utf-8') as f:
    f.write(cfg)
print("Config siap. Cek path train/val:")
os.system("grep -E 'label_file_list|character_dict|checkpoints|use_gpu' -A1 /kaggle/working/train_v2.yml")
```

### Cell 4 — Training
```python
import subprocess, sys, os
os.chdir('/kaggle/working/PaddleOCR')
p = subprocess.Popen([sys.executable, 'tools/train.py', '-c', '/kaggle/working/train_v2.yml'],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end='')
p.wait()
print("Exit:", p.returncode)
```

### Cell 5 — Export model (setelah training selesai)
```python
import os
os.chdir('/kaggle/working/PaddleOCR')
BEST = f'{OUTPUT_DIR}/best_accuracy'
if not os.path.exists(BEST + '.pdparams'):
    BEST = f'{OUTPUT_DIR}/latest'
!python tools/export_model.py \
  -c /kaggle/working/train_v2.yml \
  -o Global.pretrained_model={BEST} \
  -o Global.save_inference_dir=/kaggle/working/inference_v2
print("Cek hasil:")
os.system("ls -la /kaggle/working/inference_v2")
```

### Cell 6 — Zip hasil untuk download
```python
import shutil
shutil.make_archive('/kaggle/working/inference_v2', 'zip', '/kaggle/working/inference_v2')
print("Download: /kaggle/working/inference_v2.zip")
```

## C. Setelah download
1. Extract `inference_v2.zip`
2. Kirim ke saya / taruh di project -> saya pasang ke `Src/Model/Inference/`
3. Test lokal (eval_cer_wer.py) -> kalau bagus, deploy ke HuggingFace

## Target
- CER < 15% (v1 sudah 13%, harusnya membaik dengan data bersih)
- WER turun dari 39%
- Akurasi naik dari 61%

## Tips
- GPU Kaggle: training 200 epoch ~ 2-4 jam tergantung ukuran data
- Kaggle session maks 12 jam, simpan checkpoint berkala (save_epoch_step=10)
- Kalau session putus, jalankan ulang Cell 3-4 (auto-resume dari latest)
