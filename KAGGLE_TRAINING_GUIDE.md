# Panduan Lengkap Training Recognition v2 di Kaggle (dari nol)

Dataset v2 sudah bersih (anti-leakage) + sintetik + roboflow.
Total gambar ~252MB. Ikuti urutan ini.

---

## BAGIAN 0: Siapkan & Upload Dataset ke Kaggle

### Struktur upload (PENTING - path harus persis)
Buat satu folder lokal, misal `svtr_kaggle_upload/`, dengan isi:

```
svtr_kaggle_upload/
├── Data/
│   ├── Training/
│   │   ├── Dataset_train_1/
│   │   ├── train_data ICDAR/
│   │   ├── train_data_27_2_2025_1024/
│   │   ├── train_data_real+icdar/
│   │   ├── train_data_Real+Sintetik/
│   │   ├── synthetic/            (baru)
│   │   └── roboflow_crops/       (baru)
│   └── Annotation/
│       ├── train_v2.txt
│       └── val_v2.txt
├── custom_dict.txt
└── svtr_finetune_v2.yml
```

> Cara cepat: zip folder `Data/` dari project + tambahkan custom_dict.txt
> dan svtr_finetune_v2.yml. Upload zip ke Kaggle (auto-extract).

### Langkah upload
1. kaggle.com → **Datasets** → **New Dataset**
2. Drag folder/zip di atas
3. Beri nama, mis. `svtr-dataset-v2`
4. Create. Catat path-nya: `/kaggle/input/svtr-dataset-v2`

### Aktifkan GPU
Notebook → Settings (kanan) → Accelerator → **GPU T4 x2** atau **P100**

---

## BAGIAN 1: Cell-cell Training

### Cell 1 — Install
```python
!pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q
!pip install "paddleocr==2.7.3" lmdb -q
print("Install selesai")
```

### Cell 2 — Clone PaddleOCR release/2.7
```python
import os
if not os.path.exists('/kaggle/working/PaddleOCR'):
    os.system('git clone https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR -q')
    os.system('git -C /kaggle/working/PaddleOCR checkout release/2.7 -q')
print("train.py:", os.path.exists('/kaggle/working/PaddleOCR/tools/train.py'))
```

### Cell 3 — Siapkan config
```python
import os, re

# ====== SESUAIKAN NAMA DATASET KAMU ======
DATASET_ROOT = '/kaggle/input/svtr-dataset-v2'
# =========================================

DICT       = f'{DATASET_ROOT}/custom_dict.txt'
YML_SRC    = f'{DATASET_ROOT}/svtr_finetune_v2.yml'
OUTPUT_DIR = '/kaggle/working/output/PP-OCRv5_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# cek file penting ada
for f in [DICT, YML_SRC,
          f'{DATASET_ROOT}/Data/Annotation/train_v2.txt',
          f'{DATASET_ROOT}/Data/Annotation/val_v2.txt']:
    print(("OK  " if os.path.exists(f) else "HILANG ") + f)

# auto-resume kalau ada checkpoint
ckpt = 'null'
if os.path.exists(f'{OUTPUT_DIR}/latest.pdparams'):
    ckpt = f'{OUTPUT_DIR}/latest'

with open(YML_SRC, encoding='utf-8') as f:
    cfg = f.read()
cfg = cfg.replace('USE_GPU_PLACEHOLDER', 'true')
cfg = cfg.replace('DICT_PATH_PLACEHOLDER', DICT)
cfg = cfg.replace('CHECKPOINT_PLACEHOLDER', ckpt)
cfg = cfg.replace('PROJECT_PATH', DATASET_ROOT)
cfg = re.sub(r'save_model_dir:.*', f'save_model_dir: {OUTPUT_DIR}/', cfg)
cfg = re.sub(r'save_res_path:.*', f'save_res_path: {OUTPUT_DIR}/predicts.txt', cfg)

# Karena label_file_list di yml = PROJECT_PATH/train_v2.txt, tapi file ada di
# PROJECT_PATH/Data/Annotation/ -> betulkan:
cfg = cfg.replace(f'{DATASET_ROOT}/train_v2.txt', f'{DATASET_ROOT}/Data/Annotation/train_v2.txt')
cfg = cfg.replace(f'{DATASET_ROOT}/val_v2.txt',   f'{DATASET_ROOT}/Data/Annotation/val_v2.txt')

with open('/kaggle/working/train_v2.yml', 'w', encoding='utf-8') as f:
    f.write(cfg)
print("\nConfig siap.")
os.system("grep -E 'label_file_list|character_dict|use_gpu|epoch_num' -A1 /kaggle/working/train_v2.yml")
```

### Cell 4 — Training (2-4 jam)
```python
import subprocess, sys, os
os.chdir('/kaggle/working/PaddleOCR')
p = subprocess.Popen([sys.executable, 'tools/train.py', '-c', '/kaggle/working/train_v2.yml'],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
log = '/kaggle/working/train_log.txt'
with open(log, 'a') as lf:
    for line in p.stdout:
        print(line, end='')
        lf.write(line)
p.wait()
print("Exit:", p.returncode)
```

### Cell 5 — Export model
```python
import os
os.chdir('/kaggle/working/PaddleOCR')
BEST = f'{OUTPUT_DIR}/best_accuracy'
if not os.path.exists(BEST + '.pdparams'):
    BEST = f'{OUTPUT_DIR}/latest'
print("Export dari:", BEST)
!python tools/export_model.py -c /kaggle/working/train_v2.yml \
  -o Global.pretrained_model={BEST} \
  -o Global.save_inference_dir=/kaggle/working/inference_v2
os.system("ls -la /kaggle/working/inference_v2")
```

---

## BAGIAN 2: Evaluasi & Visualisasi (untuk laporan/paper)

### Cell 6 — Prediksi val set (kumpulkan GT vs prediksi)
```python
import os, sys
os.chdir('/kaggle/working/PaddleOCR')
sys.path.insert(0, '/kaggle/working/PaddleOCR')
import cv2
from tools.infer.predict_rec import TextRecognizer
import tools.infer.utility as utility

args = utility.parse_args()
args.rec_model_dir = '/kaggle/working/inference_v2'
args.rec_char_dict_path = DICT
args.rec_image_shape = "3, 48, 320"
args.use_space_char = True
args.use_gpu = True

rec = TextRecognizer(args)

# baca val_v2
val_path = f'{DATASET_ROOT}/Data/Annotation/val_v2.txt'
gts, paths = [], []
with open(val_path, encoding='utf-8') as f:
    for line in f:
        line=line.strip()
        if '\t' in line:
            p,t = line.split('\t',1)
            full = os.path.join(DATASET_ROOT, p)
            if os.path.exists(full):
                paths.append(full); gts.append(t)

imgs = [cv2.imread(p) for p in paths]
imgs = [im for im in imgs if im is not None]
preds = []
B=128
for i in range(0,len(imgs),B):
    r,_ = rec(imgs[i:i+B]); preds += [t for t,_ in r]
print("Total dievaluasi:", len(preds))
```

### Cell 7 — CER, WER, Accuracy
```python
def edit_distance(a,b):
    m,n=len(a),len(b); dp=list(range(n+1))
    for i in range(1,m+1):
        prev=dp[0]; dp[0]=i
        for j in range(1,n+1):
            tmp=dp[j]; cost=0 if a[i-1]==b[j-1] else 1
            dp[j]=min(dp[j]+1,dp[j-1]+1,prev+cost); prev=tmp
    return dp[n]

tot_c=tot_ce=exact=tot_w=tot_we=0
for gt,pr in zip(gts,preds):
    tot_c+=len(gt); tot_ce+=edit_distance(pr,gt)
    if pr.strip()==gt.strip(): exact+=1
    gw,pw=gt.split(),pr.split(); tot_w+=len(gw)
    tot_we+=edit_distance(pw,gw)

cer=100*tot_ce/max(1,tot_c)
wer=100*(1-exact/max(1,len(gts)))
acc=100*exact/max(1,len(gts))
print(f"CER : {cer:.2f}%")
print(f"WER : {wer:.2f}%")
print(f"Accuracy (exact match): {acc:.2f}%")
```

### Cell 8 — Confusion Matrix Alphanumerik (level karakter)
```python
import numpy as np, matplotlib.pyplot as plt

chars = [c for c in open(DICT,encoding='utf-8').read().split('\n') if c!='']
# fokus alphanumerik
alnum = [c for c in chars if c.isalnum()]
idx = {c:i for i,c in enumerate(alnum)}
N=len(alnum)
cm = np.zeros((N,N),dtype=int)

# align per posisi (greedy) untuk pasangan char
for gt,pr in zip(gts,preds):
    L=min(len(gt),len(pr))
    for k in range(L):
        g,p=gt[k],pr[k]
        if g in idx and p in idx:
            cm[idx[g]][idx[p]]+=1

plt.figure(figsize=(16,14))
plt.imshow(cm, cmap='Blues')
plt.xticks(range(N), alnum, rotation=90, fontsize=6)
plt.yticks(range(N), alnum, fontsize=6)
plt.xlabel('Prediksi'); plt.ylabel('Ground Truth')
plt.title('Confusion Matrix Alphanumerik (level karakter)')
plt.colorbar()
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()
print("Disimpan: confusion_matrix.png")
```

### Cell 9 — F1 / Precision / Recall per karakter (tabel)
```python
import pandas as pd
rows=[]
for i,c in enumerate(alnum):
    tp=cm[i][i]
    fp=cm[:,i].sum()-tp
    fn=cm[i,:].sum()-tp
    prec=tp/(tp+fp) if tp+fp>0 else 0
    rec_=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*prec*rec_/(prec+rec_) if prec+rec_>0 else 0
    sup=cm[i,:].sum()
    rows.append([c,round(prec,3),round(rec_,3),round(f1,3),int(sup)])
df=pd.DataFrame(rows,columns=['char','precision','recall','f1','support'])
# macro avg
print("Macro F1 :", round(df['f1'].mean(),3))
print("Macro Precision:", round(df['precision'].mean(),3))
print("Macro Recall:", round(df['recall'].mean(),3))
df.to_csv('/kaggle/working/per_char_metrics.csv', index=False)
df.sort_values('f1').head(15)   # 15 karakter paling sulit
```

### Cell 10 — Grafik kurva training (loss & accuracy per epoch)
```python
import re, matplotlib.pyplot as plt

log = open('/kaggle/working/train_log.txt', encoding='utf-8', errors='ignore').read()
# parse loss & acc dari log paddleocr
epochs_loss = [float(x) for x in re.findall(r'loss: ([\d.]+)', log)]
accs = [float(x) for x in re.findall(r'acc: ([\d.]+)', log)]

fig, ax = plt.subplots(1,2, figsize=(14,5))
ax[0].plot(epochs_loss); ax[0].set_title('Training Loss'); ax[0].set_xlabel('step'); ax[0].set_ylabel('loss')
ax[1].plot(accs, color='green'); ax[1].set_title('Eval Accuracy'); ax[1].set_xlabel('eval step'); ax[1].set_ylabel('acc')
plt.tight_layout()
plt.savefig('/kaggle/working/training_curves.png', dpi=150)
plt.show()
print("Disimpan: training_curves.png")
```

### Cell 11 — Zip semua hasil untuk download
```python
import shutil, os
os.makedirs('/kaggle/working/hasil', exist_ok=True)
for f in ['confusion_matrix.png','per_char_metrics.csv','training_curves.png','train_log.txt']:
    src='/kaggle/working/'+f
    if os.path.exists(src): shutil.copy(src,'/kaggle/working/hasil/')
shutil.make_archive('/kaggle/working/inference_v2','zip','/kaggle/working/inference_v2')
shutil.make_archive('/kaggle/working/hasil','zip','/kaggle/working/hasil')
print("Download:")
print("  /kaggle/working/inference_v2.zip  (model)")
print("  /kaggle/working/hasil.zip         (grafik+tabel+log)")
```

---

## Setelah download
1. Extract `inference_v2.zip` → kirim ke saya untuk dipasang & deploy
2. `hasil.zip` = bahan laporan (confusion matrix, CER/WER, F1, grafik)

## Target
- CER < 15%, WER turun dari 39%, akurasi naik dari 61%

## Catatan
- Training 200 epoch ~2-4 jam (GPU). Session Kaggle maks 12 jam.
- Kalau putus, jalankan ulang Cell 3-4 (auto-resume dari latest checkpoint).
- save_epoch_step=10 → checkpoint tersimpan tiap 10 epoch.


---

# BAGIAN 3: Resume Training 100 -> 200 epoch (di device lain)

Checkpoint v2 (epoch 100) sudah tersimpan via "Save Version" di Kaggle.
Tidak perlu download apapun. Ikuti ini di device lain (akun Kaggle sama).

## Setup input
1. Login Kaggle (akun sama)
2. Buat notebook baru, aktifkan **GPU**
3. **Add Input** (panel kanan):
   - Tab "Your Datasets" -> tambahkan `svtr-dataset-v2` (gambar + label)
   - Tab "Notebook Output" -> tambahkan notebook training v2 yang sudah di-Save
     (yang isinya folder output/PP-OCRv5_v2/)

## Cell A - Cek path checkpoint
```python
import os
for root, dirs, files in os.walk('/kaggle/input'):
    if any(f.startswith('iter_epoch_100') for f in files):
        print("CHECKPOINT DITEMUKAN DI:", root)
        print("  isi:", [f for f in files if f.startswith(('iter_epoch_100','best_accuracy','latest'))])
```
Catat path yang muncul -> pakai sebagai CKPT_SRC di Cell C.

## Cell 1 & 2 - sama seperti Bagian 1 (install + clone)
```python
!pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q
!pip install "paddleocr==2.7.3" lmdb -q
!pip install "numpy==1.26.4" "opencv-python-headless==4.6.0.66" -q
print("RESTART KERNEL setelah cell ini, lalu lanjut")
```
(Restart kernel, lalu)
```python
import os
if not os.path.exists('/kaggle/working/PaddleOCR'):
    os.system('git clone https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR -q')
    os.system('git -C /kaggle/working/PaddleOCR checkout release/2.7 -q')
print("OK")
```

## Cell C - Config resume (epoch 100 -> 200)
```python
import os, re, shutil

DATASET_ROOT = '/kaggle/input/datasets/erpandeeplearning/svtr-dataset-v2/svtr_kaggle_upload'
# GANTI dengan path dari Cell A:
CKPT_SRC = '/kaggle/input/NAMA-NOTEBOOK-OUTPUT/output/PP-OCRv5_v2'

DICT = f'{DATASET_ROOT}/custom_dict.txt'
YML_SRC = f'{DATASET_ROOT}/svtr_finetune_v2.yml'
OUTPUT_DIR = '/kaggle/working/output/PP-OCRv5_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# salin checkpoint ke working dir (supaya bisa di-resume + ditimpa)
for f in os.listdir(CKPT_SRC):
    if f.startswith(('latest','best_accuracy','iter_epoch_100')):
        shutil.copy(os.path.join(CKPT_SRC, f), os.path.join(OUTPUT_DIR, f))
print("checkpoint disalin:", sorted(os.listdir(OUTPUT_DIR)))

# resume dari iter_epoch_100 (atau latest)
resume = f'{OUTPUT_DIR}/iter_epoch_100'
if not os.path.exists(resume + '.pdparams'):
    resume = f'{OUTPUT_DIR}/latest'

with open(YML_SRC, encoding='utf-8') as f:
    cfg = f.read()
cfg = cfg.replace('USE_GPU_PLACEHOLDER', 'true')
cfg = cfg.replace('DICT_PATH_PLACEHOLDER', DICT)
cfg = cfg.replace('CHECKPOINT_PLACEHOLDER', resume)
cfg = cfg.replace('PROJECT_PATH', DATASET_ROOT)
cfg = re.sub(r'epoch_num:\s*\d+', 'epoch_num: 200', cfg)
cfg = re.sub(r'save_model_dir:.*', f'save_model_dir: {OUTPUT_DIR}/', cfg)
cfg = re.sub(r'save_res_path:.*', f'save_res_path: {OUTPUT_DIR}/predicts.txt', cfg)
cfg = cfg.replace(f'{DATASET_ROOT}/train_v2.txt', f'{DATASET_ROOT}/Data/Annotation/train_v2.txt')
cfg = cfg.replace(f'{DATASET_ROOT}/val_v2.txt',   f'{DATASET_ROOT}/Data/Annotation/val_v2.txt')

with open('/kaggle/working/train_v2.yml','w',encoding='utf-8') as f:
    f.write(cfg)
print("resume dari:", resume, "-> target 200")
os.system("grep -E 'checkpoints|epoch_num' /kaggle/working/train_v2.yml")
```

## Cell D - Training (lanjut 100->200) + Cell export/eval
Sama seperti Cell 4-11 di Bagian 1 & 2.

## PENTING
- Sebelum tinggalkan: klik "Save Version" -> "Save & Run All (Commit)"
  supaya jalan di background, device boleh dimatikan.
- Setelah 200 epoch: download inference_v2.zip + hasil.zip + checkpoint baru.
