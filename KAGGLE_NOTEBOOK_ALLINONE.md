# Notebook Kaggle ALL-IN-ONE — Training v3 200 Epoch (1 sesi)

Fresh training 200 epoch sekaligus + export + semua visualisasi (CER/WER,
confusion matrix, F1, grafik). Satu notebook, satu dataset, tanpa
sambung-menyambung checkpoint.

## Soal file yang sudah kamu download (iter_100, iter_140, latest, train.log)
**TIDAK perlu dipakai** untuk pendekatan fresh-200 ini. Simpan sebagai backup
saja. Kalau nanti mau hemat waktu (resume 140->200), baru dipakai — tapi
sesuai keputusanmu kita fresh 200 biar log & grafik kontinu 0-200 otomatis.

## Soal keep-alive & cooldown
- **Keep-alive TIDAK diperlukan** kalau pakai "Save & Run All (Commit)" —
  notebook jalan di server Kaggle, bukan browser. Popup "are you still here?"
  TIDAK akan muncul.
- **Cooldown/quota**: Kaggle kasih ~30 jam GPU/minggu per akun. Akun baru =
  kuota fresh. 200 epoch (~6-7 jam) aman di bawah limit 12 jam/sesi.

## PERSIAPAN
- Dataset `svtr-dataset-v3` sudah ter-upload (Add Input).
- Aktifkan **GPU** (Settings -> Accelerator -> GPU T4 x2 / P100).
- Setelah semua cell dimasukkan: **Save Version -> Save & Run All (Commit)**,
  lalu TUTUP TAB. Jangan Run All biasa.

---

## CELL 1 — Install (urutan penting, TANPA perlu restart di mode Commit)
```python
# Install paddle + paddleocr, lalu PIN numpy 1.26.4 PALING AKHIR.
# Di mode Commit (kernel fresh), tidak perlu restart karena belum ada
# cell yang meng-import numpy/paddle sebelum ini.
!pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html -q
!pip install "paddleocr==2.7.3" lmdb -q
!pip install "opencv-python-headless==4.6.0.66" -q
!pip install "numpy==1.26.4" -q --force-reinstall
print("Install selesai. (Mode Commit: TIDAK perlu restart kernel.)")
```
> CATATAN: Jangan tambahkan import numpy/paddle/cv2 di Cell 1.
> Import pertama harus di Cell 2 dst, supaya numpy 1.26.4 yang terpakai.
> Di mode Commit kernel selalu fresh, jadi urutan ini aman tanpa restart.

## CELL 2 — Clone PaddleOCR release/2.7
```python
import os
if not os.path.exists('/kaggle/working/PaddleOCR'):
    os.system('git clone https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR -q')
    os.system('git -C /kaggle/working/PaddleOCR checkout release/2.7 -q')
print("train.py ada:", os.path.exists('/kaggle/working/PaddleOCR/tools/train.py'))
```

## CELL 3 — Config (fresh 200 epoch)
```python
import os, re

# GANTI sesuai nama dataset kamu (cek di panel Input):
DATASET_ROOT = '/kaggle/input/datasets/ainurrizza/svtr-dataset-v3/svtr_kaggle_upload'

DICT       = f'{DATASET_ROOT}/custom_dict.txt'
YML_SRC    = f'{DATASET_ROOT}/svtr_finetune_v2.yml'
OUTPUT_DIR = '/kaggle/working/output/PP-OCRv5_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# cek file penting
for f in [DICT, YML_SRC,
          f'{DATASET_ROOT}/Data/Annotation/train_v3.txt',
          f'{DATASET_ROOT}/Data/Annotation/val_v3.txt']:
    print(("OK  " if os.path.exists(f) else "HILANG ") + f)

with open(YML_SRC, encoding='utf-8') as f:
    cfg = f.read()
cfg = cfg.replace('USE_GPU_PLACEHOLDER', 'true')
cfg = cfg.replace('DICT_PATH_PLACEHOLDER', DICT)
cfg = cfg.replace('CHECKPOINT_PLACEHOLDER', 'null')   # fresh dari 0
cfg = cfg.replace('PROJECT_PATH', DATASET_ROOT)
cfg = re.sub(r'epoch_num:\s*\d+', 'epoch_num: 200', cfg)
cfg = re.sub(r'save_model_dir:.*', f'save_model_dir: {OUTPUT_DIR}/', cfg)
cfg = re.sub(r'save_res_path:.*', f'save_res_path: {OUTPUT_DIR}/predicts.txt', cfg)
cfg = cfg.replace(f'{DATASET_ROOT}/train_v3.txt', f'{DATASET_ROOT}/Data/Annotation/train_v3.txt')
cfg = cfg.replace(f'{DATASET_ROOT}/val_v3.txt',   f'{DATASET_ROOT}/Data/Annotation/val_v3.txt')

with open('/kaggle/working/train_v3.yml', 'w', encoding='utf-8') as f:
    f.write(cfg)
print("\nConfig siap (200 epoch, fresh).")
os.system("grep -E 'epoch_num|checkpoints|character_dict|use_gpu' /kaggle/working/train_v3.yml")
```

## CELL 4 — Training 200 epoch (log tersimpan kontinu)
```python
import subprocess, sys, os
os.chdir('/kaggle/working/PaddleOCR')
p = subprocess.Popen([sys.executable, 'tools/train.py', '-c', '/kaggle/working/train_v3.yml'],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open('/kaggle/working/train_log.txt', 'a') as lf:
    for line in p.stdout:
        print(line, end='')
        lf.write(line)
p.wait()
print("Exit:", p.returncode)
```

## CELL 5 — Export model
```python
import os
os.chdir('/kaggle/working/PaddleOCR')
CKPT = f'{OUTPUT_DIR}/best_accuracy'
if not os.path.exists(CKPT + '.pdparams'):
    CKPT = f'{OUTPUT_DIR}/latest'
print("Export dari:", CKPT, "| ada?", os.path.exists(CKPT + '.pdparams'))
!python tools/export_model.py -c /kaggle/working/train_v3.yml \
  -o Global.pretrained_model={CKPT} \
     Global.save_inference_dir=/kaggle/working/inference_v3
# PASTIKAN log muncul "load pretrain successful", BUKAN "train from scratch"
os.system("ls -la /kaggle/working/inference_v3")
```

## CELL 6 — Prediksi val set (paddle inference langsung, anti error skimage)
```python
import os, cv2, numpy as np
import paddle.inference as pi

INFER='/kaggle/working/inference_v3'
chars=[l for l in open(DICT,'rb').read().decode('utf-8').split('\n') if l!='']
cfg=pi.Config(f'{INFER}/inference.pdmodel', f'{INFER}/inference.pdiparams')
cfg.disable_gpu(); cfg.enable_memory_optim()
pred=pi.create_predictor(cfg)
ih=pred.get_input_handle(pred.get_input_names()[0])
oh=pred.get_output_handle(pred.get_output_names()[0])

def prep(img):
    h,w=img.shape[:2]; tw=min(320,max(1,int(48*w/h)))
    r=cv2.resize(img,(tw,48)); c=np.zeros((48,320,3),np.float32); c[:,:tw,:]=r
    return ((c.transpose(2,0,1)/255.-0.5)/0.5)[None].astype('float32')

def decode(idx):
    out=[];prev=-1
    for i in idx:
        if i!=prev and i!=0 and 1<=i<=len(chars): out.append(chars[i-1])
        prev=i
    return ''.join(out)

val=f'{DATASET_ROOT}/Data/Annotation/val_v3.txt'
gts,paths=[],[]
for l in open(val,encoding='utf-8'):
    if '\t' in l:
        p,t=l.strip().split('\t',1); fp=os.path.join(DATASET_ROOT,p)
        if os.path.exists(fp): paths.append(fp); gts.append(t)
imgs=[cv2.imread(p) for p in paths]
pairs=[(im,g) for im,g in zip(imgs,gts) if im is not None]
imgs=[x[0] for x in pairs]; gts=[x[1] for x in pairs]
preds=[]
for im in imgs:
    ih.copy_from_cpu(prep(im)); pred.run()
    preds.append(decode(oh.copy_to_cpu()[0].argmax(1)))
print("Dievaluasi:", len(preds))
print("Contoh:", list(zip(gts[:5],preds[:5])))
```

## CELL 7 — CER / WER / Accuracy
```python
def ed(a,b):
    m,n=len(a),len(b);dp=list(range(n+1))
    for i in range(1,m+1):
        pv=dp[0];dp[0]=i
        for j in range(1,n+1):
            t=dp[j];dp[j]=min(dp[j]+1,dp[j-1]+1,pv+(0 if a[i-1]==b[j-1] else 1));pv=t
    return dp[n]
tc=tce=ex=0
for g,p in zip(gts,preds):
    tc+=len(g); tce+=ed(p,g); ex+= (p.strip()==g.strip())
print(f"CER : {100*tce/max(1,tc):.2f}%")
print(f"WER : {100*(1-ex/max(1,len(gts))):.2f}%")
print(f"Accuracy: {100*ex/max(1,len(gts)):.2f}%")
```

## CELL 8 — Confusion Matrix Alphanumerik
```python
import numpy as np, matplotlib.pyplot as plt
aln=[c for c in chars if c.isalnum()]; idx={c:i for i,c in enumerate(aln)}; N=len(aln)
cm=np.zeros((N,N),int)
for g,p in zip(gts,preds):
    for k in range(min(len(g),len(p))):
        if g[k] in idx and p[k] in idx: cm[idx[g[k]]][idx[p[k]]]+=1
plt.figure(figsize=(16,14)); plt.imshow(cm,cmap='Blues')
plt.xticks(range(N),aln,rotation=90,fontsize=6); plt.yticks(range(N),aln,fontsize=6)
plt.xlabel('Prediksi'); plt.ylabel('Ground Truth'); plt.title('Confusion Matrix Alphanumerik')
plt.colorbar(); plt.tight_layout(); plt.savefig('/kaggle/working/confusion_matrix.png',dpi=150); plt.show()
```

## CELL 9 — F1 per karakter (CSV, tanpa pandas)
```python
import csv, numpy as np
rows=[]
for i,c in enumerate(aln):
    tp=cm[i][i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
    pr=tp/(tp+fp) if tp+fp>0 else 0; rc=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0
    rows.append([c,round(pr,3),round(rc,3),round(f1,3),int(cm[i,:].sum())])
print("Macro F1:",round(np.mean([r[3] for r in rows]),3))
with open('/kaggle/working/per_char_metrics.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['char','precision','recall','f1','support']); w.writerows(rows)
print("per_char_metrics.csv disimpan")
```

## CELL 10 — Grafik kurva training (KONTINU 0-200)
```python
import re, matplotlib.pyplot as plt
log=open('/kaggle/working/train_log.txt',encoding='utf-8',errors='ignore').read()
by={}
for m in re.finditer(r'epoch:\s*\[(\d+)/\d+\][^\n]*?loss:\s*([\d.]+)',log):
    e=int(m.group(1)); by.setdefault(e,[]).append(float(m.group(2)))
eps=sorted(by); loss=[sum(by[e])/len(by[e]) for e in eps]
plt.figure(figsize=(11,5)); plt.plot(eps,loss,color='steelblue')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Kurva Loss Training v3 (0-200, kontinu)')
plt.grid(alpha=0.3); plt.savefig('/kaggle/working/training_curve.png',dpi=150); plt.show()
print(f"epoch tercatat: {eps[0]}-{eps[-1]}")
```

## CELL 11 — Zip semua hasil
```python
import shutil, os
os.makedirs('/kaggle/working/hasil',exist_ok=True)
for f in ['confusion_matrix.png','per_char_metrics.csv','training_curve.png','train_log.txt']:
    s='/kaggle/working/'+f
    if os.path.exists(s): shutil.copy(s,'/kaggle/working/hasil/')
shutil.make_archive('/kaggle/working/inference_v3','zip','/kaggle/working/inference_v3')
shutil.make_archive('/kaggle/working/hasil','zip','/kaggle/working/hasil')
print("Download: inference_v3.zip (model) + hasil.zip (grafik/metrik)")
```

---

## URUTAN FINAL
1. Masukkan semua cell di atas
2. Jalankan Cell 1 -> Restart Kernel
3. **Save Version -> Save & Run All (Commit)** -> tutup tab
4. ~6-7 jam kemudian: buka Output, download inference_v3.zip + hasil.zip
5. Grafik di hasil.zip = kontinu 0-200 (karena 1 sesi, tidak terputus)
