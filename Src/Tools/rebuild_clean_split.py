# -*- coding: utf-8 -*-
"""
Bangun ulang train/val split yang BERSIH tanpa data leakage.

Prinsip:
- Split dilakukan berdasarkan LABEL UNIK (bukan per-gambar).
- Semua gambar/augmentasi dengan label sama masuk ke SATU sisi saja
  (semua ke train ATAU semua ke val), tidak menyeberang.
- Dengan begitu, val set berisi string yang BENAR-BENAR tidak dilihat
  model saat training -> metrik jujur.

Opsional:
- Batasi jumlah augmentasi maksimal per label (cap) agar tidak ada
  string yang mendominasi (mis. 'BB 17072025 OTAK-OTAK' 573x).

Output:
- Data/Annotation/train_clean.txt
- Data/Annotation/val_clean.txt

Tidak menimpa file lama. Jalankan:
    python Src/Tools/rebuild_clean_split.py
"""
import os
import random
import collections

random.seed(42)

ANN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "Annotation")
TRAIN = os.path.join(ANN_DIR, "train_final.txt")
VAL = os.path.join(ANN_DIR, "val_final.txt")
OUT_TRAIN = os.path.join(ANN_DIR, "train_clean.txt")
OUT_VAL = os.path.join(ANN_DIR, "val_clean.txt")

VAL_RATIO = 0.15      # 15% label unik untuk validasi
MAX_PER_LABEL = 20    # cap augmentasi per label (kurangi dominasi)


def parse(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split("\t")
            if len(p) >= 2:
                rows.append((p[0], p[1]))
    return rows


def main():
    # Gabung semua data dulu (train_final + val_final) lalu split ulang bersih
    allrows = parse(TRAIN) + parse(VAL)
    print(f"Total gabungan: {len(allrows)} sampel")

    # Group berdasarkan label
    by_label = collections.defaultdict(list)
    for img, label in allrows:
        by_label[label].append(img)

    labels = list(by_label.keys())
    random.shuffle(labels)
    print(f"Label unik: {len(labels)}")

    # Cap augmentasi per label
    capped = 0
    for label in labels:
        imgs = by_label[label]
        if len(imgs) > MAX_PER_LABEL:
            by_label[label] = random.sample(imgs, MAX_PER_LABEL)
            capped += 1
    print(f"Label yang di-cap ke {MAX_PER_LABEL}: {capped}")

    # Split berdasarkan label (bukan gambar)
    n_val = int(len(labels) * VAL_RATIO)
    val_labels = set(labels[:n_val])
    train_labels = set(labels[n_val:])

    train_rows, val_rows = [], []
    for label in train_labels:
        for img in by_label[label]:
            train_rows.append((img, label))
    for label in val_labels:
        for img in by_label[label]:
            val_rows.append((img, label))

    random.shuffle(train_rows)
    random.shuffle(val_rows)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for img, label in train_rows:
            f.write(f"{img}\t{label}\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for img, label in val_rows:
            f.write(f"{img}\t{label}\n")

    # Verifikasi: tidak ada leakage
    overlap = set(l for _, l in train_rows) & set(l for _, l in val_rows)

    print("\n=== HASIL SPLIT BERSIH ===")
    print(f"Train: {len(train_rows)} sampel, {len(train_labels)} label unik")
    print(f"Val  : {len(val_rows)} sampel, {len(val_labels)} label unik")
    print(f"Label overlap train<->val: {len(overlap)} (harus 0)")
    print(f"\nFile ditulis:")
    print(f"  {OUT_TRAIN}")
    print(f"  {OUT_VAL}")


if __name__ == "__main__":
    main()
