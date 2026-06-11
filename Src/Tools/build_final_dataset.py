# -*- coding: utf-8 -*-
"""
Gabungkan SEMUA sumber data recognition jadi satu split final tanpa leakage:
  1. Data lama bersih (train_clean + val_clean) - teks umum + kode/tanggal riil
  2. Data sintetik TRDG (synthetic_labels.txt) - kode/tanggal unik
  3. Roboflow crop riil (rec_roboflow_final.txt) - kondisi kemasan nyata

Prinsip anti-leakage:
  - Split berdasarkan LABEL UNIK (semua gambar berlabel sama -> satu sisi)
  - Cap augmentasi per label
  - Val berisi string yang TIDAK ada di train

Output:
  Data/Annotation/train_v2.txt
  Data/Annotation/val_v2.txt

Jalankan: python Src/Tools/build_final_dataset.py
"""
import os
import random
import collections

random.seed(42)

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
ANN = os.path.join(PROJ, "Data", "Annotation")

SOURCES = [
    os.path.join(ANN, "train_clean.txt"),
    os.path.join(ANN, "val_clean.txt"),
    os.path.join(ANN, "synthetic_labels.txt"),
    os.path.join(ANN, "rec_roboflow_final.txt"),
]

OUT_TRAIN = os.path.join(ANN, "train_v2.txt")
OUT_VAL = os.path.join(ANN, "val_v2.txt")

VAL_RATIO = 0.12
MAX_PER_LABEL = 20
MIN_LEN = 2   # buang label <2 char


def parse(path):
    rows = []
    if not os.path.exists(path):
        print(f"  (lewati, tidak ada: {os.path.basename(path)})")
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            p = line.split("\t")
            if len(p) >= 2 and len(p[1].strip()) >= MIN_LEN:
                rows.append((p[0], p[1]))
    return rows


def main():
    allrows = []
    print("Membaca sumber:")
    for src in SOURCES:
        r = parse(src)
        print(f"  {os.path.basename(src)}: {len(r)} sampel")
        allrows += r
    print(f"\nTotal gabungan: {len(allrows)}")

    # group by label
    by_label = collections.defaultdict(list)
    for img, label in allrows:
        by_label[label].append(img)

    labels = list(by_label.keys())
    random.shuffle(labels)

    # cap augmentasi
    for label in labels:
        if len(by_label[label]) > MAX_PER_LABEL:
            by_label[label] = random.sample(by_label[label], MAX_PER_LABEL)

    # split by label
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

    overlap = set(l for _, l in train_rows) & set(l for _, l in val_rows)
    print("\n=== HASIL ===")
    print(f"Train: {len(train_rows)} sampel, {len(train_labels)} label unik")
    print(f"Val  : {len(val_rows)} sampel, {len(val_labels)} label unik")
    print(f"Leakage (overlap label): {len(overlap)} (harus 0)")
    print(f"\n  {OUT_TRAIN}")
    print(f"  {OUT_VAL}")


if __name__ == "__main__":
    main()
