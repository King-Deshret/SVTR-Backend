# -*- coding: utf-8 -*-
"""
Bangun dataset v3 dengan komposisi SEIMBANG untuk domain kode produksi & tanggal.

Masalah v2: teks umum 67% (tidak relevan), riil cuma 2.7%.
v3 menyeimbangkan supaya model fokus ke domain target:
  - Teks umum (ICDAR dll)  : di-downsample ke ~target
  - Sintetik kode/tanggal  : dipertahankan (relevan domain)
  - Roboflow riil          : semua dipakai (paling berharga)

Tetap anti-leakage: split per-label unik, augmentasi tidak menyeberang.

Output:
  Data/Annotation/train_v3.txt
  Data/Annotation/val_v3.txt

Jalankan: python Src/Tools/build_balanced_dataset.py
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

OUT_TRAIN = os.path.join(ANN, "train_v3.txt")
OUT_VAL = os.path.join(ANN, "val_v3.txt")

VAL_RATIO = 0.12
MAX_PER_LABEL = 15
MIN_LEN = 2

# Target maksimum porsi teks umum (general). Sisanya domain (sintetik+riil).
# Kita batasi teks umum agar tidak mendominasi.
GENERAL_CAP = 3500   # maksimum sampel teks umum yang dipakai (dari ~7700)


def parse(path):
    rows = []
    if not os.path.exists(path):
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


def is_general(path):
    """Teks umum = bukan sintetik & bukan roboflow."""
    return ("synthetic" not in path) and ("roboflow" not in path)


def main():
    allrows = []
    for src in SOURCES:
        allrows += parse(src)

    general = [r for r in allrows if is_general(r[0])]
    domain = [r for r in allrows if not is_general(r[0])]

    print(f"Total: {len(allrows)} | teks umum: {len(general)} | domain (sintetik+riil): {len(domain)}")

    # downsample teks umum berdasarkan LABEL unik (jaga diversitas)
    by_label = collections.defaultdict(list)
    for img, lbl in general:
        by_label[lbl].append(img)
    gen_labels = list(by_label.keys())
    random.shuffle(gen_labels)

    kept_general = []
    for lbl in gen_labels:
        if len(kept_general) >= GENERAL_CAP:
            break
        for img in by_label[lbl][:MAX_PER_LABEL]:
            kept_general.append((img, lbl))

    print(f"Teks umum setelah downsample: {len(kept_general)}")

    combined = kept_general + domain
    pct_gen = 100 * len(kept_general) / len(combined)
    pct_dom = 100 * len(domain) / len(combined)
    print(f"Komposisi v3: teks umum {pct_gen:.1f}% | domain {pct_dom:.1f}%")

    # split anti-leakage per label
    by_label2 = collections.defaultdict(list)
    for img, lbl in combined:
        by_label2[lbl].append(img)
    labels = list(by_label2.keys())
    random.shuffle(labels)

    for lbl in labels:
        if len(by_label2[lbl]) > MAX_PER_LABEL:
            by_label2[lbl] = random.sample(by_label2[lbl], MAX_PER_LABEL)

    n_val = int(len(labels) * VAL_RATIO)
    val_labels = set(labels[:n_val])
    train_labels = set(labels[n_val:])

    train_rows, val_rows = [], []
    for lbl in train_labels:
        for img in by_label2[lbl]:
            train_rows.append((img, lbl))
    for lbl in val_labels:
        for img in by_label2[lbl]:
            val_rows.append((img, lbl))

    random.shuffle(train_rows); random.shuffle(val_rows)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for img, lbl in train_rows:
            f.write(f"{img}\t{lbl}\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for img, lbl in val_rows:
            f.write(f"{img}\t{lbl}\n")

    overlap = set(l for _, l in train_rows) & set(l for _, l in val_rows)
    print("\n=== HASIL v3 ===")
    print(f"Train: {len(train_rows)} sampel, {len(train_labels)} label unik")
    print(f"Val  : {len(val_rows)} sampel, {len(val_labels)} label unik")
    print(f"Leakage: {len(overlap)} (harus 0)")
    print(f"\n  {OUT_TRAIN}")
    print(f"  {OUT_VAL}")


if __name__ == "__main__":
    main()
