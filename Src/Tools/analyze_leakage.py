# -*- coding: utf-8 -*-
"""
Analisa data leakage antara train dan val set.

Data leakage terjadi ketika label (teks) yang sama muncul di train DAN val.
Karena dataset ini banyak augmentasi (1 gambar asli -> ratusan varian),
string yang sama bocor ke kedua set, membuat metrik validasi terlalu optimis.

Output:
- Berapa banyak label val yang juga ada di train (leakage rate)
- Distribusi duplikasi
- Estimasi akurasi "jujur" vs "bocor"

Jalankan: python Src/Tools/analyze_leakage.py
"""
import os
import collections

ANN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Data", "Annotation")
TRAIN = os.path.join(ANN_DIR, "train_final.txt")
VAL = os.path.join(ANN_DIR, "val_final.txt")


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
    train = parse(TRAIN)
    val = parse(VAL)

    train_labels = [l for _, l in train]
    val_labels = [l for _, l in val]

    train_set = set(train_labels)
    val_set = set(val_labels)

    # Leakage: label val yang juga ada di train
    leaked = [l for l in val_labels if l in train_set]
    leak_rate = 100 * len(leaked) / max(1, len(val_labels))

    print("=" * 60)
    print("ANALISA DATA LEAKAGE (train <-> val)")
    print("=" * 60)
    print(f"Train samples       : {len(train)}")
    print(f"Val samples         : {len(val)}")
    print(f"Train label unik    : {len(train_set)}")
    print(f"Val label unik      : {len(val_set)}")
    print()
    print(f"Val samples yang labelnya JUGA ada di train: {len(leaked)} / {len(val_labels)}")
    print(f">>> LEAKAGE RATE: {leak_rate:.1f}% <<<")
    print()
    print("Interpretasi:")
    if leak_rate > 50:
        print("  PARAH. Lebih dari separuh val bocor. Metrik validasi TIDAK VALID.")
    elif leak_rate > 20:
        print("  SIGNIFIKAN. Metrik validasi terlalu optimis.")
    else:
        print("  Relatif kecil.")

    # Berapa val label yang benar-benar unik (tidak ada di train) = ujian jujur
    truly_unseen = val_set - train_set
    print(f"\nVal label yang TIDAK pernah ada di train (unseen): {len(truly_unseen)}")
    print(f"  -> Ini yang seharusnya jadi dasar evaluasi jujur.")
    print(f"  Contoh unseen: {list(truly_unseen)[:10]}")

    # Duplikasi dalam train
    tc = collections.Counter(train_labels)
    dup = {k: v for k, v in tc.items() if v > 10}
    print(f"\nLabel di train dengan >10 duplikat: {len(dup)}")
    top = sorted(dup.items(), key=lambda x: -x[1])[:10]
    for label, n in top:
        print(f"   {n:5d}x  {label!r}")


if __name__ == "__main__":
    main()
