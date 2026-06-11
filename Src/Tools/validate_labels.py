# -*- coding: utf-8 -*-
"""
Validasi label hasil anotasi terhadap dictionary (custom_dict.txt).
Laporkan:
- Karakter yang TIDAK ada di dictionary (akan bermasalah saat training)
- Label sangat pendek (<=1 char) yang mungkin noise
- Statistik panjang

Jalankan: python Src/Tools/validate_labels.py <file_label.txt>
Default: rec_roboflow_final.txt
"""
import os
import sys
import collections

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
DICT = os.path.join(PROJ, "Src", "Model", "Inference", "custom_dict.txt")
DEFAULT = os.path.join(PROJ, "Data", "Annotation", "rec_roboflow_final.txt")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT

    with open(DICT, "rb") as f:
        chars = set(l for l in f.read().decode("utf-8").split("\n") if l != "")
    # spasi termasuk valid
    valid = set(chars) | {" "}
    print(f"Dictionary: {len(chars)} karakter (+ spasi)")

    rows = []
    with open(target, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" in line:
                p, t = line.split("\t", 1)
                rows.append((p, t))

    print(f"Total label: {len(rows)}\n")

    oov = collections.Counter()       # out-of-vocabulary chars
    oov_rows = []
    short = []
    lengths = []
    for p, t in rows:
        lengths.append(len(t))
        bad = [c for c in t if c not in valid]
        if bad:
            oov.update(bad)
            oov_rows.append((p, t, bad))
        if len(t.strip()) <= 1:
            short.append((p, t))

    print("=== KARAKTER DI LUAR DICTIONARY (OOV) ===")
    if oov:
        for c, n in oov.most_common():
            print(f"   {c!r} (U+{ord(c):04X}) x{n}")
        print(f"\n   {len(oov_rows)} label mengandung OOV. Contoh:")
        for p, t, bad in oov_rows[:10]:
            print(f"     {t!r} -> OOV={bad}  ({os.path.basename(p)})")
    else:
        print("   Tidak ada. Semua karakter valid.")

    print(f"\n=== LABEL SANGAT PENDEK (<=1 char): {len(short)} ===")
    for p, t in short[:10]:
        print(f"   {t!r}  ({os.path.basename(p)})")

    if lengths:
        print(f"\n=== STATISTIK ===")
        print(f"   panjang rata-rata: {sum(lengths)/len(lengths):.1f}")
        print(f"   panjang min/max  : {min(lengths)}/{max(lengths)}")


if __name__ == "__main__":
    main()
