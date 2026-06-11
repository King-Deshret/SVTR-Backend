# -*- coding: utf-8 -*-
"""
Generator data sintetik kode produksi & tanggal kadaluarsa untuk training SVTR.

Menghasilkan ribuan string UNIK (bukan augmentasi dari string sama) dengan
variasi visual (blur, distorsi, skew, background) lewat TRDG.

Output:
- Gambar di: Data/Training/synthetic/
- Label di : Data/Annotation/synthetic_labels.txt  (format: path<TAB>teks)

Jalankan dengan venv yang punya TRDG (Pillow 9.5):
    D:\svtr_export_work\exp_env\Scripts\python.exe Src/Tools/generate_synthetic.py

Catatan: gabungkan synthetic_labels.txt ke train_clean.txt SEBELUM split,
lalu jalankan rebuild_clean_split.py agar tetap bebas leakage.
"""
import os
import random
import string

random.seed(123)

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
OUT_IMG = os.path.join(PROJ, "Data", "Training", "synthetic")
OUT_LBL = os.path.join(PROJ, "Data", "Annotation", "synthetic_labels.txt")
os.makedirs(OUT_IMG, exist_ok=True)

N_KODE = 2500      # jumlah kode produksi unik
N_TANGGAL = 1500   # jumlah tanggal unik

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def rand_kode():
    """Kode produksi acak ala industri: huruf+angka, mis. OK14703AA0, L25B3."""
    fmts = [
        lambda: "".join(random.choices(string.ascii_uppercase, k=2)) +
                "".join(random.choices(string.digits, k=5)) +
                "".join(random.choices(string.ascii_uppercase, k=2)) + str(random.randint(0, 9)),
        lambda: "L" + "".join(random.choices(string.digits, k=random.randint(3, 5))) +
                random.choice(string.ascii_uppercase),
        lambda: "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(8, 10))),
        lambda: random.choice(["LOT", "BATCH", "PRD"]) + " " +
                "".join(random.choices(string.ascii_uppercase + string.digits, k=6)),
    ]
    return random.choice(fmts)()


def rand_tanggal():
    """Tanggal/expiry beragam format."""
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = random.randint(2024, 2027)
    fmts = [
        lambda: f"{d:02d}{m:02d}{y}",
        lambda: f"{d:02d}/{m:02d}/{y}",
        lambda: f"{d:02d}-{m:02d}-{y}",
        lambda: f"{d:02d} {MONTHS[m-1]} {str(y)[2:]}",
        lambda: f"EXP {d:02d}{MONTHS[m-1]}{str(y)[2:]}",
        lambda: f"BB {d:02d}{m:02d}{y}",
        lambda: f"BEST BEFORE {d:02d}/{m:02d}/{y}",
    ]
    return random.choice(fmts)()


def main():
    from trdg.generators import GeneratorFromStrings

    # buat string unik
    strings = set()
    while len(strings) < N_KODE:
        strings.add(rand_kode())
    while len(strings) < N_KODE + N_TANGGAL:
        strings.add(rand_tanggal())
    strings = list(strings)
    random.shuffle(strings)
    print(f"Total string unik: {len(strings)}")

    # TRDG generator dengan variasi:
    #   skewing, distorsi, blur, background acak
    gen = GeneratorFromStrings(
        strings,
        count=len(strings),
        size=48,                    # tinggi 48px (sesuai rec_image_shape)
        skewing_angle=3,
        random_skew=True,
        blur=2,
        random_blur=True,
        background_type=random.choice([0, 1, 2]),
        distorsion_type=3,          # random distortion
    )

    labels = []
    rel_dir = "Data/Training/synthetic"
    for i, (img, lbl) in enumerate(gen):
        if img is None:
            continue
        fname = f"syn_{i:05d}.jpg"
        img.convert("RGB").save(os.path.join(OUT_IMG, fname))
        labels.append(f"{rel_dir}/{fname}\t{lbl}")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(strings)} generated...")

    with open(OUT_LBL, "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")

    print(f"\nSelesai. {len(labels)} gambar sintetik.")
    print(f"  Gambar : {OUT_IMG}")
    print(f"  Label  : {OUT_LBL}")


if __name__ == "__main__":
    main()
