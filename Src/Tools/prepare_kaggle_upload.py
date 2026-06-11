# -*- coding: utf-8 -*-
"""
Siapkan folder upload Kaggle otomatis: kumpulkan semua gambar + label + config
ke satu folder D:/svtr_kaggle_upload/ dengan struktur path yang benar,
lalu (opsional) zip.

Jalankan: python Src/Tools/prepare_kaggle_upload.py
"""
import os
import shutil

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = r"D:\svtr_kaggle_upload"

# folder gambar yang dipakai di train_v2/val_v2
IMG_FOLDERS = [
    "Dataset_train_1",
    "train_data ICDAR",
    "train_data_27_2_2025_1024",
    "train_data_real+icdar",
    "train_data_Real+Sintetik",
    "synthetic",
    "roboflow_crops",
]

FILES = [
    ("Data/Annotation/train_v2.txt", "Data/Annotation/train_v2.txt"),
    ("Data/Annotation/val_v2.txt", "Data/Annotation/val_v2.txt"),
    ("Src/Model/Inference/custom_dict.txt", "custom_dict.txt"),
    ("Src/Tools/svtr_finetune_v2.yml", "svtr_finetune_v2.yml"),
]


def main():
    if os.path.exists(DEST):
        print(f"Hapus folder lama: {DEST}")
        shutil.rmtree(DEST)
    os.makedirs(DEST)

    # copy folder gambar
    for folder in IMG_FOLDERS:
        src = os.path.join(PROJ, "Data", "Training", folder)
        dst = os.path.join(DEST, "Data", "Training", folder)
        if os.path.exists(src):
            print(f"copy {folder} ...")
            shutil.copytree(src, dst)
        else:
            print(f"  LEWATI (tidak ada): {folder}")

    # copy file label & config
    for src_rel, dst_rel in FILES:
        src = os.path.join(PROJ, src_rel)
        dst = os.path.join(DEST, dst_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"copy {dst_rel}")
        else:
            print(f"  LEWATI (tidak ada): {src_rel}")

    print(f"\nSelesai. Folder upload siap di: {DEST}")
    print("Upload folder ini (atau zip-nya) ke Kaggle sebagai dataset baru.")


if __name__ == "__main__":
    main()
