"""
Revisi program, gunanya buat baca semua dataset, fix path secara otomatis lalu digabung, dan pisah 
atau split train/val final (semoga aman wkwk)

Menangani semua format yang ditemukan:
- Dataset_train_1    : word_1.png, "teks"
- train_data_27...   : D:\PaddleOCR\...\file.png  teks
- train_data ICDAR   : D:\PaddleOCR\...\file.png  teks
- train_data_Real+S  : D:\PaddleOCR\...\file.png  teks
- train_data_real+i  : rec\train\file.png  teks
(source dari laptop saya untuk saat ini)
"""

import os
import re
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Konfigurasi ──
PROJECT_ROOT = r'D:\SVTR-Project'
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'Data', 'Training')
ANNOT_DIR    = os.path.join(PROJECT_ROOT, 'Data', 'Annotation')
VAL_RATIO    = 0.2

os.makedirs(ANNOT_DIR, exist_ok=True)


def fix_path(raw_path, folder_path, folder_name):
    """
    Konversi berbagai format path ke format
    relatif yang benar untuk PaddleOCR training.

    Format output: Data/Training/[folder]/train/file.png
    """
    raw_path = raw_path.strip()

    # Ambil nama file saja (strip semua path prefix)
    filename = os.path.basename(
        raw_path.replace('\\', '/')
    )

    if not filename:
        return None

    # Cari file di subfolder train/ atau langsung
    candidates = [
        os.path.join(folder_path, 'train', filename),
        os.path.join(folder_path, filename),
        os.path.join(folder_path, 'rec', 'train', filename),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            # Konversi ke path relatif dari project root
            rel = os.path.relpath(
                candidate, PROJECT_ROOT
            ).replace('\\', '/')
            return rel

    return None

def parse_label_line(line, fmt):
    """
    Parse satu baris label berdasarkan format.
    Return (path_raw, teks) atau None kalau gagal.
    """
    line = line.strip()
    if not line:
        return None

    if fmt == 'comma_quote':
        # Format: word_1.png, "Genaxis Theatre"
        match = re.match(
            r'^(.+?),\s*["\']?(.+?)["\']?\s*$', line
        )
        if match:
            return match.group(1).strip(), \
                   match.group(2).strip()

    elif fmt == 'tab':
        # Format: path\file.png[TAB]teks
        parts = line.split('\t')
        if len(parts) >= 2:
            return parts[0].strip(), \
                   '\t'.join(parts[1:]).strip()

    elif fmt == 'space':
        # Format: path\file.png teks
        parts = line.split(None, 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    return None

def detect_format(gt_path):
    """Deteksi format file label."""
    with open(gt_path, 'r',
              encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                return 'tab'
            elif re.match(r'^.+?,\s*["\']', line):
                return 'comma_quote'
            else:
                return 'space'
    return 'tab'


def load_dataset(folder_path, folder_name):
    """
    Load satu folder dataset.
    Cari file label, parse, fix path.
    """
    label_candidates = [
        os.path.join(folder_path, 'gt.txt'),
        os.path.join(folder_path, 'rec_gt_train.txt'),
        os.path.join(folder_path, 'rec', 'rec_gt_train.txt'),
        os.path.join(folder_path, 'label.txt'),
        os.path.join(folder_path, 'train.txt'),
    ]

    gt_path = None
    for candidate in label_candidates:
        if os.path.exists(candidate):
            gt_path = candidate
            break

    if not gt_path:
        logger.warning(
            f"[{folder_name}] Tidak ada file label!"
        )
        return []

    fmt = detect_format(gt_path)
    logger.info(
        f"[{folder_name}] "
        f"Label: {os.path.basename(gt_path)} "
        f"| Format: {fmt}"
    )

    labels  = []
    skipped = 0

    with open(gt_path, 'r',
              encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_label_line(line, fmt)
            if not parsed:
                continue

            raw_path, text = parsed

            text = text.strip().strip('"').strip("'")
            if not text:
                skipped += 1
                continue

            # Fix path ke format relatif
            fixed_path = fix_path(
                raw_path, folder_path, folder_name
            )

            if not fixed_path:
                skipped += 1
                continue

            # Format final untuk PaddleOCR
            labels.append(f"{fixed_path}\t{text}")

    logger.info(
        f"[{folder_name}] "
        f"Berhasil: {len(labels)} | "
        f"Skip: {skipped}"
    )
    return labels


def prepare():
    logger.info("="*55)
    logger.info("PREPARE TRAINING — Mulai")
    logger.info("="*55)

    all_labels = []

    # Scan semua folder di Training/
    folders = sorted([
        f for f in os.listdir(TRAINING_DIR)
        if os.path.isdir(
            os.path.join(TRAINING_DIR, f)
        )
    ])

    logger.info(
        f"Ditemukan {len(folders)} folder dataset\n"
    )

    for folder_name in folders:
        folder_path = os.path.join(
            TRAINING_DIR, folder_name
        )
        labels = load_dataset(folder_path, folder_name)
        all_labels.extend(labels)
        logger.info("")

    if not all_labels:
        logger.error(
            "Tidak ada label valid ditemukan!\n"
            "Cek apakah file gambar ada di "
            "folder train/ masing-masing dataset."
        )
        return

    all_labels = list(set(all_labels))
    logger.info(f"Total label unik: {len(all_labels)}")

    # Filter teks kosong atau terlalu pendek
    valid = [
        l for l in all_labels
        if len(l.split('\t')) == 2
        and len(l.split('\t')[1].strip()) >= 1
    ]
    logger.info(f"Label valid     : {len(valid)}")

    if not valid:
        logger.error("Tidak ada label valid!")
        return

    # Shuffle dengan seed tetap (reproducible)
    random.seed(42)
    random.shuffle(valid)

    # Split 80% train / 20% val
    val_size     = int(len(valid) * VAL_RATIO)
    val_labels   = valid[:val_size]
    train_labels = valid[val_size:]

    train_path = os.path.join(
        ANNOT_DIR, 'train_final.txt'
    )
    val_path = os.path.join(
        ANNOT_DIR, 'val_final.txt'
    )

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_labels))

    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_labels))

    logger.info("\n" + "="*55)
    logger.info("SELESAI")
    logger.info(f"Total valid : {len(valid)}")
    logger.info(f"Train       : {len(train_labels)}")
    logger.info(f"Validasi    : {len(val_labels)}")
    logger.info(f"Train file  : {train_path}")
    logger.info(f"Val file    : {val_path}")
    logger.info("="*55)
    logger.info("\nLangkah berikutnya:")
    logger.info(
        "  Cek 5 baris pertama train_final.txt:"
    )
    logger.info(
        "  Get-Content Data\\Annotation\\"
        "train_final.txt -TotalCount 5"
    )


if __name__ == "__main__":
    prepare()