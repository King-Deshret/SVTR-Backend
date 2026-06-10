"""
Revisi program, gunanya buat baca semua dataset, fix path secara otomatis lalu digabung, dan pisah 
atau split train/val final (semoga aman wkwk)
"""

import os
import re
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = r'D:\SVTR-Project'
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'Data', 'Training')
ANNOT_DIR    = os.path.join(PROJECT_ROOT, 'Data', 'Annotation')
VAL_RATIO    = 0.2

os.makedirs(ANNOT_DIR, exist_ok=True)


def fix_path(raw_path, folder_path):
    """
    Cari file gambar di lokasi yang benar.
    Coba semua kemungkinan subfolder.
    """
    filename = os.path.basename(raw_path.replace('\\', '/'))
    if not filename:
        return None

    candidates = [
        os.path.join(folder_path, filename),
        os.path.join(folder_path, 'train', filename),
        os.path.join(folder_path, 'rec', 'train', filename),
        os.path.join(folder_path, 'rec', filename),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            rel = os.path.relpath(candidate, PROJECT_ROOT)
            return rel.replace('\\', '/')

    return None


def parse_label_line(line, fmt):
    """Parse satu baris label."""
    line = line.strip()
    if not line:
        return None

    if fmt == 'comma_quote':
        match = re.match(r"^(.+?),\s*[\"']?(.+?)[\"']?\s*$", line)
        if match:
            return match.group(1).strip(), match.group(2).strip()

    elif fmt == 'tab':
        parts = line.split('\t')
        if len(parts) >= 2:
            return parts[0].strip(), '\t'.join(parts[1:]).strip()

    elif fmt == 'space':
        parts = line.split(None, 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()

    return None


def detect_format(gt_path):
    """Deteksi format file label otomatis."""
    with open(gt_path, 'r', encoding='utf-8', errors='ignore') as f:
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
    """Load satu folder dataset."""

    # Urutan prioritas pencarian file label
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
        logger.warning(f"[{folder_name}] Tidak ada file label!")
        return []

    fmt = detect_format(gt_path)
    logger.info(f"[{folder_name}] File: {os.path.basename(gt_path)} | Format: {fmt}")

    labels  = []
    skipped = 0

    with open(gt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_label_line(line, fmt)
            if not parsed:
                continue

            raw_path, text = parsed
            text = text.strip().strip('"').strip("'")

            # Skip label kosong atau terlalu pendek
            if not text or len(text) < 1:
                skipped += 1
                continue

            # Skip label yang hanya simbol semua (noise)
            if all(not c.isalnum() for c in text):
                skipped += 1
                logger.debug(f"  Skip noise: '{text}'")
                continue

            fixed = fix_path(raw_path, folder_path)
            if not fixed:
                skipped += 1
                logger.debug(f"  File tidak ditemukan: {raw_path}")
                continue

            labels.append(f"{fixed}\t{text}")

    logger.info(f"[{folder_name}] OK: {len(labels)} | Skip: {skipped}")
    return labels


def prepare():
    logger.info("=" * 55)
    logger.info("PREPARE TRAINING — Mulai")
    logger.info("=" * 55)

    all_labels = []
    folders = sorted([
        f for f in os.listdir(TRAINING_DIR)
        if os.path.isdir(os.path.join(TRAINING_DIR, f))
    ])
    logger.info(f"Ditemukan {len(folders)} folder\n")

    for folder_name in folders:
        folder_path = os.path.join(TRAINING_DIR, folder_name)
        labels = load_dataset(folder_path, folder_name)
        all_labels.extend(labels)
        logger.info("")

    if not all_labels:
        logger.error("Tidak ada label valid!")
        return

    # Deduplikasi
    all_labels = list(set(all_labels))
    logger.info(f"Total label unik  : {len(all_labels)}")

    # Filter valid
    valid = [
        l for l in all_labels
        if len(l.split('\t')) == 2
        and len(l.split('\t')[1].strip()) >= 1
    ]
    logger.info(f"Label valid       : {len(valid)}")

    if not valid:
        logger.error("Tidak ada label valid!")
        return

    random.seed(42)
    random.shuffle(valid)

    val_size     = int(len(valid) * VAL_RATIO)
    val_labels   = valid[:val_size]
    train_labels = valid[val_size:]

    train_path = os.path.join(ANNOT_DIR, 'train_final.txt')
    val_path   = os.path.join(ANNOT_DIR, 'val_final.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_labels))
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_labels))

    logger.info("\n" + "=" * 55)
    logger.info("SELESAI")
    logger.info(f"Total valid  : {len(valid)}")
    logger.info(f"Train        : {len(train_labels)}")
    logger.info(f"Val          : {len(val_labels)}")
    logger.info(f"Train file   : {train_path}")
    logger.info(f"Val file     : {val_path}")
    logger.info("=" * 55)
    logger.info("\nCek hasil:")
    logger.info("  Get-Content Data\\Annotation\\train_final.txt -TotalCount 5")


if __name__ == "__main__":
    prepare()