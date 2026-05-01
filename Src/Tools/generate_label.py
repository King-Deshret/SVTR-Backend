"""
Ini dibuat Auto-generate gt.txt untuk dataset
        yang belum punya label
        Menggunakan SVTRmodel.py untuk baca teks
        dari tiap gambar lesgo
"""

import os
import sys
import logging

sys.path.insert(0, r'D:\SVTR-Project')
from Src.Model.SVTRmodel import SVTRModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_labels(folder_path, model):
    """
    Baca semua gambar di folder
    Deteksi teks → simpan ke gt.txt
    """
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and '_' not in f  
    ])

    if not image_files:
        logger.warning(f"Tidak ada gambar di {folder_path}")
        return 0

    labels  = []
    success = 0
    failed  = 0

    for fname in image_files:
        img_path = os.path.join(folder_path, fname)

        import cv2
        image = cv2.imread(img_path)
        if image is None:
            failed += 1
            continue

        # Deteksi teks
        results = model.predict(image)

        if not results:
            failed += 1
            logger.warning(f"Tidak ada teks: {fname}")
            continue

        best = max(results, key=lambda x: x['confidence'])

        if best['confidence'] < 0.60:
            failed += 1
            logger.warning(
                f"Confidence terlalu rendah: "
                f"{fname} → {best['text']} "
                f"({best['confidence']*100:.1f}%)"
            )
            continue

        # Untuk gambar augmentasi, pakai label
        # yang sama dengan gambar originalnya
        base_name = fname.rsplit('.', 1)[0]  
        text      = best['text']

        # Simpan label untuk semua variasinya juga
        # (augmented version punya teks yang sama)
        variations = [
            f for f in os.listdir(folder_path)
            if f.startswith(base_name + '_')
            and f.endswith('.png')
        ]
        # Label gambar original
        labels.append(f"{fname}\t{text}")

        for var in variations:
            labels.append(f"{var}\t{text}")

        success += 1
        logger.info(
            f"  {fname} → '{text}' "
            f"({best['confidence']*100:.1f}%)"
            f" + {len(variations)} variasi"
        )

    gt_path = os.path.join(folder_path, 'gt.txt')
    with open(gt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(labels))

    logger.info(
        f"Selesai: {success} berhasil, "
        f"{failed} gagal → {gt_path}"
    )
    return len(labels)

if __name__ == "__main__":
    model = SVTRModel(debug=False)

    # Folder yang perlu di-generate labelnya
    # (yang belum punya gt.txt)
    target_folders = [
        r'D:\SVTR-Project\Data\Training\Dataset_kemasan',
        r'D:\SVTR-Project\Data\Training\Dataset_expdate',

    ]
    total = 0
    for folder in target_folders:
        if not os.path.exists(folder):
            logger.warning(f"Folder tidak ada: {folder}")
            continue

        gt_path = os.path.join(folder, 'gt.txt')
        if os.path.exists(gt_path):
            logger.info(
                f"Sudah punya label: {folder}"
            )
            continue

        logger.info(f"\nGenerate label: {folder}")
        count = generate_labels(folder, model)
        total += count

    logger.info(f"\nTotal label dibuat: {total}")
    logger.info(
        "Langkah berikutnya: "
        "jalankan prepare_training.py"
    )