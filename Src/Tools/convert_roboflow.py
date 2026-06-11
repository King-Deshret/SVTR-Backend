# -*- coding: utf-8 -*-
"""
Konversi dataset Roboflow COCO (Exp_date_dataset) menjadi:

1. DETECTION (format PaddleOCR det):
   <image_path>\t[{"transcription":"date","points":[[x1,y1],...]}]
   -> Data/Annotation/det_roboflow_{split}.txt

2. RECOGNITION CROP:
   Crop tiap bounding box jadi gambar terpisah di Data/Training/roboflow_crops/
   + template anotasi (teks DIKOSONGKAN, untuk diisi manual)
   -> Data/Annotation/rec_roboflow_template.txt
      format: <crop_path>\t???   <- ganti ??? dengan teks asli

Jalankan:
    python Src/Tools/convert_roboflow.py
"""
import os
import json

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
DS = os.path.join(PROJ, "Data", "Training", "Exp_date_dataset")
ANN_DIR = os.path.join(PROJ, "Data", "Annotation")
CROP_DIR = os.path.join(PROJ, "Data", "Training", "roboflow_crops")
os.makedirs(CROP_DIR, exist_ok=True)

import cv2

SPLITS = ["train", "valid", "test"]


def main():
    rec_template = []
    crop_idx = 0

    for split in SPLITS:
        split_dir = os.path.join(DS, split)
        coco_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(coco_path):
            print(f"SKIP {split} (no coco json)")
            continue

        with open(coco_path, encoding="utf-8") as f:
            coco = json.load(f)

        # map image_id -> file info
        img_map = {im["id"]: im for im in coco["images"]}
        # group annotations by image
        anns_by_img = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        det_lines = []
        for img_id, im in img_map.items():
            fname = im["file_name"]
            img_path = os.path.join(split_dir, fname)
            if not os.path.exists(img_path):
                continue
            anns = anns_by_img.get(img_id, [])
            if not anns:
                continue

            # build detection label (polygon dari bbox)
            regions = []
            img = cv2.imread(img_path)
            if img is None:
                continue
            for ann in anns:
                x, y, w, h = ann["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                regions.append({"transcription": "date", "points": pts})

                # crop untuk recognition
                crop = img[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    continue
                crop_name = f"rfcrop_{crop_idx:05d}.jpg"
                cv2.imwrite(os.path.join(CROP_DIR, crop_name), crop)
                rec_template.append(f"Data/Training/roboflow_crops/{crop_name}\t???")
                crop_idx += 1

            rel = f"Data/Training/Exp_date_dataset/{split}/{fname}"
            det_lines.append(f"{rel}\t{json.dumps(regions, ensure_ascii=False)}")

        out_det = os.path.join(ANN_DIR, f"det_roboflow_{split}.txt")
        with open(out_det, "w", encoding="utf-8") as f:
            f.write("\n".join(det_lines) + "\n")
        print(f"{split}: {len(det_lines)} gambar -> {out_det}")

    out_rec = os.path.join(ANN_DIR, "rec_roboflow_template.txt")
    with open(out_rec, "w", encoding="utf-8") as f:
        f.write("\n".join(rec_template) + "\n")
    print(f"\nTotal crop recognition: {crop_idx}")
    print(f"Template anotasi: {out_rec}")
    print("  -> Ganti '???' dengan teks asli tiap crop (anotasi manual)")


if __name__ == "__main__":
    main()
