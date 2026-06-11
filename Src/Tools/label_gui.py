# -*- coding: utf-8 -*-
"""
Tool pelabelan interaktif untuk crop recognition Roboflow.

Menampilkan tiap gambar crop + prediksi awal (pre-label).
Kamu tinggal koreksi teks lalu Enter -> lanjut ke gambar berikutnya.
Progress otomatis tersimpan, bisa lanjut kapan saja.

FITUR:
- Gambar di-zoom besar biar jelas
- Teks prediksi sudah terisi (tinggal koreksi)
- Filter confidence: default tampilkan yang perlu koreksi (0.30-0.90)
- Enter = simpan & lanjut | Tombol Skip = lewati (hapus dari final)
- Auto-save tiap aksi

Jalankan (pakai Python utama, bukan venv):
    python Src/Tools/label_gui.py

Output: Data/Annotation/rec_roboflow_final.txt  (format: path<TAB>teks)
"""
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

PROJ = os.path.join(os.path.dirname(__file__), "..", "..")
PRELABEL = os.path.join(PROJ, "Data", "Annotation", "rec_roboflow_prelabeled.txt")
FINAL = os.path.join(PROJ, "Data", "Annotation", "rec_roboflow_final.txt")

# Filter: hanya label crop dengan confidence di rentang ini (yang perlu koreksi).
# Yang >MAX dianggap sudah benar (auto-terima), yang <MIN dianggap sampah (auto-skip).
CONF_MIN = 0.30
CONF_MAX = 0.90
AUTO_ACCEPT_HIGH = True   # confidence > CONF_MAX langsung diterima tanpa review


def load_items():
    items = []
    with open(PRELABEL, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            path, text, conf = parts[0], parts[1], float(parts[2])
            items.append({"path": path, "text": text, "conf": conf})
    return items


def load_done():
    done = {}
    if os.path.exists(FINAL):
        with open(FINAL, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if "\t" in line:
                    p, t = line.split("\t", 1)
                    done[p] = t
    return done


class LabelApp:
    def __init__(self, root, items, done):
        self.root = root
        self.done = done
        # bagi item: auto-accept tinggi, review menengah, skip rendah
        self.auto = [it for it in items if it["conf"] > CONF_MAX]
        self.review = [it for it in items
                       if CONF_MIN <= it["conf"] <= CONF_MAX
                       and it["path"] not in done]
        self.idx = 0

        # auto-accept yang confidence tinggi (sekali saja)
        if AUTO_ACCEPT_HIGH:
            for it in self.auto:
                if it["path"] not in self.done:
                    self.done[it["path"]] = it["text"]
            self.save()

        root.title("SVTR Crop Labeler")
        root.geometry("700x500")

        self.progress = tk.Label(root, text="", font=("Arial", 11))
        self.progress.pack(pady=5)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.conf_label = tk.Label(root, text="", font=("Arial", 9), fg="gray")
        self.conf_label.pack()

        self.entry = tk.Entry(root, font=("Consolas", 18), width=30, justify="center")
        self.entry.pack(pady=10)
        self.entry.bind("<Return>", self.save_next)

        btns = tk.Frame(root)
        btns.pack(pady=5)
        tk.Button(btns, text="Simpan & Lanjut (Enter)", command=self.save_next).pack(side="left", padx=5)
        tk.Button(btns, text="Skip (buang)", command=self.skip).pack(side="left", padx=5)
        tk.Button(btns, text="Mundur", command=self.prev).pack(side="left", padx=5)

        self.show()
        self.entry.focus_set()

    def show(self):
        if self.idx >= len(self.review):
            self.progress.config(text="SELESAI! Semua crop sudah dilabeli.")
            self.img_label.config(image="")
            self.entry.delete(0, tk.END)
            self.conf_label.config(text=f"Total tersimpan: {len(self.done)}")
            return
        it = self.review[self.idx]
        self.progress.config(
            text=f"Review {self.idx+1}/{len(self.review)}  "
                 f"(auto-terima conf tinggi: {len(self.auto)})")
        full = os.path.join(PROJ, it["path"])
        try:
            img = Image.open(full)
            # zoom: tinggi target 120px, lebar proporsional, maks 640
            h = 120
            w = min(640, int(img.width * h / img.height))
            img = img.resize((w, h))
            self.tkimg = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tkimg)
        except Exception as e:
            self.img_label.config(text=f"(gagal load: {e})", image="")
        self.conf_label.config(text=f"confidence pra-label: {it['conf']:.2f}  |  file: {os.path.basename(it['path'])}")
        self.entry.delete(0, tk.END)
        self.entry.insert(0, it["text"])
        self.entry.focus_set()
        self.entry.select_range(0, tk.END)

    def save_next(self, event=None):
        if self.idx >= len(self.review):
            return
        it = self.review[self.idx]
        text = self.entry.get().strip()
        if text:
            self.done[it["path"]] = text
        self.save()
        self.idx += 1
        self.show()

    def skip(self):
        if self.idx >= len(self.review):
            return
        it = self.review[self.idx]
        self.done.pop(it["path"], None)
        self.save()
        self.idx += 1
        self.show()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.show()

    def save(self):
        with open(FINAL, "w", encoding="utf-8") as f:
            for p, t in self.done.items():
                f.write(f"{p}\t{t}\n")


def main():
    items = load_items()
    done = load_done()
    root = tk.Tk()
    LabelApp(root, items, done)
    root.mainloop()


if __name__ == "__main__":
    main()
