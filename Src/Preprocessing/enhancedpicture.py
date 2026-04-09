import cv2
import numpy as np

class EnhancedPicture:
    def __init__(self):
        self.target_size = (128, 32)  

    def correct_lighting(self, image):
        """
        Menangani: terlalu terang, terlalu gelap.
        Teknik   : CLAHE + Gamma Correction adaptif
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        mean_brightness = np.mean(l_clahe)
        if mean_brightness < 80:
            gamma = 0.6
        elif mean_brightness > 180:

            gamma = 1.6
        else:
            gamma = 1.0

        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")
            l_clahe = cv2.LUT(l_clahe, table)

        lab_merged = cv2.merge([l_clahe, a, b])
        result = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
        return result

    def reduce_noise(self, image):
        """
        Menangani: noise kamera, noise kompresi, dot-matrix.
        Teknik   : Non-local Means Denoising (lebih baik dari Gaussian)
        """
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=10,           
            hColor=10,      
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised

    def deskew(self, image):
        """
        Menangani: gambar miring saat foto kemasan.
        Teknik   : Hough Line Transform → deteksi sudut → rotasi
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return image 

        angles = []
        for line in lines[:20]: 
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 <= angle <= 45:
                angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        if abs(median_angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated

        return image

    def correct_perspective(self, image):
        """
        Menangani: kemasan kaleng/botol melengkung, sudut pengambilan foto.
        Teknik   : Contour detection → Perspective Transform
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return image

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) == 4:
            pts_src = np.float32(
                [point[0] for point in approx]
            )
            h, w = image.shape[:2]
            pts_dst = np.float32([
                [0, 0], [w, 0], [w, h], [0, h]
            ])
    
            pts_src = self._order_points(pts_src)
            M = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped = cv2.warpPerspective(image, M, (w, h))
            return warped

        return image

    def _order_points(self, pts):
        """Urutkan titik sudut: TL, TR, BR, BL"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   
        rect[2] = pts[np.argmax(s)]   
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  
        rect[3] = pts[np.argmax(diff)]  
        return rect

    def sharpen(self, image):
        """
        Menangani: gambar blur / tidak fokus.
        Teknik   : Unsharp Masking (lebih halus dari kernel sharpen biasa)
        """
      
        gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)

        sharpened = cv2.addWeighted(
            image, 1.5,      
            gaussian, -0.5,   
            0
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 100:
            return sharpened
        return image

    def binarize(self, image):
        """
        Menangani: font unik, teks emboss, teks dot-matrix.
        Teknik   : Adaptive Gaussian Threshold + Morfologi
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,   
            C=8             
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2, 2)
        )

        cleaned = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=1
        )

        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return result

    def resize_for_svtr(self, image):
        """
        Resize ke 128×32 px — format standar input SVTR.
        Aspect ratio dipertahankan dengan padding.
        """
        h, w = image.shape[:2]
        target_w, target_h = self.target_size  
        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        resized = cv2.resize(
            image, (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        padded = np.ones(
            (target_h, target_w, 3), dtype=np.uint8
        ) * 255
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded[y_offset:y_offset+new_h,
               x_offset:x_offset+new_w] = resized

        return padded

    def process(self, image_path):
        """
        Jalankan semua tahap preprocessing secara berurutan.
        Input  : path gambar kemasan
        Output : citra siap masuk SVTR (numpy array 32×128×3)
        """

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(
                f"Gambar tidak ditemukan: {image_path}"
            )

        print(f"[1/7] Input  : {image.shape}")

        # Tahap 1 — Koreksi pencahayaan
        image = self.correct_lighting(image)
        print(f"[2/7] Lighting correction selesai")

        # Tahap 2 — Mengurangi noise
        image = self.reduce_noise(image)
        print(f"[3/7] Noise reduction selesai")

        # Tahap 3 — Cek kemiringan gambar
        image = self.deskew(image)
        print(f"[4/7] Deskewing selesai")

        # Tahap 4 — Koreksi perspektif gambar
        image = self.correct_perspective(image)
        print(f"[5/7] Perspective correction selesai")

        # Tahap 5 — Sharpening
        image = self.sharpen(image)
        print(f"[6/7] Sharpening selesai")

        # Tahap 6 — Binarisasi
        image = self.binarize(image)
        print(f"[7/7] Binarization selesai")

        # Tahap 7 — Resize ke format SVTR
        image = self.resize_for_svtr(image)
        print(f"[OK]  Output : {image.shape} — siap masuk SVTR")

        return image

if __name__ == "__main__":
    preprocessor = EnhancedPicture()

    img_path = r"D:\SVTR-Project\Data\Raw\greenfieldsusu.jpg"

    try:
        result = preprocessor.process(img_path)

        output_path = "Data/Processed/kemasan_processed.jpg"
        cv2.imwrite(output_path, result)
        print(f"\nHasil disimpan di: {output_path}")

        original = cv2.imread(img_path)
        original = cv2.imread(img_path)
        original_resized = cv2.resize(original, (128, 32))
        comparison = np.hstack([original_resized, result])
        
        display_scale = 4
        h, w = comparison.shape[:2]
        comparison_display = cv2.resize(comparison, (w * display_scale, h * display_scale), interpolation=cv2.INTER_NEAREST)
        
    
        cv2.imshow("Kiri: Original (128x32) | Kanan: Processed (128x32) - DIPERBESAR DEMO", comparison_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)