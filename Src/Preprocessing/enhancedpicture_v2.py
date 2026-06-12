import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnhancedPictureV2:
    """
    Versi eksperimental preprocessing dengan logika yang DIPERBAIKI:

    Perbaikan dari versi lama:
    1. Deteksi GLARE benar — pakai persentase piksel over-exposed (lokal),
       bukan np.std global yang sebenarnya mengukur kontras.
    2. Deteksi BLUR benar — pakai variance of Laplacian (metrik blur standar),
       bukan np.std.
    3. Koreksi gamma TIDAK terbalik — gelap dinaikkan (gamma>1 pada formula
       output = input^(1/gamma)), terang diturunkan.
    4. Denoise lebih lembut + opsional, supaya tidak merusak dot-matrix.
    5. Semua parameter bisa di-override (untuk eksperimen optimasi numerik).

    Metrik analisis dikembalikan supaya bisa dipakai untuk korelasi
    metode numerik (data nyata: brightness, blur, glare ratio).
    """

    def __init__(self, debug=False,
                 gamma_dark=1.71, gamma_bright=0.6,
                 clahe_clip=2.80, denoise_h=5,
                 dark_thresh=80, bright_thresh=180,
                 blur_thresh=100.0, glare_thresh=0.08):
        self.debug = debug
        # parameter yang bisa dioptimasi (golden section, dll)
        self.gamma_dark = gamma_dark        # untuk gambar gelap (>1 = mencerahkan)
        self.gamma_bright = gamma_bright    # untuk gambar silau (<1 = menggelapkan)
        self.clahe_clip = clahe_clip
        self.denoise_h = denoise_h
        self.dark_thresh = dark_thresh
        self.bright_thresh = bright_thresh
        self.blur_thresh = blur_thresh      # var Laplacian < ini = blur
        self.glare_thresh = glare_thresh    # rasio piksel >240 > ini = glare
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    # ---------- LOADING ----------
    def _load_image(self, image_input):
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Gambar tidak ditemukan: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise ValueError("Input harus path (str) atau numpy array")
        return image

    # ---------- ANALISIS (metrik benar) ----------
    def analyze(self, image):
        """Kembalikan metrik citra yang BENAR untuk pengambilan keputusan."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        # BLUR: variance of Laplacian (makin kecil = makin blur)
        blur_metric = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # GLARE: rasio piksel sangat terang (over-exposed)
        glare_ratio = float(np.mean(gray > 240))
        # NOISE estimasi: median absolute deviation di high-freq
        noise_metric = float(np.median(np.abs(gray.astype(np.float32) -
                                              cv2.medianBlur(gray, 3).astype(np.float32))))
        metrics = {
            "brightness": brightness,
            "contrast": contrast,
            "blur_var_laplacian": blur_metric,
            "glare_ratio": glare_ratio,
            "noise_mad": noise_metric,
        }
        logger.debug(f"Metrik: {metrics}")
        return metrics

    # ---------- KOREKSI GAMMA (logika benar) ----------
    @staticmethod
    def _apply_gamma(channel, gamma):
        """output = 255 * (input/255)^(1/gamma).
        gamma>1 => mencerahkan; gamma<1 => menggelapkan."""
        inv = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(channel, table)

    def correct_lighting(self, image, brightness):
        """CLAHE untuk kontras lokal + gamma sesuai kondisi (tidak terbalik)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        l = clahe.apply(l)

        if brightness < self.dark_thresh:
            l = self._apply_gamma(l, self.gamma_dark)     # gelap -> cerahkan
            logger.debug(f"  gamma_dark={self.gamma_dark} (cerahkan)")
        elif brightness > self.bright_thresh:
            l = self._apply_gamma(l, self.gamma_bright)   # silau -> gelapkan
            logger.debug(f"  gamma_bright={self.gamma_bright} (gelapkan)")

        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ---------- GLARE REDUCTION ----------
    def reduce_glare(self, image):
        """Kurangi area over-exposed via inpainting pada highlight."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        if np.mean(mask > 0) > 0.001:
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return image

    # ---------- DENOISE (lembut) ----------
    def reduce_noise(self, image):
        # h kecil supaya tidak merusak dot-matrix
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=self.denoise_h, hColor=self.denoise_h,
            templateWindowSize=7, searchWindowSize=21
        )

    # ---------- SHARPEN (untuk blur) ----------
    def sharpen(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    # ---------- PIPELINE ----------
    def process(self, image_input, resize=False, return_metrics=False):
        image = self._load_image(image_input)
        m = self.analyze(image)
        applied = []

        # 1. Glare (jika rasio over-exposed tinggi)
        if m["glare_ratio"] > self.glare_thresh:
            image = self.reduce_glare(image)
            applied.append("glare_reduction")

        # 2. Lighting (gelap atau silau)
        if m["brightness"] < self.dark_thresh or m["brightness"] > self.bright_thresh:
            image = self.correct_lighting(image, m["brightness"])
            applied.append("lighting")

        # 3. Blur -> sharpen (var Laplacian rendah = blur)
        if m["blur_var_laplacian"] < self.blur_thresh:
            image = self.sharpen(image)
            applied.append("sharpen")

        # 4. Noise -> denoise lembut (hanya jika noise tinggi DAN tidak blur)
        if m["noise_mad"] > 8 and m["blur_var_laplacian"] >= self.blur_thresh:
            image = self.reduce_noise(image)
            applied.append("denoise")

        if resize:
            image = self.normalize_size(image)

        logger.debug(f"Diterapkan: {applied}")
        if return_metrics:
            return image, {**m, "applied": applied}
        return image

    def process_without_resize(self, image_input, return_metrics=False):
        return self.process(image_input, resize=False, return_metrics=return_metrics)

    def normalize_size(self, image):
        h, w = image.shape[:2]
        if w < 640:
            r = 640 / w
            image = cv2.resize(image, (640, int(h * r)), interpolation=cv2.INTER_CUBIC)
        elif w > 1280:
            r = 1280 / w
            image = cv2.resize(image, (1280, int(h * r)), interpolation=cv2.INTER_AREA)
        return image
