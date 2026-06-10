import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnhancedPicture:

    def __init__(self, debug=False):
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

    def _load_image(self, image_input):
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise FileNotFoundError(f"Gambar tidak ditemukan: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise ValueError("Input harus berupa path (str) atau numpy array")
        return image

    def _analyze(self, image):
        gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        noise      = np.std(gray)
        logger.debug(f"Analisa — Brightness:{brightness:.1f}  Noise:{noise:.1f}")
        return brightness, noise

    def correct_lighting(self, image):
        lab        = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe    = clahe.apply(l)
        brightness = np.mean(l_clahe)

        if brightness < 80:
            gamma = 0.6
        elif brightness > 180:
            gamma = 1.6
        else:
            gamma = 1.0

        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
            ]).astype("uint8")
            l_clahe = cv2.LUT(l_clahe, table)

        return cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    def reduce_noise(self, image):
        return cv2.fastNlMeansDenoisingColored(
            image, None, h=10, hColor=10,
            templateWindowSize=7, searchWindowSize=21
        )

    def normalize_size(self, image):
        h, w = image.shape[:2]
        if w < 640:
            ratio = 640 / w
            image = cv2.resize(image, (640, int(h * ratio)), interpolation=cv2.INTER_CUBIC)
        elif w > 1280:
            ratio = 1280 / w
            image = cv2.resize(image, (1280, int(h * ratio)), interpolation=cv2.INTER_AREA)
        return image

    def process(self, image_input):
        """Aktifkan preprocessing"""
        image            = self._load_image(image_input)
        brightness, noise = self._analyze(image)

        if brightness < 80 or brightness > 180:
            image = self.correct_lighting(image)
            logger.debug(f"[1] Lighting diterapkan (brightness={brightness:.1f})")
        else:
            logger.debug(f"[1] Lighting dilewati (normal={brightness:.1f})")

        if noise > 60:
            image = self.reduce_noise(image)
            logger.debug(f"[2] Denoise diterapkan (noise={noise:.1f})")
        else:
            logger.debug(f"[2] Denoise dilewati (noise={noise:.1f})")

        image = self.normalize_size(image)
        logger.debug(f"[3] Output: {image.shape}")
        return image

    def process_without_resize(self, image_input):
 
        image            = self._load_image(image_input)
        brightness, noise = self._analyze(image)

        if brightness < 80 or brightness > 180:
            image = self.correct_lighting(image)
            logger.debug(f"[1] Lighting diterapkan (brightness={brightness:.1f})")
        else:
            logger.debug(f"[1] Lighting dilewati (normal={brightness:.1f})")

        if noise > 60:
            image = self.reduce_noise(image)
            logger.debug(f"[2] Denoise diterapkan (noise={noise:.1f})")
        else:
            logger.debug(f"[2] Denoise dilewati (noise={noise:.1f})")

        logger.debug(f"[3] Output (no-resize): {image.shape}")
        return image