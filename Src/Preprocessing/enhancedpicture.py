import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnhancedPicture:

    def __init__(self, debug=False):
        self.target_size = (128, 32)
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

    def correct_lighting(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
        return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    def reduce_noise(self, image):
        return cv2.fastNlMeansDenoisingColored(
            image, None,
            h=10, hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

    def deskew(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            M    = cv2.getRotationMatrix2D((w//2, h//2), median_angle, 1.0)
            return cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return image

    def correct_perspective(self, image):
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        largest = max(contours, key=cv2.contourArea)
        peri    = cv2.arcLength(largest, True)
        approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            pts_src = np.float32([p[0] for p in approx])
            h, w    = image.shape[:2]
            pts_dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
            pts_src = self._order_points(pts_src)
            M       = cv2.getPerspectiveTransform(pts_src, pts_dst)
            return cv2.warpPerspective(image, M, (w, h))
        return image

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def sharpen(self, image):
        gaussian   = cv2.GaussianBlur(image, (9, 9), 10.0)
        sharpened  = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return sharpened if blur_score < 100 else image

#define function untuk PPOCR v5 yang mana lebar min 640px dan lebar max 1280px
def normalize_size(self, image):
  
    h, w = image.shape[:2]
    if w < 640:
        ratio = 640 / w
        image = cv2.resize(
            image, (640, int(h * ratio)),
            interpolation=cv2.INTER_CUBIC
        )
    elif w > 1280:
        ratio = 1280 / w
        image = cv2.resize(
            image, (1280, int(h * ratio)),
            interpolation=cv2.INTER_AREA
        )
    return image

def process(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(
                f"Gambar tidak ditemukan: {image_path}"
            )
        gray        = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness  = np.mean(gray)
        blur_score  = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_score = np.std(gray)

        terlalu_gelap  = brightness < 80
        terlalu_terang = brightness > 180
        ada_blur       = blur_score < 100
        ada_noise      = noise_score > 60

        logger.debug(
            f"Analisa — Brightness:{brightness:.1f} "
            f"Blur:{blur_score:.1f} "
            f"Noise:{noise_score:.1f}"
        )
        image = self.deskew(image)
        logger.debug("[1] Deskew selesai")

        image = self.correct_perspective(image)
        logger.debug("[2] Perspective selesai")

        if terlalu_gelap or terlalu_terang:
            image = self.correct_lighting(image)
            logger.debug(
                f"[3] Lighting diterapkan "
                f"brightness={brightness:.1f}"
            )
        else:
            logger.debug(
                f"[3] Lighting dilewati "
                f"brightness={brightness:.1f}"
            )

        if ada_noise:
            image = self.reduce_noise(image)
            logger.debug(
                f"[4] Denoise diterapkan "
                f"noise={noise_score:.1f}"
            )
        else:
            logger.debug(
                f"[4] Denoise dilewati "
                f"noise={noise_score:.1f}"
            )

        if ada_blur:
            image = self.sharpen(image)
            logger.debug(
                f"[5] Sharpen diterapkan "
                f"blur={blur_score:.1f}"
            )
        else:
            logger.debug(
                f"[5] Sharpen dilewati "
                f"blur={blur_score:.1f}"
            )


        image = self.resize_for_svtr(image)
        logger.debug(f"[6] Resize output:{image.shape}")

        return image


if __name__ == "__main__":
    preprocessor = EnhancedPicture(debug=True)
    img_path     = r"D:\SVTR-Project\Data\Raw\Ultramilk.jpeg"
    try:
        result      = preprocessor.process(img_path)
        output_path = r"D:\SVTR-Project\Data\Processed\test_output.jpg"
        cv2.imwrite(output_path, result)
        logger.debug(f"Hasil disimpan: {output_path}")
    except FileNotFoundError as e:
        print(e)