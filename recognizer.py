"""Core two-stage Persian plate recognizer (no web / DB code).

Stage 1: YOLOv8 plate detector  -> plate boxes
Stage 2: YOLOv8 character detector on each crop -> characters sorted left→right

Shared by the API (admin_api.py), the evaluator (evaluate.py) and the demo (demo.py)
so that all three run exactly the same inference settings.
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

# The 30 classes the OCR model was trained on (class id → Persian glyph).
CLASS_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'الف',  # A
    11: 'ب',    # B
    12: 'ث',    # C
    13: 'د',    # D
    14: 'ع',    # EIN
    15: 'ق',    # GH
    16: 'ه',    # H
    17: 'ج',    # J
    18: 'ژ',    # JHMJ (پلاک معلولین)
    19: 'ل',    # L
    20: 'م',    # M
    21: 'ن',    # N
    22: 'پ',    # P
    23: 'س',    # S
    24: 'ص',    # SAD
    25: 'ش',    # SH
    26: 'ت',    # T
    27: 'ط',    # TA
    28: 'و',    # V
    29: 'ی',    # Y
}


def is_valid_plate_format(chars: list) -> bool:
    """Standard private Iranian plate: 2 digits, 1 letter, 3 digits, 2-digit region code."""
    if len(chars) != 8:
        return False
    digits = [c.isdigit() for c in chars]
    return digits[0] and digits[1] and not digits[2] and all(digits[3:8])


@dataclass
class PlateResult:
    box: tuple                      # (x1, y1, x2, y2) in the input image
    det_conf: float                 # plate detector confidence
    chars: list = field(default_factory=list)   # [(x1, char, conf)] sorted left→right
    char_boxes: list = field(default_factory=list)  # [(x1, y1, x2, y2)] in crop coordinates, same order

    @property
    def text(self) -> str:
        return ''.join(c[1] for c in self.chars)

    @property
    def char_list(self) -> list:
        return [c[1] for c in self.chars]

    @property
    def ocr_conf(self) -> float:
        """Mean character confidence (0 if nothing was read)."""
        return sum(c[2] for c in self.chars) / len(self.chars) if self.chars else 0.0

    @property
    def valid(self) -> bool:
        return is_valid_plate_format(self.char_list)


class PlateRecognizer:
    def __init__(self, plate_model_path, ocr_model_path, conf_plate=0.5, conf_char=0.4, ocr_imgsz=800):
        self.plate_model = YOLO(str(plate_model_path))
        self.ocr_model = YOLO(str(ocr_model_path))
        self.conf_plate = conf_plate
        self.conf_char = conf_char
        self.ocr_imgsz = ocr_imgsz     # must match the OCR model's training size (800)
        self._lock = threading.Lock()  # YOLO models are not thread-safe

    def detect_plates(self, img):
        with self._lock:
            return self.plate_model(img, conf=self.conf_plate, verbose=False)[0]

    def read_chars_boxes(self, plate_img):
        """([(x1, char, conf)], [(x1, y1, x2, y2)]) sorted left→right, boxes in crop coordinates."""
        with self._lock:
            res = self.ocr_model(
                plate_img,
                conf=self.conf_char,
                imgsz=self.ocr_imgsz,
                agnostic_nms=True,   # two classes on the same glyph → keep only the best
                verbose=False,
            )[0]
        items = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            items.append(((x1, CLASS_MAP.get(int(b.cls[0]), '?'), float(b.conf[0])), (x1, y1, x2, y2)))
        items.sort(key=lambda it: it[0][0])
        return [it[0] for it in items], [it[1] for it in items]

    def read_chars(self, plate_img):
        """[(x1, char, conf), ...] sorted left→right."""
        return self.read_chars_boxes(plate_img)[0]

    def recognize(self, img):
        """Full pipeline on one BGR image → list[PlateResult], best detection first."""
        results = []
        for box in self.detect_plates(img).boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            chars, boxes = self.read_chars_boxes(crop)
            results.append(PlateResult((x1, y1, x2, y2), float(box.conf[0]), chars, boxes))
        results.sort(key=lambda r: r.det_conf, reverse=True)
        return results
