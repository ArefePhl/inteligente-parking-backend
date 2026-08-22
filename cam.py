# cam.py
# اجرای مستقل دوربین برای تست (بدون API) — نسخه‌ای برای استفاده از فایل .env
# uvicorn admin_api:app --reload

import os
import cv2
import time
import sqlite3
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv

# ─────────────── تنظیمات (از فایل .env خونده می‌شن) ───────────────
load_dotenv()

def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f" متغیر محیطی {key} تنظیم نشده. یه فایل .env بساز (نمونه‌اش .env.example هست) "
            f"و مقدار {key} رو توش بذار."
        )
    return value

PLATE_MODEL_PATH = _require_env("PLATE_MODEL_PATH")
OCR_MODEL_PATH   = _require_env("OCR_MODEL_PATH")
FONT_PATH        = _require_env("FONT_PATH")
DB_PATH          = os.getenv("DB_PATH", "plates.db")

CONF_PLATE = 0.5          # حداقل اطمینان برای تشخیص خود پلاک
CONF_CHAR  = 0.4          # حداقل اطمینان برای هر کاراکتر (زیر این، اصلاً کاراکتر رو نمی‌بینه)
CONFIDENCE_THRESHOLD = 0.6
DEDUP_SECONDS = 10        # جلوگیری از ثبت تکراری همون پلاک

# نگاشت کلاس‌های مدل OCR به کاراکتر واقعی
CLASS_MAP = {
    0: '0',   1: '1',   2: '2',   3: '3',   4: '4',
    5: '5',   6: '6',   7: '7',   8: '8',   9: '9',
    10: 'الف',   # A
    11: 'ب',     # B
    12: 'ث',     # C
    13: 'د',     # D
    14: 'ع',     # EIN
    15: 'ق',     # GH
    16: 'ه',     # H
    17: 'ج',     # J
    18: 'ژ',     # JHMJ (پلاک معلولین)
    19: 'ل',     # L
    20: 'م',     # M
    21: 'ن',     # N
    22: 'پ',     # P
    23: 'س',     # S
    24: 'ص',     # SAD
    25: 'ش',     # SH
    26: 'ت',     # T
    27: 'ط',     # TA
    28: 'و',     # V
    29: 'ی',     # Y
    30: 'دیپلمات',       # DIPLOMAT
    31: 'ستاد',          # F_SETAD
    32: 'گذر موقت',      # GH_GOZAR
    33: 'کشاورزی',       # K_KESHAVARZI
    34: 'ارتش',          # ARTESH
    35: 'تشریفات',       # TASHRIFAT
    36: 'سپاه',          # SEPAH
    37: 'وزارت دفاع',    # Z_DEFA
}

EN_TO_FA_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')

# ─────────────── دیتابیس ───────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                allowed INTEGER NOT NULL,
                needs_review INTEGER NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allowed_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL UNIQUE,
                owner TEXT
            )
        """)

def is_allowed(plate: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM allowed_plates WHERE plate = ?", (plate,)
        ).fetchone()
    return row is not None

def log_detection(plate: str, allowed: bool, needs_review: bool, confidence: float):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO detections (plate, allowed, needs_review, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (plate, int(allowed), int(needs_review), confidence,
             datetime.now().isoformat(timespec="seconds"))
        )

# ─────────────── اعتبارسنجی فرمت پلاک ───────────────
def is_valid_plate(chars: list) -> bool:
    """فرمت استاندارد: ۲ رقم + ۱ حرف + ۳ رقم + ۲ رقم = ۸ کاراکتر"""
    if len(chars) != 8:
        return False
    digits = [c.isdigit() for c in chars]
    return (digits[0] and digits[1] and not digits[2] and all(digits[3:8]))

# ─────────────── رندر متن فارسی روی تصویر ───────────────
def draw_persian_text(frame, text, pos, color=(0, 255, 0), size=28):
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = str(get_display(reshaped))
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, size)
    except OSError:
        font = ImageFont.load_default()
    draw.text(pos, bidi_text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ─────────────── خواندن کاراکترهای پلاک + اطمینان هر کدوم ───────────────
def read_plate(ocr_model, plate_img):
    """
    خروجی: لیستی از (x, کاراکتر, اطمینان) — مرتب‌شده از چپ به راست
    """
    results = ocr_model(plate_img, conf=CONF_CHAR, verbose=False)[0]
    chars = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1 = float(box.xyxy[0][0])
        confidence = float(box.conf[0])
        char = CLASS_MAP.get(cls_id, '?')
        chars.append((x1, char, confidence))
    chars.sort(key=lambda c: c[0])  # چپ به راست
    return chars

# ─────────────── حلقه اصلی ───────────────
def main():
    init_db()
    plate_model = YOLO(PLATE_MODEL_PATH)
    ocr_model   = YOLO(OCR_MODEL_PATH)

    last_seen = {}  # {plate: last_timestamp} برای جلوگیری از ثبت تکراری

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" دوربین باز نشد!")
        return

    print(" سیستم فعال شد. برای خروج کلید q رو بزن.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        plate_results = plate_model(frame, conf=CONF_PLATE, verbose=False)[0]

        for box in plate_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # خوندن کاراکترها (هر کدوم با عدد اطمینانش)
            chars_with_conf = read_plate(ocr_model, plate_crop)

            # فقط کاراکترها (بدون اطمینان) رو جدا می‌کنیم برای اعتبارسنجی فرمت
            chars_only = [c[1] for c in chars_with_conf]

            if not is_valid_plate(chars_only):
                # فرمت درست نبود -> اصلاً پلاک معتبر نیست، فقط کادر زرد کم‌رنگ بکش و رد شو
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                continue

            # میانگین اطمینان کل پلاک رو حساب کن
            avg_confidence = sum(c[2] for c in chars_with_conf) / len(chars_with_conf)
            needs_review = avg_confidence < CONFIDENCE_THRESHOLD

            plate_text = ''.join(chars_only)
            allowed = is_allowed(plate_text)

            # ─── تعیین رنگ و وضعیت ───
            if needs_review:
                color = (0, 255, 255)      # زرد = نامطمئن
                status_text = "نیاز به بررسی"
            elif allowed:
                color = (0, 255, 0)        # سبز = مجاز
                status_text = "مجاز"
            else:
                color = (0, 0, 255)        # قرمز = غیرمجاز
                status_text = "غیرمجاز"

            # ─── ضدتکرار: هر پلاک حداکثر هر ۱۰ ثانیه یک بار ثبت بشه ───
            now = time.time()
            if now - last_seen.get(plate_text, 0) > DEDUP_SECONDS:
                last_seen[plate_text] = now
                log_detection(plate_text, allowed, needs_review, avg_confidence)
                print(f"[{datetime.now():%H:%M:%S}] {plate_text} → {status_text} "
                      f"(اطمینان: {avg_confidence:.2f})")

            # ─── نمایش روی تصویر ───
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            fa_digits_text = ''.join(c.translate(EN_TO_FA_DIGITS) if c.isdigit() else c
                                      for c in plate_text)
            label = f"{fa_digits_text} - {status_text}"
            frame = draw_persian_text(frame, label, (x1, max(0, y1 - 35)), color)

        cv2.imshow("Plate Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
