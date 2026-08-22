from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import contextmanager
import sqlite3
import os
from fastapi import File, UploadFile, Form
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import threading
import time
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv
from plate_normalizer import normalize_plate, InvalidPlateFormatError

# ─────────────── بارگذاری تنظیمات از فایل .env ───────────────
load_dotenv()

def _require_env(key: str) -> str:
    """اگه یه متغیر محیطی ضروری تنظیم نشده باشه، سرور بالا نمیاد
    (بهتره سرور اصلاً روشن نشه تا اینکه با مسیر اشتباه/خالی روشن بشه و بعداً خطای عجیب بده)"""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f" متغیر محیطی {key} تنظیم نشده. یه فایل .env بساز (نمونه‌اش .env.example هست) "
            f"و مقدار {key} رو توش بذار."
        )
    return value

# ─────────────── تنظیمات ───────────────
DB_PATH = os.getenv("DB_PATH", "plates.db")
PLATE_MODEL_PATH = _require_env("PLATE_MODEL_PATH")
OCR_MODEL_PATH = _require_env("OCR_MODEL_PATH")
FONT_PATH = _require_env("FONT_PATH")
ADMIN_API_KEY = _require_env("ADMIN_API_KEY")

_cors_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "")
CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if not CORS_ALLOWED_ORIGINS:
    raise RuntimeError(
        " متغیر محیطی CORS_ALLOWED_ORIGINS تنظیم نشده یا خالیه. "
        "آدرس فرانت‌اندت رو (مثلاً http://localhost:5173) توی .env بذار."
    )

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
print("در حال بارگذاری مدل‌ها... لطفاً صبر کن")
plate_model = YOLO(PLATE_MODEL_PATH)
ocr_model   = YOLO(OCR_MODEL_PATH)
print("مدل‌ها آماده‌ن ")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,   # فقط دامنه‌های تعریف‌شده توی .env
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────── وابستگی (dependency) بررسی کلید API ───────────────
def require_api_key(x_api_key: str = Header(default=None)):
    """
    این تابع رو جلوی هر endpoint حساس می‌ذاریم (گیت، حذف پلاک).
    درخواست‌کننده باید هدر X-API-Key رو با مقدار درست بفرسته، وگرنه رد می‌شه.
    """
    if not x_api_key or x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="کلید API نامعتبر یا وجود نداره")
    return True

# ─────────────── کمک‌کننده برای وصل شدن به دیتابیس ───────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# ─────────────── درگاه شماره ۱: تست ───────────────
@app.get("/")
def home():
    return {"message": "!سرور روشنه "}

# ─────────────── درگاه سلامت سرویس ───────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────── درگاه شماره ۲: لیست پلاک‌های مجاز ───────────────
@app.get("/allowed-plates")
def get_allowed_plates():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM allowed_plates").fetchall()
    return [dict(row) for row in rows]

# ─────────────── مدل ورودی برای افزودن پلاک ───────────────
# قبلاً plate/owner/... به‌شکل query parameter خونده می‌شدن، ولی فرانت‌اند
# اینا رو توی بدنه‌ی JSON می‌فرسته. این مدل باعث می‌شه از JSON body خونده بشن.
class AllowedPlateInput(BaseModel):
    plate: str
    owner: str = ""
    vehicle_type: str = ""
    notes: str = ""


# ─────────────── درگاه شماره ۳: اضافه کردن پلاک جدید ───────────────
@app.post("/allowed-plates")
def add_allowed_plate(data: AllowedPlateInput):
    # قدم ۱: پلاک خام رو یکدست کن؛ اگه فرمتش غلط بود، ۴۲۲ برگردون
    try:
        normalized_plate = normalize_plate(data.plate)
    except InvalidPlateFormatError as e:
        raise HTTPException(status_code=422, detail=str(e))

    with get_db() as conn:
        # قدم ۲: چک کن این پلاک از قبل وجود نداره (وگرنه ۴۰۹)
        existing = conn.execute(
            "SELECT id FROM allowed_plates WHERE plate = ?", (normalized_plate,)
        ).fetchone()
        if existing is not None:
            raise HTTPException(status_code=409, detail="این پلاک قبلاً توی لیست مجازها ثبت شده")

        # قدم ۳: فقط مقدار یکدست‌شده رو ذخیره کن، نه ورودی خام کاربر
        conn.execute(
            """INSERT INTO allowed_plates (plate, owner, vehicle_type, notes, created_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (normalized_plate, data.owner, data.vehicle_type, data.notes)
        )
    return {"status": "ok", "plate": normalized_plate}
# ─────────────── درگاه شماره ۴: لیست پلاک‌های "نیاز به بررسی" ───────────────
@app.get("/pending-review")
def get_pending_review():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE needs_review = 1 ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(row) for row in rows]

# ─────────────── مدل ورودی برای تأیید پلاک نامطمئن ───────────────
class ApproveDetectionInput(BaseModel):
    owner: str = ""


# ─────────────── درگاه شماره ۵: تأیید یه پلاک نامطمئن ───────────────
@app.post("/pending-review/{detection_id}/approve")
def approve_detection(detection_id: int, data: ApproveDetectionInput = ApproveDetectionInput()):
    with get_db() as conn:
        # قدم ۱: اول رکورد رو می‌خونیم فقط برای اینکه پلاکش رو بدونیم
        row = conn.execute(
            "SELECT plate, needs_review FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="این رکورد پیدا نشد")

        plate = row["plate"]

        # قدم ۲: کلید ماجرا اینجاست — به‌جای فقط UPDATE ساده، شرط
        # needs_review = 1 رو هم توی WHERE می‌ذاریم. یعنی این UPDATE فقط
        # وقتی واقعاً چیزی رو تغییر می‌ده که رکورد هنوز "نیاز به بررسی" باشه.
        # اگه یکی دیگه (یه اپراتور دیگه) قبلاً همین رکورد رو تأیید/رد کرده باشه،
        # needs_review از قبل صفر شده و این UPDATE هیچ ردیفی رو تغییر نمی‌ده.
        cursor = conn.execute(
            "UPDATE detections SET needs_review = 0 WHERE id = ? AND needs_review = 1",
            (detection_id,)
        )

        if cursor.rowcount == 0:
            # یعنی یکی دیگه زودتر از ما همین رکورد رو بررسی کرده
            raise HTTPException(
                status_code=409,
                detail="این رکورد قبلاً توسط یک اپراتور دیگه بررسی شده"
            )

        # قدم ۳: فقط اگه واقعاً ما بودیم که رکورد رو "بردیم"، پلاک رو
        # به لیست مجازها اضافه کن
        conn.execute(
            "INSERT OR IGNORE INTO allowed_plates (plate, owner) VALUES (?, ?)",
            (plate, data.owner)
        )

    return {"status": "ok", "plate": plate, "message": "تأیید شد و به لیست مجازها اضافه شد"}

# ─────────────── درگاه شماره ۶: رد کردن یه پلاک نامطمئن ───────────────
@app.post("/pending-review/{detection_id}/reject")
def reject_detection(detection_id: int):
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE detections SET needs_review = 0 WHERE id = ? AND needs_review = 1",
            (detection_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=409,
                detail="این رکورد قبلاً توسط یک اپراتور دیگه بررسی شده"
            )
    return {"status": "ok", "message": "رد شد"}



# ─────────────── درگاه شماره ۷: اضافه کردن پلاک از روی عکس ───────────────
@app.post("/allowed-plates/from-image")
async def add_from_image(file: UploadFile = File(...), owner: str = Form("")):
    # عکس آپلود شده رو بخون
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "فایل عکس معتبر نبود"}

    # قدم ۱: پیدا کردن پلاک توی عکس
    plate_results = plate_model(img, conf=0.5, verbose=False)[0]

    if len(plate_results.boxes) == 0:
        return {"status": "error", "message": "هیچ پلاکی توی عکس پیدا نشد"}

    # فقط اولین پلاک پیدا‌شده رو در نظر بگیر
    box = plate_results.boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    plate_crop = img[y1:y2, x1:x2]

    # قدم ۲: خوندن کاراکترهای پلاک
    ocr_results = ocr_model(plate_crop, conf=0.4, verbose=False)[0]
    chars = sorted(
        [(float(b.xyxy[0][0]), CLASS_MAP.get(int(b.cls[0]), '?')) for b in ocr_results.boxes],
        key=lambda c: c[0]
    )
    plate_text = ''.join([c[1] for c in chars])

    if not plate_text:
        return {"status": "error", "message": "نتونستم کاراکترهای پلاک رو بخونم"}

    # قدم ۳: یکدست‌سازی و بررسی فرمت (همون شکافی که قبلاً اینجا وجود نداشت)
    try:
        normalized_plate = normalize_plate(plate_text)
    except InvalidPlateFormatError as e:
        return {"status": "error", "message": f"فرمت پلاک شناسایی‌شده نامعتبره: {e}"}

    # قدم ۴: اضافه کردن به لیست پلاک‌های مجاز (فقط اگه قبلاً نبود)
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM allowed_plates WHERE plate = ?", (normalized_plate,)
        ).fetchone()
        if existing is not None:
            return {"status": "error", "message": "این پلاک قبلاً توی لیست مجازها ثبت شده", "plate": normalized_plate}

        conn.execute(
            "INSERT INTO allowed_plates (plate, owner) VALUES (?, ?)",
            (normalized_plate, owner)
        )

    return {"status": "ok", "plate": normalized_plate, "message": "پلاک با موفقیت اضافه شد"}



# ─────────────── درگاه شماره ۸: پلاک‌های غیرمجاز در ۲۴ ساعت اخیر (گروه‌بندی‌شده) ───────────────
@app.get("/unauthorized-last-24h")
def get_unauthorized_last_24h():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                plate,
                COUNT(*) AS times_seen,
                MAX(timestamp) AS last_seen
            FROM detections
            WHERE allowed = 0
              AND timestamp >= datetime('now', '-24 hours')
            GROUP BY plate
            ORDER BY last_seen DESC
        """).fetchall()
    return [dict(row) for row in rows]
# ─────────────── درگاه شماره ۹: آمار کلی امروز ───────────────
@app.get("/stats")
def get_stats():
    with get_db() as conn:
        today_total = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now')"
        ).fetchone()[0]
        today_allowed = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now') AND allowed = 1"
        ).fetchone()[0]
        today_unauthorized = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now') AND allowed = 0 AND needs_review = 0"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM detections WHERE needs_review = 1"
        ).fetchone()[0]
        total_allowed_plates = conn.execute(
            "SELECT COUNT(*) FROM allowed_plates"
        ).fetchone()[0]

    return {
        "total_today": today_total,
        "allowed_today": today_allowed,
        "unauthorized_today": today_unauthorized,
        "pending_review": pending,
        "allowed_plates_total": total_allowed_plates,
    }

# ─────────────── درگاه شماره ۱۰: لیست کامل لاگ تردد ───────────────
@app.get("/detections")
def get_all_detections():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
    return [dict(row) for row in rows]

# ─────────────── درگاه شماره ۱۱: حذف یه پلاک از لیست مجازها (نیازمند کلید API) ───────────────
@app.delete("/allowed-plates/{plate_id}")
def delete_allowed_plate(plate_id: int, authorized: bool = Depends(require_api_key)):
    with get_db() as conn:
        conn.execute("DELETE FROM allowed_plates WHERE id = ?", (plate_id,))
    return {"status": "ok", "message": "پلاک حذف شد"}

# ─────────────── مدل ورودی برای کنترل گیت ───────────────
class GateToggleInput(BaseModel):
    action: str = ""
    mode: str = ""


# ─────────────── درگاه شماره ۱۲: کنترل گیت (باز/بسته دستی) — نیازمند کلید API ───────────────
gate_state = {"isOpen": False, "mode": "AUTO", "lastAction": "سیستم آماده است"}

@app.post("/gate/toggle")
def toggle_gate(data: GateToggleInput, authorized: bool = Depends(require_api_key)):
    if data.mode:
        gate_state["mode"] = data.mode
    if data.action == "OPEN":
        gate_state["isOpen"] = True
        gate_state["lastAction"] = "گیت به صورت دستی باز شد"
    elif data.action == "CLOSE":
        gate_state["isOpen"] = False
        gate_state["lastAction"] = "گیت به صورت دستی بسته شد"
    return {"status": "ok", "gateState": gate_state}

# ─────────────── اصلاح جدول: اضافه کردن ستون‌های گم‌شده ───────────────
def upgrade_db():
    with get_db() as conn:
        # چک می‌کنیم این ستون‌ها از قبل هستن یا نه، تا خطا نده
        existing_columns = [row["name"] for row in conn.execute("PRAGMA table_info(allowed_plates)")]

        if "vehicle_type" not in existing_columns:
            conn.execute("ALTER TABLE allowed_plates ADD COLUMN vehicle_type TEXT DEFAULT ''")
        if "notes" not in existing_columns:
            conn.execute("ALTER TABLE allowed_plates ADD COLUMN notes TEXT DEFAULT ''")
        if "created_at" not in existing_columns:
            conn.execute("ALTER TABLE allowed_plates ADD COLUMN created_at TEXT DEFAULT ''")
            # برای پلاک‌های قدیمی که created_at ندارن، یه تاریخ بذاریم
            conn.execute("UPDATE allowed_plates SET created_at = datetime('now') WHERE created_at = ''")

upgrade_db()


# بخش جدید: دوربین زنده داخل بک‌اند


CONF_PLATE = 0.5
CONF_CHAR = 0.4
CONFIDENCE_THRESHOLD = 0.6
DEDUP_SECONDS = 10
EN_TO_FA_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')

# ─────────────── کمک‌کننده: نوشتن فارسی روی تصویر ───────────────
def draw_persian_text(frame, text, pos, color=(0, 255, 0), size=26):
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

# ─────────────── کمک‌کننده: خوندن کاراکترهای پلاک ───────────────
def read_plate_chars(plate_img):
    results = ocr_model(plate_img, conf=CONF_CHAR, verbose=False)[0]
    chars = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1 = float(box.xyxy[0][0])
        conf = float(box.conf[0])
        char = CLASS_MAP.get(cls_id, '?')
        chars.append((x1, char, conf))
    chars.sort(key=lambda c: c[0])
    return chars

def is_valid_plate_format(chars: list) -> bool:
    # منسوخ: دیگه هیچ‌جا صدا زده نمی‌شه. فرمت‌چک الان توی plate_normalizer.py انجام می‌شه.
    # اینجا نگهش داشتیم که تغییر قابل‌برگشت بمونه؛ بعداً می‌شه کامل حذفش کرد.
    if len(chars) != 8:
        return False
    digits = [c.isdigit() for c in chars]
    return digits[0] and digits[1] and not digits[2] and all(digits[3:8])

def check_is_allowed(plate: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM allowed_plates WHERE plate = ?", (plate,)
        ).fetchone()
    return row is not None

def save_detection(plate: str, allowed: bool, needs_review: bool, confidence: float):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO detections (plate, allowed, needs_review, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (plate, int(allowed), int(needs_review), confidence,
             datetime.now().isoformat(timespec="seconds"))
        )

# ─────────────── حلقه‌ی دوربین (توی یه ریسمان جدا اجرا میشه) ───────────────
latest_frame = None
frame_lock = threading.Lock()
last_seen_plates = {}

def camera_loop():
    global latest_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(" دوربین باز نشد! بخش تصویر زنده کار نمی‌کنه.")
        return

    print(" دوربین داخل بک‌اند روشن شد.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        plate_results = plate_model(frame, conf=CONF_PLATE, verbose=False)[0]

        for box in plate_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            chars_with_conf = read_plate_chars(plate_crop)
            chars_only = [c[1] for c in chars_with_conf]
            raw_plate_text = ''.join(chars_only)

            # به‌جای is_valid_plate_format قدیمی، از normalizer مشترک استفاده می‌کنیم
            # همون تابعی که توی endpoint افزودن پلاک هم استفاده می‌شه
            try:
                plate_text = normalize_plate(raw_plate_text)
            except InvalidPlateFormatError:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                continue

            avg_conf = sum(c[2] for c in chars_with_conf) / len(chars_with_conf)
            needs_review = avg_conf < CONFIDENCE_THRESHOLD
            # فقط مقدار یکدست‌شده با دیتابیس مقایسه می‌شه
            allowed = check_is_allowed(plate_text)

            if needs_review:
                color = (0, 255, 255)
                status_text = "نیاز به بررسی"
            elif allowed:
                color = (0, 255, 0)
                status_text = "مجاز"
            else:
                color = (0, 0, 255)
                status_text = "غیرمجاز"

            now = time.time()
            if now - last_seen_plates.get(plate_text, 0) > DEDUP_SECONDS:
                last_seen_plates[plate_text] = now
                save_detection(plate_text, allowed, needs_review, avg_conf)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            fa_text = ''.join(c.translate(EN_TO_FA_DIGITS) if c.isdigit() else c for c in plate_text)
            frame = draw_persian_text(frame, f"{fa_text} - {status_text}", (x1, max(0, y1 - 32)), color)

        # فریم رو به فرمت JPEG تبدیل کن و توی حافظه‌ی مشترک بذار
        ok, buffer = cv2.imencode('.jpg', frame)
        if ok:
            with frame_lock:
                latest_frame = buffer.tobytes()

        time.sleep(0.03)  # حدود ۳۰ فریم در ثانیه

# ─────────────── راه‌اندازی خودکار دوربین هنگام روشن شدن سرور ───────────────
@app.on_event("startup")
def start_camera_thread():
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()

# ─────────────── درگاه شماره ۱۳: استریم زنده‌ی تصویر ───────────────
def frame_generator():
    while True:
        with frame_lock:
            frame_bytes = latest_frame
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )