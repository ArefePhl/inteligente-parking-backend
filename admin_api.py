"""Persian license-plate gate: FastAPI backend + live camera loop.

Run:
    set ADMIN_API_KEY=your-secret          (Windows)   /  export ADMIN_API_KEY=...
    uvicorn admin_api:app --host 127.0.0.1 --port 8000

All settings come from environment variables (see .env.example).
"""
import hmac
import os
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from viz import draw_persian_text, to_fa_digits
from recognizer import PlateRecognizer, is_valid_plate_format

# ─────────────── تنظیمات (همه از متغیر محیطی، با مقدار پیش‌فرض نسبی) ───────────────
BASE_DIR = Path(__file__).resolve().parent

DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "plates.db"))
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", str(BASE_DIR / "models" / "plate" / "best.pt"))
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH", str(BASE_DIR / "models" / "ocr" / "best.pt"))
FONT_PATH = os.getenv("FONT_PATH", r"C:\Windows\Fonts\tahoma.ttf")

# امنیت
API_KEY = os.getenv("ADMIN_API_KEY", "")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if o.strip()
]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# مدل‌ها / دوربین
CONF_PLATE = float(os.getenv("CONF_PLATE", "0.5"))
CONF_CHAR = float(os.getenv("CONF_CHAR", "0.4"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
OCR_IMGSZ = int(os.getenv("OCR_IMGSZ", "800"))  # مدل OCR با 800 آموزش دیده
DEDUP_SECONDS = int(os.getenv("DEDUP_SECONDS", "10"))
ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "1") == "1"
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
LOAD_MODELS = os.getenv("LOAD_MODELS", "1") == "1"


recognizer = None  # PlateRecognizer؛ قفل داخلی دارد چون دوربین و درخواست‌ها هم‌زمان استفاده می‌کنند
stop_event = threading.Event()


# ─────────────── زمان: همیشه ساعت محلی، با فرمت سازگار با SQLite ───────────────
def now_str() -> str:
    """'2026-10-24 11:14:05' (فاصله به‌جای T تا با datetime() در SQLite درست مقایسه شود)."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


# ─────────────── دیتابیس ───────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS allowed_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL UNIQUE,
                owner TEXT DEFAULT '',
                vehicle_type TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT ''
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                allowed INTEGER NOT NULL DEFAULT 0,
                needs_review INTEGER NOT NULL DEFAULT 0,
                confidence REAL,
                timestamp TEXT NOT NULL
            )""")

        # دیتابیس‌های قدیمی: ستون‌های گم‌شده
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(allowed_plates)")]
        if "vehicle_type" not in cols:
            conn.execute("ALTER TABLE allowed_plates ADD COLUMN vehicle_type TEXT DEFAULT ''")
        if "notes" not in cols:
            conn.execute("ALTER TABLE allowed_plates ADD COLUMN notes TEXT DEFAULT ''")
        if "created_at" not in cols:
            conn.execute("ALTER TABLE allowed_plates ADD COLUMN created_at TEXT DEFAULT ''")
        conn.execute("UPDATE allowed_plates SET created_at = datetime('now','localtime') WHERE created_at = ''")

        # تضمین یکتایی پلاک (INSERT OR IGNORE بدون آن بی‌اثر است)
        try:
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_allowed_plate ON allowed_plates(plate)")
        except sqlite3.IntegrityError:
            print("هشدار: allowed_plates پلاک تکراری دارد؛ ایندکس یکتا ساخته نشد.")

        # رکوردهای قدیمی با فرمت ISO دارای 'T' را به فرمت فاصله‌دار تبدیل کن
        conn.execute("UPDATE detections SET timestamp = replace(timestamp, 'T', ' ') WHERE timestamp LIKE '%T%'")


# ─────────────── احراز هویت ───────────────
def _key_ok(candidate: str) -> bool:
    return bool(API_KEY) and hmac.compare_digest(candidate.encode(), API_KEY.encode())


def require_api_key(x_api_key: str = Header(default="")):
    if not _key_ok(x_api_key):
        raise HTTPException(status_code=401, detail="کلید API نامعتبر است")


def require_api_key_or_query(x_api_key: str = Header(default=""), api_key: str = Query(default="")):
    """برای /video_feed: تگ <img> نمی‌تواند هدر بفرستد، پس کوئری هم پذیرفته می‌شود."""
    if not (_key_ok(x_api_key) or _key_ok(api_key)):
        raise HTTPException(status_code=401, detail="کلید API نامعتبر است")


# ─────────────── کمک‌کننده‌های مدل ───────────────
def detect_plates(img):
    return recognizer.detect_plates(img)


def read_plate_chars(plate_img):
    """[(x1, char, conf), ...] مرتب‌شده از چپ به راست."""
    return recognizer.read_chars(plate_img)


def check_is_allowed(plate: str) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM allowed_plates WHERE plate = ?", (plate,)).fetchone()
    return row is not None


def save_detection(plate: str, allowed: bool, needs_review: bool, confidence: float):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO detections (plate, allowed, needs_review, confidence, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (plate, int(allowed), int(needs_review), confidence, now_str()),
        )


# ─────────────── حلقه‌ی دوربین ───────────────
latest_frame = None
frame_lock = threading.Lock()
last_seen_plates = {}


def camera_loop():
    global latest_frame

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not cap.isOpened():
        print("دوربین باز نشد! بخش تصویر زنده کار نمی‌کند.")
        return
    print("دوربین داخل بک‌اند روشن شد.")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            plate_results = detect_plates(frame)

            for box in plate_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                chars_with_conf = read_plate_chars(plate_crop)
                chars_only = [c[1] for c in chars_with_conf]

                if not is_valid_plate_format(chars_only):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    continue

                avg_conf = sum(c[2] for c in chars_with_conf) / len(chars_with_conf)
                needs_review = avg_conf < CONFIDENCE_THRESHOLD
                plate_text = ''.join(chars_only)
                allowed = check_is_allowed(plate_text)

                if needs_review:
                    color, status_text = (0, 255, 255), "نیاز به بررسی"
                elif allowed:
                    color, status_text = (0, 255, 0), "مجاز"
                else:
                    color, status_text = (0, 0, 255), "غیرمجاز"

                now = time.time()
                if now - last_seen_plates.get(plate_text, 0) > DEDUP_SECONDS:
                    last_seen_plates[plate_text] = now
                    save_detection(plate_text, allowed, needs_review, avg_conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                fa_text = to_fa_digits(plate_text)
                frame = draw_persian_text(frame, f"{fa_text} - {status_text}", (x1, max(0, y1 - 32)), color)

            ok, buffer = cv2.imencode('.jpg', frame)
            if ok:
                with frame_lock:
                    latest_frame = buffer.tobytes()

            time.sleep(0.03)
    finally:
        cap.release()


# ─────────────── چرخه‌ی عمر برنامه ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    if not API_KEY:
        raise RuntimeError("متغیر محیطی ADMIN_API_KEY تنظیم نشده است؛ سرور بدون کلید بالا نمی‌آید.")

    init_db()
    if LOAD_MODELS:
        print("در حال بارگذاری مدل‌ها... لطفاً صبر کن")
        recognizer = PlateRecognizer(
            PLATE_MODEL_PATH, OCR_MODEL_PATH,
            conf_plate=CONF_PLATE, conf_char=CONF_CHAR, ocr_imgsz=OCR_IMGSZ,
        )
        print("مدل‌ها آماده‌اند")

    thread = None
    if ENABLE_CAMERA and LOAD_MODELS:
        stop_event.clear()
        thread = threading.Thread(target=camera_loop, daemon=True)
        thread.start()

    yield

    stop_event.set()
    if thread is not None:
        thread.join(timeout=3)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

protected = APIRouter(dependencies=[Depends(require_api_key)])


# ─────────────── درگاه‌ها ───────────────
@app.get("/")
def home():
    return {"message": "!سرور روشنه "}


@protected.get("/allowed-plates")
def get_allowed_plates():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM allowed_plates").fetchall()
    return [dict(row) for row in rows]


@protected.post("/allowed-plates")
def add_allowed_plate(plate: str, owner: str = "", vehicle_type: str = "", notes: str = ""):
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO allowed_plates (plate, owner, vehicle_type, notes, created_at)
               VALUES (?, ?, ?, ?, datetime('now','localtime'))""",
            (plate, owner, vehicle_type, notes),
        )
    return {"status": "ok", "plate": plate}


@protected.get("/pending-review")
def get_pending_review():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM detections WHERE needs_review = 1 ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@protected.post("/pending-review/{detection_id}/approve")
def approve_detection(detection_id: int, owner: str = ""):
    with get_db() as conn:
        row = conn.execute("SELECT plate FROM detections WHERE id = ?", (detection_id,)).fetchone()
        if row is None:
            return {"status": "error", "message": "این رکورد پیدا نشد"}
        plate = row["plate"]
        conn.execute(
            """INSERT OR IGNORE INTO allowed_plates (plate, owner, created_at)
               VALUES (?, ?, datetime('now','localtime'))""",
            (plate, owner),
        )
        conn.execute("UPDATE detections SET needs_review = 0 WHERE id = ?", (detection_id,))
    return {"status": "ok", "plate": plate, "message": "تأیید شد و به لیست مجازها اضافه شد"}


@protected.post("/pending-review/{detection_id}/reject")
def reject_detection(detection_id: int):
    with get_db() as conn:
        conn.execute("UPDATE detections SET needs_review = 0 WHERE id = ?", (detection_id,))
    return {"status": "ok", "message": "رد شد"}


# def (نه async def): استنتاج YOLO سنگین است و باید در threadpool اجرا شود تا event loop قفل نشود
@protected.post("/allowed-plates/from-image")
def add_from_image(file: UploadFile = File(...), owner: str = ""):
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        return {"status": "error", "message": "حجم عکس بیش از حد مجاز است"}

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "message": "فایل عکس معتبر نبود"}

    plate_results = detect_plates(img)
    if len(plate_results.boxes) == 0:
        return {"status": "error", "message": "هیچ پلاکی توی عکس پیدا نشد"}

    # بهترین پلاک (بالاترین اطمینان)، نه اولین
    best = max(plate_results.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    plate_crop = img[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return {"status": "error", "message": "ناحیه‌ی پلاک نامعتبر بود"}

    chars_with_conf = read_plate_chars(plate_crop)
    chars_only = [c[1] for c in chars_with_conf]

    # خوانش اشتباه نباید وارد لیست مجازها شود
    if not is_valid_plate_format(chars_only):
        return {"status": "error", "message": "فرمت پلاک خوانده‌شده معتبر نیست؛ عکس واضح‌تری بدهید",
                "read": ''.join(chars_only)}
    avg_conf = sum(c[2] for c in chars_with_conf) / len(chars_with_conf)
    if avg_conf < CONFIDENCE_THRESHOLD:
        return {"status": "error", "message": "اطمینان خوانش پایین است؛ عکس واضح‌تری بدهید",
                "read": ''.join(chars_only), "confidence": round(avg_conf, 3)}

    plate_text = ''.join(chars_only)
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO allowed_plates (plate, owner, created_at)
               VALUES (?, ?, datetime('now','localtime'))""",
            (plate_text, owner),
        )
    return {"status": "ok", "plate": plate_text, "confidence": round(avg_conf, 3),
            "message": "پلاک با موفقیت اضافه شد"}


@protected.get("/unauthorized-last-24h")
def get_unauthorized_last_24h():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT plate, COUNT(*) AS times_seen, MAX(timestamp) AS last_seen
            FROM detections
            WHERE allowed = 0
              AND timestamp >= datetime('now', 'localtime', '-24 hours')
            GROUP BY plate
            ORDER BY last_seen DESC
        """).fetchall()
    return [dict(row) for row in rows]


@protected.get("/stats")
def get_stats():
    today = "date(timestamp) = date('now','localtime')"
    with get_db() as conn:
        today_total = conn.execute(f"SELECT COUNT(*) FROM detections WHERE {today}").fetchone()[0]
        today_allowed = conn.execute(
            f"SELECT COUNT(*) FROM detections WHERE {today} AND allowed = 1").fetchone()[0]
        today_unauthorized = conn.execute(
            f"SELECT COUNT(*) FROM detections WHERE {today} AND allowed = 0 AND needs_review = 0").fetchone()[0]
        pending = conn.execute("SELECT COUNT(*) FROM detections WHERE needs_review = 1").fetchone()[0]
        total_allowed_plates = conn.execute("SELECT COUNT(*) FROM allowed_plates").fetchone()[0]

    return {
        "total_today": today_total,
        "allowed_today": today_allowed,
        "unauthorized_today": today_unauthorized,
        "pending_review": pending,
        "allowed_plates_total": total_allowed_plates,
    }


@protected.get("/detections")
def get_all_detections():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM detections ORDER BY timestamp DESC LIMIT 200").fetchall()
    return [dict(row) for row in rows]


@protected.delete("/allowed-plates/{plate_id}")
def delete_allowed_plate(plate_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM allowed_plates WHERE id = ?", (plate_id,))
    return {"status": "ok", "message": "پلاک حذف شد"}


gate_state = {"isOpen": False, "mode": "AUTO", "lastAction": "سیستم آماده است"}


@protected.post("/gate/toggle")
def toggle_gate(action: str = "", mode: str = ""):
    if mode:
        gate_state["mode"] = mode
    if action == "OPEN":
        gate_state["isOpen"] = True
        gate_state["lastAction"] = "گیت به صورت دستی باز شد"
    elif action == "CLOSE":
        gate_state["isOpen"] = False
        gate_state["lastAction"] = "گیت به صورت دستی بسته شد"
    return {"status": "ok", "gateState": gate_state}


app.include_router(protected)


# ─────────────── استریم زنده ───────────────
def frame_generator():
    while True:
        with frame_lock:
            frame_bytes = latest_frame
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)


@app.get("/video_feed", dependencies=[Depends(require_api_key_or_query)])
def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
