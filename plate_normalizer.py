

import re


class InvalidPlateFormatError(Exception):
    """
    وقتی پلاک بعد از تمیزکاری هم فرمت درستی نداره، این خطا داده می‌شه.
    پیام‌ش قابل‌فهمه، نه یه traceback گنگ.
    """
    pass


# ─────────────── جدول تبدیل ارقام فارسی/عربی به انگلیسی ───────────────
_DIGIT_MAP = str.maketrans({
    # ارقام فارسی
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    # ارقام عربی (شکل‌شون با فارسی فرق داره، مثلاً ٤ عربی vs ۴ فارسی)
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
})

# ─────────────── جدول یکسان‌سازی حروف عربی به معادل فارسی‌شون ───────────────
# چون توی صفحه‌کلید عربی بعضی حروف با فارسی شبیه ولی کدشون فرق داره
_LETTER_MAP = str.maketrans({
    'ي': 'ی',   # یای عربی -> یای فارسی
    'ك': 'ک',   # کاف عربی -> کاف فارسی
    'ة': 'ه',   # تای مربوط عربی -> های فارسی (برای احتیاط)
})

# ─────────────── کاراکترهایی که باید کلاً حذف بشن ───────────────
# فاصله، نیم‌فاصله (ZWNJ)، خط تیره، خط زیر خط تیره فارسی
_CHARS_TO_STRIP = re.compile(r'[\s\u200c\u200e\u200f\-_]+')

# حروف مجازی که می‌تونن توی یه پلاک ایرانی وسط باشن (طبق CLASS_MAP خود پروژه)
_VALID_PLATE_LETTERS = set('الفبثدعقهجژلمنپسصشتطویدیپلماتستادگذرموقتکشاورزیارتشتشریفاتسپاهوزارتدفاع')


def normalize_plate(raw_text: str) -> str:
    """
    یه متن خام پلاک می‌گیره و نسخه‌ی یکدست‌شده‌شو برمی‌گردونه.

    مراحل:
    1. حذف فاصله/نیم‌فاصله/خط‌تیره
    2. تبدیل ارقام فارسی و عربی به انگلیسی
    3. یکسان‌سازی حروف عربی به معادل فارسی
    4. بررسی فرمت نهایی (۲ رقم + ۱ حرف + ۵ رقم = ۸ کاراکتر)

    اگه فرمت نهایی درست نبود، InvalidPlateFormatError با پیام قابل‌فهم می‌ده.

    مثال:
        normalize_plate("۵۹ ب ۶۹۴۶۷")  ->  "59ب69467"
    """
    if raw_text is None:
        raise InvalidPlateFormatError("متن پلاک خالیه")

    # قدم ۱: حذف فاصله‌ها و کاراکترهای جداکننده
    text = _CHARS_TO_STRIP.sub('', raw_text.strip())

    if not text:
        raise InvalidPlateFormatError("بعد از حذف فاصله‌ها، متنی برای پلاک باقی نموند")

    # قدم ۲: تبدیل ارقام فارسی/عربی به انگلیسی
    text = text.translate(_DIGIT_MAP)

    # قدم ۳: یکسان‌سازی حروف عربی به فارسی
    text = text.translate(_LETTER_MAP)

    # قدم ۴: بررسی فرمت نهایی
    _validate_format(text)

    return text


def _validate_format(text: str) -> None:
    """
    فرمت استاندارد پلاک ایرانی: ۲ رقم + ۱ حرف + ۵ رقم = ۸ کاراکتر
    (همون فرمتی که توی admin_api.py و cam.py هم چک می‌شه)
    """
    chars = list(text)

    if len(chars) != 8:
        raise InvalidPlateFormatError(
            f"طول پلاک باید ۸ کاراکتر باشه (۲ رقم + ۱ حرف + ۵ رقم)، "
            f"ولی بعد از تمیزکاری {len(chars)} کاراکتر شد: {text!r}"
        )

    digit_positions = [c.isdigit() for c in chars]

    # جایگاه ۰ و ۱ باید رقم باشن
    if not (digit_positions[0] and digit_positions[1]):
        raise InvalidPlateFormatError(
            f"دو کاراکتر اول پلاک باید رقم باشن: {text!r}"
        )

    # جایگاه ۲ باید حرف باشه، نه رقم
    if digit_positions[2]:
        raise InvalidPlateFormatError(
            f"کاراکتر سوم پلاک باید حرف باشه، نه رقم: {text!r}"
        )

    letter = chars[2]
    if letter not in _VALID_PLATE_LETTERS:
        raise InvalidPlateFormatError(
            f"حرف پلاک ({letter!r}) جزو حروف مجاز پلاک ایرانی نیست: {text!r}"
        )

    # جایگاه‌های ۳ تا ۷ باید همه رقم باشن
    if not all(digit_positions[3:8]):
        raise InvalidPlateFormatError(
            f"پنج کاراکتر آخر پلاک باید همه رقم باشن: {text!r}"
        )


def is_normalizable(raw_text: str) -> bool:
    """
    یه کمک‌کننده‌ی ساده: بدون پرت کردن خطا، فقط می‌گه آیا این متن
    قابل تبدیل به یه پلاک معتبره یا نه (True/False).
    برای جاهایی مفیده که فقط می‌خوایم چک کنیم، نه اینکه خطا بگیریم.
    """
    try:
        normalize_plate(raw_text)
        return True
    except InvalidPlateFormatError:
        return False