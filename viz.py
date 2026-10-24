"""Drawing helpers (Persian text on OpenCV images)."""
import os

import arabic_reshaper
import cv2
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    os.getenv("FONT_PATH", ""),
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
EN_TO_FA_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')


def to_fa_digits(text: str) -> str:
    return ''.join(c.translate(EN_TO_FA_DIGITS) if c.isdigit() else c for c in text)


def _font(size):
    for path in _FONT_CANDIDATES:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_persian_text(frame, text, pos, color=(0, 255, 0), size=26):
    """Draw RTL Persian text onto a BGR frame; returns the new frame."""
    bidi_text = str(get_display(arabic_reshaper.reshape(text)))
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img_pil).text(pos, bidi_text, font=_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
