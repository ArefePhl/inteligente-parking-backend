"""Run the recognizer on images and save annotated demo panels.

Usage:
    python demo.py IMAGE_OR_DIR [IMAGE_OR_DIR ...] --out docs/demo

Each output panel = annotated frame (left) + upscaled plate crop with character
boxes (right) + the recognised text, so a reader can see both pipeline stages.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from recognizer import BASE_DIR, PlateRecognizer
from viz import draw_persian_text, to_fa_digits

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GREEN, YELLOW, GRAY = (60, 200, 60), (0, 200, 255), (200, 200, 200)


def collect_images(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    return files


def make_panel(img, results, height=480):
    frame = img.copy()
    crop_panels = []
    for r in results[:1]:  # panel shows the most confident plate
        x1, y1, x2, y2 = r.box
        color = GREEN if r.valid else YELLOW
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{to_fa_digits(r.text)}" if r.text else "؟"
        frame = draw_persian_text(frame, label, (x1, max(0, y1 - 34)), color, 26)

        crop = img[max(0, y1):y2, max(0, x1):x2].copy()
        scale = 420 / max(crop.shape[1], 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for (bx1, by1, bx2, by2) in r.char_boxes:
            cv2.rectangle(crop, (int(bx1 * scale), int(by1 * scale)), (int(bx2 * scale), int(by2 * scale)),
                          GREEN if r.valid else YELLOW, 1)
        crop_panels.append((crop, r))

    h = frame.shape[0]
    side = np.full((h, 460, 3), 30, np.uint8)
    if crop_panels:
        crop, r = crop_panels[0]
        ch, cw = crop.shape[:2]
        side[10:10 + ch, 20:20 + cw] = crop[: h - 130, :440]
        status = "valid format" if r.valid else "INVALID format"
        side = draw_persian_text(side, to_fa_digits(r.text) or "-", (20, h - 110), GREEN if r.valid else YELLOW, 34)
        cv2.putText(side, f"{status} | det {r.det_conf:.2f} | ocr {r.ocr_conf:.2f}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1, cv2.LINE_AA)
    else:
        cv2.putText(side, "no plate detected", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GRAY, 1, cv2.LINE_AA)
    return np.hstack([frame, side])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="docs/demo")
    ap.add_argument("--plate-model", default=os.getenv("PLATE_MODEL_PATH", str(BASE_DIR / "models/plate/best.pt")))
    ap.add_argument("--ocr-model", default=os.getenv("OCR_MODEL_PATH", str(BASE_DIR / "models/ocr/best.pt")))
    args = ap.parse_args()

    rec = PlateRecognizer(args.plate_model, args.ocr_model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for f in collect_images(args.inputs):
        img = cv2.imread(str(f))
        if img is None:
            print("skip (unreadable):", f)
            continue
        results = rec.recognize(img)
        panel = make_panel(img, results)
        cv2.imwrite(str(out / f"{f.stem}_demo.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
        top = results[0] if results else None
        summary.append({
            "image": f.name,
            "plates_found": len(results),
            "text": top.text if top else None,
            "valid_format": top.valid if top else None,
            "det_conf": round(top.det_conf, 3) if top else None,
            "ocr_conf": round(top.ocr_conf, 3) if top else None,
        })
        print(f"{f.name:28s} plates={len(results)} text={top.text if top else '-'} valid={top.valid if top else '-'}")
    (out / "demo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
