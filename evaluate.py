"""Measure end-to-end plate recognition accuracy on a labelled test set.

Ground truth, either:
  --csv labels.csv --images DIR      CSV columns: filename,plate   (plate like 42د61613)
  --data DIR                         YOLO layout: DIR/images/*.jpg + DIR/labels/*.txt using the
                                     OCR model's class ids (text = boxes sorted by x-centre)

Modes:
  --mode e2e   (default) images are full photos: plate detector → OCR  (the number to report)
  --mode ocr   images are already plate crops: OCR only                (isolates the OCR stage)

Outputs (in --out): results.json, results.md, errors.csv  [+ error panels with --save-errors N]
"""
import argparse
import csv
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from recognizer import BASE_DIR, CLASS_MAP, PlateRecognizer, is_valid_plate_format

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
ALEF_FORMS = 'الف'


# ───────────────────────── text helpers ─────────────────────────
def tokenize(text: str) -> list:
    """'42د61613' / '۴۲ الف ۶۱۶ ۱۳' → ['4','2','د','6',...]; 'الف' is one token."""
    text = text.translate(PERSIAN_DIGITS).replace('ي', 'ی').replace('ك', 'ک')
    text = ''.join(ch for ch in text if not ch.isspace() and ch not in '-_|.')
    tokens, i = [], 0
    while i < len(text):
        if text.startswith(ALEF_FORMS, i):
            tokens.append(ALEF_FORMS)
            i += len(ALEF_FORMS)
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def edit_ops(ref: list, hyp: list):
    """Levenshtein on token lists → (distance, [(ref_tok, hyp_tok) substitutions])."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    subs, i, j = [], n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j - 1] + 1:
            subs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return d[n][m], subs


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# ───────────────────────── ground truth loading ─────────────────────────
def load_csv(csv_path: Path, images_dir: Path):
    items = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            items.append((images_dir / row["filename"].strip(), tokenize(row["plate"])))
    return items


def load_yolo(data_dir: Path, ocr_model_names: dict):
    images_dir, labels_dir = data_dir / "images", data_dir / "labels"
    yaml_path = data_dir / "data.yaml"
    if yaml_path.exists():  # refuse to evaluate with a mismatched class order
        import yaml
        names = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("names")
        names = list(names.values()) if isinstance(names, dict) else list(names or [])
        model_names = [ocr_model_names[i] for i in sorted(ocr_model_names)]
        if names and names != model_names:
            raise SystemExit(f"data.yaml class names differ from the OCR model's classes:\n"
                             f"  data : {names}\n  model: {model_names}")
    items = []
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT):
        lab = labels_dir / (img.stem + ".txt")
        if not lab.exists():
            continue
        boxes = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.append((float(parts[1]), CLASS_MAP.get(int(float(parts[0])), '?')))
        boxes.sort()
        items.append((img, [c for _, c in boxes]))
    return items


# ───────────────────────── evaluation ─────────────────────────
def percentile_ms(xs, q):
    return float(np.percentile(xs, q)) * 1000 if xs else 0.0


def evaluate(items, rec, mode):
    rows, t_det, t_ocr, t_tot = [], [], [], []
    warm = np.zeros((640, 640, 3), np.uint8)  # first call includes model/JIT warm-up; keep it out of timings
    rec.detect_plates(warm)
    rec.read_chars(warm)
    for path, gt in items:
        img = cv2.imread(str(path))
        if img is None:
            print("skip (unreadable):", path)
            continue
        t0 = time.perf_counter()
        det_conf, detected = None, True
        if mode == "e2e":
            t1 = time.perf_counter()
            res = rec.detect_plates(img)
            t_det.append(time.perf_counter() - t1)
            if len(res.boxes) == 0:
                chars, detected = [], False
            else:
                best = max(res.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                det_conf = float(best.conf[0])
                crop = img[max(0, y1):y2, max(0, x1):x2]
                t2 = time.perf_counter()
                chars = rec.read_chars(crop) if crop.size else []
                t_ocr.append(time.perf_counter() - t2)
        else:
            t2 = time.perf_counter()
            chars = rec.read_chars(img)
            t_ocr.append(time.perf_counter() - t2)
        t_tot.append(time.perf_counter() - t0)

        pred = [c[1] for c in chars]
        ocr_conf = sum(c[2] for c in chars) / len(chars) if chars else 0.0
        dist, subs = edit_ops(gt, pred)
        rows.append({
            "image": path.name, "gt": ''.join(gt), "pred": ''.join(pred), "detected": detected,
            "valid_format": is_valid_plate_format(pred), "det_conf": det_conf, "ocr_conf": ocr_conf,
            "exact": pred == gt, "edit_distance": dist, "gt_len": len(gt), "subs": subs,
        })
    return rows, t_det, t_ocr, t_tot


def summarise(rows, t_det, t_ocr, t_tot, mode, conf_threshold):
    n = len(rows)
    detected = [r for r in rows if r["detected"]]
    exact = sum(r["exact"] for r in rows)
    lo, hi = wilson(exact, n)
    tot_gt = sum(r["gt_len"] for r in rows)
    out = {
        "mode": mode, "n_images": n,
        "plate_detected_rate": len(detected) / n if n else 0,
        "exact_match_all": exact / n if n else 0,
        "exact_match_ci95": [lo, hi],
        "exact_match_given_detected": (sum(r["exact"] for r in detected) / len(detected)) if detected else 0,
        "cer": sum(r["edit_distance"] for r in rows) / tot_gt if tot_gt else 0,
        "format_valid_rate": sum(r["valid_format"] for r in rows) / n if n else 0,
    }
    # Operating point = what the API does: accept only valid-format reads with conf ≥ threshold
    def accepted(th):
        return [r for r in rows if r["valid_format"] and r["ocr_conf"] >= th]
    curve = []
    for th in sorted({0.0, 0.5, 0.6, 0.7, 0.8, 0.9, conf_threshold}):
        acc = accepted(th)
        curve.append({
            "threshold": th, "coverage": len(acc) / n if n else 0,
            "accuracy_on_accepted": (sum(r["exact"] for r in acc) / len(acc)) if acc else None,
            "accepted": len(acc),
        })
    out["accept_curve"] = curve
    out["operating_point"] = next(c for c in curve if c["threshold"] == conf_threshold)
    confusions = Counter(s for r in rows for s in r["subs"])
    out["top_confusions"] = [{"gt": a, "pred": b, "count": c} for (a, b), c in confusions.most_common(10)]
    out["latency_ms"] = {
        "detect_mean": float(np.mean(t_det)) * 1000 if t_det else None,
        "ocr_mean": float(np.mean(t_ocr)) * 1000 if t_ocr else None,
        "total_p50": percentile_ms(t_tot, 50), "total_p95": percentile_ms(t_tot, 95),
    }
    try:
        import torch
        dev = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "unknown"
    out["hardware"] = {"device": dev, "cpu": platform.processor() or platform.machine(), "os": platform.platform()}
    return out


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(s):
    lo, hi = s["exact_match_ci95"]
    lines = [
        f"# Evaluation results ({s['mode']} mode, n = {s['n_images']})", "",
        "| Metric | Value |", "|---|---|",
        f"| Plate detected | {pct(s['plate_detected_rate'])} |" if s["mode"] == "e2e" else "| Plate detected | n/a (crops) |",
        f"| **Exact plate match (all images)** | **{pct(s['exact_match_all'])}** (95% CI {pct(lo)}–{pct(hi)}) |",
        f"| Exact match, given detected | {pct(s['exact_match_given_detected'])} |",
        f"| Character error rate (CER) | {pct(s['cer'])} |",
        f"| Output has valid plate format | {pct(s['format_valid_rate'])} |", "",
        "## Accept / review trade-off",
        "Reads are *accepted* when the format is valid and mean character confidence ≥ threshold; "
        "everything else goes to human review.", "",
        "| Confidence threshold | Coverage (auto-accepted) | Accuracy on accepted |", "|---|---|---|",
    ]
    for c in s["accept_curve"]:
        lines.append(f"| {c['threshold']:.2f} | {pct(c['coverage'])} ({c['accepted']}) | {pct(c['accuracy_on_accepted'])} |")
    if s["top_confusions"]:
        lines += ["", "## Most frequent character confusions", "", "| Ground truth | Predicted | Count |", "|---|---|---|"]
        lines += [f"| {c['gt']} | {c['pred']} | {c['count']} |" for c in s["top_confusions"]]
    L, h = s["latency_ms"], s["hardware"]
    lines += ["", "## Speed", "",
              f"Device: `{h['device']}`. Mean detect {L['detect_mean'] and round(L['detect_mean'], 1)} ms, "
              f"mean OCR {L['ocr_mean'] and round(L['ocr_mean'], 1)} ms; per image p50 {L['total_p50']:.1f} ms, "
              f"p95 {L['total_p95']:.1f} ms."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--images")
    ap.add_argument("--data")
    ap.add_argument("--mode", choices=["e2e", "ocr"], default="e2e")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--conf-threshold", type=float, default=0.6)
    ap.add_argument("--save-errors", type=int, default=0, help="save up to N annotated failure panels")
    ap.add_argument("--conf-char", type=float, default=0.4, help="reader confidence threshold")
    ap.add_argument("--conf-plate", type=float, default=0.5, help="detector confidence threshold")
    ap.add_argument("--ocr-imgsz", type=int, default=800, help="reader inference size")
    ap.add_argument("--plate-model", default=str(BASE_DIR / "models/plate/best.pt"))
    ap.add_argument("--ocr-model", default=str(BASE_DIR / "models/ocr/best.pt"))
    args = ap.parse_args()

    rec = PlateRecognizer(args.plate_model, args.ocr_model, conf_plate=args.conf_plate,
                           conf_char=args.conf_char, ocr_imgsz=args.ocr_imgsz)
    if args.csv:
        if not args.images:
            raise SystemExit("--csv needs --images")
        items = load_csv(Path(args.csv), Path(args.images))
    elif args.data:
        items = load_yolo(Path(args.data), rec.ocr_model.names)
    else:
        raise SystemExit("give --csv/--images or --data")
    if not items:
        raise SystemExit("no labelled images found")

    rows, t_det, t_ocr, t_tot = evaluate(items, rec, args.mode)
    summary = summarise(rows, t_det, t_ocr, t_tot, args.mode, args.conf_threshold)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "results.md").write_text(to_markdown(summary), encoding="utf-8")
    with open(out / "errors.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "prediction", "valid_format", "det_conf", "ocr_conf", "edit_distance"])
        for r in rows:
            if not r["exact"]:
                w.writerow([r["image"], r["gt"], r["pred"], r["valid_format"],
                            None if r["det_conf"] is None else round(r["det_conf"], 3),
                            round(r["ocr_conf"], 3), r["edit_distance"]])
    if args.save_errors and args.mode == "e2e":
        from demo import make_panel
        paths = {p.name: p for p, _ in items}
        for r in [r for r in rows if not r["exact"]][: args.save_errors]:
            img = cv2.imread(str(paths[r["image"]]))
            cv2.imwrite(str(out / f"error_{Path(r['image']).stem}.jpg"), make_panel(img, rec.recognize(img)))
    print(to_markdown(summary))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
