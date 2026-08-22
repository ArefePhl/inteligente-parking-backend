# import cv2
# from ultralytics import YOLO

# # ---------- نگاشت لاتین → فارسی ----------
# CHAR_MAP = {
#     '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
#     '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
#     'A': 'الف', 'B': 'ب', 'C': 'ث', 'D': 'د', 'EIN': 'ع',
#     'GH': 'ق', 'H': 'ه', 'J': 'ج', 'JHMJ': 'ژ', 'L': 'ل',
#     'M': 'م', 'N': 'ن', 'S': 'س', 'SAD': 'ص', 'SH': 'ش',
#     'T': 'ط', 'TA': 'ت', 'V': 'و', 'Y': 'ی',
# }

# def main():
#     plate_model = YOLO(r'runs\\detect\\train-4\\weights\\best.pt')
#     ocr_model   = YOLO(r'runs\\detect\\ocr-chars-2\\weights\\best.pt')

#     # اول اینو یه بار پرینت کن تا CHAR_MAP رو با اسم‌های واقعی چک کنی
#     print("کلاس‌های مدل OCR:", ocr_model.names)

#     img = cv2.imread('test.jpg')

#     # فاز ۱: پیدا کردن پلاک‌ها
#     plates = plate_model(img, conf=0.5)[0]

#     for i, box in enumerate(plates.boxes):
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         crop = img[y1:y2, x1:x2]

#         # فاز ۲: تشخیص کاراکترها روی برش
#         chars = ocr_model(crop, imgsz=800, conf=0.4)[0]

#         # مرتب‌سازی بر اساس x (چپ به راست)
#         detected = sorted(
#             [(float(b.xyxy[0][0]), ocr_model.names[int(b.cls)]) for b in chars.boxes],
#             key=lambda c: c[0]
#         )

#         plate_text = ' '.join(CHAR_MAP.get(name, name) for _, name in detected)
#         print(f"پلاک {i+1}: {plate_text}")

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.imwrite(f'plate_{i+1}.jpg', crop)

#     cv2.imwrite('result.jpg', img)

# if __name__ == '__main__':
#     main()

import cv2
from ultralytics import YOLO

# ---------- مدل‌ها ----------
plate_model = YOLO(r'runs\\detect\\train-4\\weights\\best.pt')   # تشخیص پلاک
ocr_model   = YOLO(r'runs\\detect\\ocr-chars-2\\weights\\best.pt')  # OCR کاراکترها

# ---------- نگاشت کلاس‌ها به فارسی ----------
CHAR_MAP = {
    '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
    '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹',
    'A': 'الف', 'B': 'ب', 'D': 'د', 'EIN': 'ع', 'GH': 'ق',
    'H': 'ه', 'J': 'ج', 'JHMJ': 'ژ', 'L': 'ل', 'M': 'م',
    'N': 'ن', 'P': 'پ', 'S': 'س', 'SAD': 'ص', 'SH': 'ش',
    'T': 'ت', 'TA': 'ط', 'V': 'و', 'Y': 'ی',
}

def main():
    # ---------- خواندن تصویر ----------
    img_path = r"C:\\Users\\Amir Hossein\\Downloads\\photo_2026-02-27_12-27-00.jpg"   
    img = cv2.imread(img_path)
    assert img is not None, f"تصویر پیدا نشد: {img_path}"

    # ---------- تشخیص پلاک ----------
    plate_results = plate_model(img)[0]

    for i, box in enumerate(plate_results.boxes, start=1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]
        cv2.imwrite(f'plate_{i}.jpg', plate_crop)

        # ---------- OCR روی پلاک برش‌خورده ----------
        ocr_results = ocr_model(plate_crop, conf=0.4)[0]

        # مرتب‌سازی از چپ به راست بر اساس x
        chars = sorted(ocr_results.boxes, key=lambda b: b.xyxy[0][0])

        plate_text = ''
        for c in chars:
            cls_name = ocr_model.names[int(c.cls[0])]
            plate_text += CHAR_MAP.get(cls_name, cls_name) + ' '

        print(f"پلاک {i}: {plate_text.strip()}")

        # رسم کادر و ذخیره
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        chars_fa = [CHAR_MAP.get(ocr_model.names[int(c.cls[0])], '?') for c in chars]
        print(f"پلاک {i}: {format_plate(chars_fa)}")

    cv2.imwrite('result.jpg', img)
def format_plate(chars_list):
    if len(chars_list) == 8:
        return f"{chars_list[0]}{chars_list[1]} {chars_list[2]} {chars_list[3]}{chars_list[4]}{chars_list[5]} - ایران {chars_list[6]}{chars_list[7]}"
    return ' '.join(chars_list)  # اگه تعداد کاراکترها غیرعادی بود
if __name__ == '__main__':
    main()


