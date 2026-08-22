# from ultralytics import YOLO

# # مدل پایه (nano برای سرعت، یا yolov8s.pt برای دقت بیشتر)
# model = YOLO('yolov8n.pt')

# # آموزش روی دیتاست persian-car-numbers v3
# results = model.train(
#     data=r"C:\\Users\\Amir Hossein\\Desktop\\inteligence parking\\Persian-Car-Numbers.v3i.yolov8\\data.yaml",  # مسیر data.yaml خودت
#     epochs=100,          # کاراکترها کوچیکن، اپک بیشتر بهتره
#     imgsz=640,
#     batch=16,
#     patience=20,         # اگه بهبود نداشت زودتر متوقف می‌شه
#     name='ocr-chars',    # خروجی: runs/detect/ocr-chars
#     device=0             # GPU (اگه نداری: device='cpu')
# )

# # ارزیابی نهایی
# metrics = model.val()
# print(f"mAP50: {metrics.box.map50:.3f}")


from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=r'C:\\Users\\Amir Hossein\\Desktop\\inteligence parking\\Persian-Car-Numbers.v3i.yolov8\\data.yaml',
        epochs=100,
        imgsz=800,
        batch=16,
        patience=20,
        name='ocr-chars',
        device=0
    )

if __name__ == '__main__':
    main()
