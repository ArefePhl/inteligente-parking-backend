from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"L:\\nteligence parking\\Iranian Plate.v3-ilpv3.yolov8\\data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0
    )
