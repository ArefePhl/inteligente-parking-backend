from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # مدل پایه
model.train(data="E:\\Projects\\inteligence parking\\Iranian Plate.v3-ilpv3.yolov8\\data.yaml", epochs=50, imgsz=640)
