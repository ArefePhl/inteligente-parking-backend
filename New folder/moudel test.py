from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"C:\\Users\\Amir Hossein\\Desktop\\inteligence parking\\runs\detect\\train-4\\weights\\best.pt")
    
    results = model.predict(
        source=r"C:\\Users\\Amir Hossein\\Downloads\\images.png",
        save=True,
        conf=0.5
    )
