# import cv2

# camera = cv2.VideoCapture(0)  # 0 یعنی اولین وبکم سیستم

# while True:
#     ok, frame = camera.read()   # یه عکس از دوربین بگیر
#     if not ok:
#         break
#     cv2.imshow("Doorbin", frame)  # نشونش بده
#     if cv2.waitKey(1) == ord('q'):  # با دکمه q ببند
#         break

# camera.release()
# cv2.destroyAllWindows()


# import cv2
# print(cv2.__version__)
# print(cv2.getBuildInformation())


# import torch
# print(torch.cuda.is_available())        # باید True باشه
# print(torch.cuda.get_device_name(0))    # باید RTX 3080 Ti رو نشون بده



# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


from ultralytics import YOLO

def main():
    model = YOLO(r"C:\\Users\blueNoteBook\\OneDrive\\دسکتاپ\\inteligence parking\\runs\\detect\\ocr-chars-2\\weights\\best.pt")
    metrics = model.val(
        data=r"C:\\Users\\blueNoteBook\\OneDrive\\دسکتاپ\\inteligence parking\\#2 data set _ chars\\#2 data set _ chars\\data.yaml"
    )
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == '__main__':
    main()
