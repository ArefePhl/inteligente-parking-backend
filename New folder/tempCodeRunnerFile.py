import cv2

camera = cv2.VideoCapture(0)  # 0 یعنی اولین وبکم سیستم

while True:
    ok, frame = camera.read()   # یه عکس از دوربین بگیر
    if not ok:
        break
    cv2.imshow("Doorbin", frame)  # نشونش بده
    if cv2.waitKey(1) == ord('q'):  # با دکمه q ببند
        break

camera.release()
cv2.destroyAllWindows()
