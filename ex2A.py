import cv2
from ultralytics import YOLO

# YOLOモデルのロード
model = YOLO("yolov8x.pt")

# 画像の読み込み
img = cv2.imread('ex2.jpg')


results = model("ex2.jpg", save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes  


max_area = 0
max_box = None
for box in boxes:#boxes をbox に
    x1 = int(box.data[0][0])
    y1 = int(box.data[0][1])
    x2 = int(box.data[0][2])
    y2 = int(box.data[0][3])

    area = (x2 - x1) * (y2 - y1)  
    if area > max_area:
        max_area = area
        max_box = (x1, y1, x2, y2)

x1, y1, x2, y2 = max_box
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2) 

cv2.imwrite('ex2A_output.jpg', img)
cv2.imshow('sample', img)
cv2.destroyAllWindows()
