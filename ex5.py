import cv2
import numpy as np
from ultralytics import YOLO

# YOLOモデルの読み込み
model = YOLO("yolov8x.pt")
img = cv2.imread('ex2.jpg')

results = model("ex2.jpg", save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes

cv2.waitKey(0)
cv2.destroyAllWindows()
