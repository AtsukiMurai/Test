import cv2
from ultralytics import YOLO
import numpy as np
import math

def dist(ex1,ex4):
    sum=0
    for i in range(len(ex1)):
        #print(ex1[i],",",ex4[i])
        tr=(int(ex1[i][0])-int (ex4[i][0]))**2 + (int(ex1[i][1])- ex4[i][1])**2
        tr1 = math.sqrt(tr)
        #print(tr1)
        sum += tr1

    print(sum)    
    return sum
# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")

# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)

img = cv2.imread('ex1.jpg')
results_1 = model("ex1.jpg")
keypoints_1 = results_1[0].keypoints
ex1keys=keypoints_1.data[0]

min_value=10000000
frame_number = 1
# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    
    success, frame = cap.read()

    if success:
        # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
        results = model(frame)
        keypoints= results[0].keypoints
        ex4keys=keypoints.data[0]

        temp = dist(ex1keys,ex4keys)
        
        if temp < min_value:
            min_value = temp
            min_frame = frame_number
            print(frame_number)
        frame_number += 1
    else:
        # ビデオの終わりに到達したらループから抜ける
        break


print(f"フレーム番号は: {min_frame}")
cap.release()

