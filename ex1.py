import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8x-pose.pt")
img = cv2.imread('ex1.jpg')

results = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)

skeleton = np.array([
			[16,14],[14,12],[15,13],[13,11],
            [12,11],[12,6],[11,5],[6,5],
            [6,8],[5,7],[8,10],[7,9]
            ])
    
for i in range(len(skeleton)):
    cv2.line(img,
    (int(keypoints.data[0][skeleton[i][0]][0]), int(keypoints.data[0][skeleton[i][0]][1])),
    (int(keypoints.data[0][skeleton[i][1]][0]), int(keypoints.data[0][skeleton[i][1]][1])),
    (0, 0, 255),thickness=2
    )
    

for i in range(keypoints.data[0].size(0)-5):
    cv2.circle(
        img,(int(keypoints.data[0][i+5][0]), int(keypoints.data[0][i+5][1])),6,(0,225,255),-1
    )


cv2.imwrite('ex1_output.jpg',img)
cv2.imshow('sample',img)

