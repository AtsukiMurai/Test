import cv2
from ultralytics import YOLO
import numpy as np
import math

# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")

# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)


# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        # フレームでYOLOv8トラッキングを実行し、フレーム間でトラックを永続化
        results = model(frame)
        keypoints = results[0].keypoints
        #print(keypoints.data)

        skeleton = np.array([
                    [16,14],[14,12],[15,13],[13,11],
                    [12,11],[12,6],[11,5],[6,5],
                    [6,8],[5,7],[8,10],[7,9]
                    ])
        
        a =(int(keypoints.data[0][6][0])-int(keypoints.data[0][8][0]))**2 
        b =(int(keypoints.data[0][6][1])-int(keypoints.data[0][8][1]))**2
        c =  math.sqrt(a+b)
        print(c)##右腕

        a_1 =(int(keypoints.data[0][6][0])-int(keypoints.data[0][12][0]))**2 
        b_1 =(int(keypoints.data[0][6][1])-int(keypoints.data[0][12][1]))**2
        c_1 =  math.sqrt(a_1+b_1)##腹横

        a_2 =(int(keypoints.data[0][12][0])-int(keypoints.data[0][8][0]))**2 
        b_2 =(int(keypoints.data[0][12][1])-int(keypoints.data[0][8][1]))**2
        c_2 =  math.sqrt(a_2+b_2)##右肘から右腰の距離

        cos6=(c_1**2 +c**2 -c_2**2 )/(2*c*c_1)
        r= math.acos(cos6)
        r6= math.degrees(r)
        print(r6)

        for i in range(len(skeleton)):
            cv2.line(frame,
            (int(keypoints.data[0][skeleton[i][0]][0]), int(keypoints.data[0][skeleton[i][0]][1])),
            (int(keypoints.data[0][skeleton[i][1]][0]), int(keypoints.data[0][skeleton[i][1]][1])),
            (255, 0, 0),thickness=2
            )
            
        if(80<=r6 and r6<=100):
            cv2.line(frame,
            (int(keypoints.data[0][skeleton[8][0]][0]), int(keypoints.data[0][skeleton[8][0]][1])),
            (int(keypoints.data[0][skeleton[8][1]][0]), int(keypoints.data[0][skeleton[8][1]][1])),
            (0, 0, 255),thickness=2
            )
            cv2.line(frame,
            (int(keypoints.data[0][skeleton[10][0]][0]), int(keypoints.data[0][skeleton[10][0]][1])),
            (int(keypoints.data[0][skeleton[10][1]][0]), int(keypoints.data[0][skeleton[10][1]][1])),
            (0, 0, 255),thickness=2
            )


        for i in range(keypoints.data[0].size(0)-5):
            cv2.circle(
                frame,(int(keypoints.data[0][i+5][0]), int(keypoints.data[0][i+5][1])),6,(0,225,255),-1
            )

        # 注釈付きのフレームを表示
        cv2.imshow("YOLOv8output", frame)
        cv2.waitKey(20) 

        # 'q'が押されたらループから抜ける
        if 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break

# ビデオキャプチャオブジェクトを解放し、表示ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
