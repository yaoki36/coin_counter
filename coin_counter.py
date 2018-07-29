import cv2
import numpy as np
import copy
import random
import sys
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

while(True):
    ret, frame = cap.read()
    if not ret: continue
    h,w,c=frame.shape
    gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    th1, th2 = 100, 170
    gray[gray <= th1] = 0
    gray[gray >= th2] = 255
    gray[ np.where((gray > th1) & (gray < th2)) ] = 128
    
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    # ラベリング処理
    label = cv2.connectedComponentsWithStats(opening)
    
    # ラベリング結果書き出し用に二値画像をカラー変換
    color_src01 = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    
    # オブジェクト情報を項目別に抽出
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    center = center.astype(np.int)
    
    dif_max = np.zeros(2, dtype = int)
    dif = np.zeros(n+1, dtype = int)
    
    area = data[:,4]
    
    #面積の大きい順にソート
    area = np.sort(area)
    
    #コインの大きさのしきい値設定
    dif_max = np.zeros(2, dtype=int)
    dif = np.zeros(n, dtype=int)
    ave = np.zeros(2, dtype=int)
    for i in range(n-1):
        dif[i] = area[i+1] - area[i]
        if(dif_max[0] < dif[i]):
            dif_max[0], dif_max[1] = dif[i], dif_max[0]
            ave[0], ave[1] = (area[i+1]+area[i])/2, ave[0]
        elif(dif_max[1] < dif[i]):
            dif_max[1] = dif[i]
            ave[1] = (area[i+1]+area[i])/2
    if(ave[0] < ave[1]):
        ave[0], ave[1] = ave[1], ave[0]
    
    #コイン判別に利用する変数の初期化
    five_hundred = 0
    one_hundred = 0
    fifty = 0
    ten = 0
    five = 0
    one = 0
    
    # オブジェクト情報を利用してラベリング結果を画面に表示
    for i in range(n):
      
      # 各オブジェクトの外接矩形を赤枠で表示
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255))
        
        #コインの判別
        if data[i][4] > ave[0]:
            five_hundred += 1
            cv2.putText(frame, "500", (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        elif data[i][4] < ave[1]:
            if (color_src01[center[i][1]][center[i][0]]).any() == 0:
                fifty += 1
                cv2.putText(frame, "50", (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            else:
                one += 1
                cv2.putText(frame, "1", (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
        else:
            if (color_src01[center[i][1]][center[i][0]] == 255).all():
                one_hundred += 1
                cv2.putText(frame, "100", (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            elif (color_src01[center[i][1]][center[i][0]]).any() == 0:
                five += 1
                cv2.putText(frame, "5", (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            else:
                ten += 1
                cv2.putText(frame, "10", (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    total = 500 * five_hundred + 100 * one_hundred + 50 * fifty + 10 * ten + 5 * five + one
    cv2.putText(frame, "total:" + str(total), (260, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    # 結果の表示
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
