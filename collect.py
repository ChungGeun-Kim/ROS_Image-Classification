import numpy as np
import cv2
import os

def Webcam_720p():
    cap.set(3,640)
    cap.set(4,480)

cap = cv2.VideoCapture(2) #Set the webcam
Webcam_720p()
path = '/home/iracpc/pytorch_ex/data/hand_data1/dataset/train/up'
count = 1

# 'skin'의 범위 값 설정 
lower = np.array([0, 133, 77], dtype = "uint8")
upper = np.array([255, 173, 127], dtype = "uint8")

while True:
    ret, frame= cap.read() #Capture each frame
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # 경계선 찾아주는 곳, 타원 모양으로 매트리스 생성
    skinMask = cv2.dilate(skinMask, kernel, iterations = 1) #SkinMask의 iterations를 두번 반복(잡힌 범위 주변 margin이 뚱뚱해진다)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    contours, _ = cv2.findContours(skinMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #시작점의 X,y좌표와 너비와 높이를 배열에 저장.
    rects = [cv2.boundingRect(contour) for contour in contours]
    
    for rect in rects:
        
        if (rect[2] * rect[3] < 19200) or (rect[2] * rect[3] > 76800):
            continue
        
        val_H = int((rect[0] + rect[2]) / 8)
        val_V = int((rect[1] + rect[3]) / 8)
        
        HL = (rect[0] - val_H)
        HR = (rect[0] + rect[2] + val_H)
        VU = (rect[1] - val_V)
        VD = (rect[1] + rect[3] + val_V)
        
        if HL <= 0:
            HL = 0
        if HR >= 640:
            HR = 640
        if VU <= 0:
            VU = 0
        if VD >= 480:
            VD = 480
            
        if count % 49 == 0:
            image = frame[VU:VD, HL:HR] 
            cv2.imwrite(os.path.join(path,"frame1%d.png" % int(count/49)), image)
            print("image saved!")
            
        cv2.rectangle(frame, (HL, VU), (HR, VD), (0, 0, 255), 3)
        count += 1
            
    cv2.imshow("Hand Gesture Recognizer", frame)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
    
cap.release()
cv2.destroyWindow("Hand Gesture Recognizer")
