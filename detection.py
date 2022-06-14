#!/usr/bin/env python

import numpy as np
import torch
import torch.nn
from torchvision import transforms
import PIL
import cv2
import socket
from ResNet import ResNet50

HOST = 'localhost'
PORT1 = 12315
#PORT2 = 12316

s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s1.connect((HOST,PORT1))
#s2.connect((HOST,PORT2))


Labels = { 
        0 : 'clockwise',
        1 : 'counterclockwise',
        2 : 'down',
        3 : 'empty',
        4 : 'grab',
        5 : 'left',
        6 : 'ok',
        7 : 'open',
        8 : 'right',
        9 : 'up'}

trans = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trans_empty = transforms.Compose([
    transforms.CenterCrop((350,350)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

model = ResNet50(10).to('cuda')
model.load_state_dict(torch.load("/home/kcg/pytorch_ex/checkpoint/resnet50_new4.pth"))
model.eval()

def Webcam_720p():
    cap.set(3,640)
    cap.set(4,480)
    
def argmax(prediction):
    prediction= prediction.cpu()
    prediction= prediction.detach().numpy()
    top_1= np.argmax(prediction, axis=1)
    score= np.amax(prediction)
    score= '{:6f}'.format(score)
    prediction= top_1[0]
    result= Labels[prediction]
    return result,score

def preprocess(image):
    image= PIL.Image.fromarray(image) #Webcam frames are numpy array format Therefore transform back to PIL image
    image= trans(image)
    image= image.float()
    image= image.cuda()
    image= image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                                #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor

def preprocess_empty(image):
    image= PIL.Image.fromarray(image) #Webcam frames are numpy array format Therefore transform back to PIL image
    image= trans_empty(image)
    image= image.float()
    image= image.cuda()
    image= image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                                #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor

#Let's start the real-time classification process!
cap= cv2.VideoCapture(0) #Set the webcam
Webcam_720p()
cnt= 1
show_score= 0
show_res= 'Nothing'
sequence= 0

# 'skin'의 범위 값 설정 
lower = np.array([0, 133, 77], dtype = "uint8")
upper = np.array([255, 173, 127], dtype = "uint8")

while True:
    ret, frame= cap.read()                                                                  #Capture each frame
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))                         #SkinMask의 iterations를 두번 반복(잡힌 범위 주변 margin이 뚱뚱해진다)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 1)                                 #SkinMask의 iterations를 두번 반복(잡힌 범위 주변 margin이 뚱뚱해진다)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    contours, _ = cv2.findContours(skinMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]                             #시작점의 X,y좌표와 너비와 높이를 배열에 저장.
    
    if contours == []:
        image = frame[0:, 0:]
        image_data = preprocess_empty(image)
        prediction = model(image_data)
        result,score= argmax(prediction)
        show_res= result
        show_score= float(score)
        cnt += 1

        print(int(cnt), " : ", show_res)
        
        if cnt % 20 == 0:
            if(show_res == 'empty'):
                    commend = show_res
                    #s1.send(commend.encode())
                    #s2.send(commend.encode())
        
    else:    
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
            
            image = frame[VU:VD, HL:HR]         
            cv2.rectangle(frame, (HL, VU), (HR, VD), (0, 0, 255), 3)
            image_data = preprocess(image)
            prediction = model(image_data)
            result,score= argmax(prediction)
            show_res= result
            show_score= float(score)
            cnt += 1
    
            print(int(cnt), " : ", show_res)
            
            if cnt % 20 == 0:    
                if(show_res == 'clockwise'):
                    commend = show_res
                    #s1.send(commend.encode())
                
                elif(show_res == 'counterclockwise'):
                    commend = show_res
                    #s1.send(commend.encode())
                    
                elif(show_res == 'down'):
                    commend = show_res
                    #s1.send(commend.encode())    
                
                elif(show_res == 'grab'):
                    commend = show_res
                    #s2.send(commend.encode())
                    
                elif(show_res == 'left'):
                    commend = show_res
                    #s1.send(commend.encode())
                    
                elif(show_res == 'ok'):
                    commend = show_res
                    #s1.send(commend.encode())
                    #s2.send(commend.encode())
                    
                elif(show_res == 'open'):
                    commend = show_res
                    #s2.send(commend.encode())                
                    
                elif(show_res == 'right'):
                    commend = show_res
                    #s1.send(commend.encode())
                    
                elif(show_res == 'up'):
                    commend = show_res
                    #s1.send(commend.encode())
                
    
    cv2.putText(frame, '%s' %(show_res),(300,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(frame, '(score= %.5f)' %(show_score), (300,250), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
    cv2.imshow("Hand Gesture Recognizer", frame)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        s1.close()
        #s2.close()
        break
    
cap.release()
cv2.destroyWindow("Hand Gesture Recognizer")
