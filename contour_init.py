import numpy as np
import cv2
import torch
import torch.nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
import PIL

def preprocess(image):
    image= PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    image= trans(image)
    image= image.float()
    #image= Variable(image, requires_autograd=True)
    image= image.cuda()
    image= image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                                #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor

def imshow(img):
    img = img / 2 + 0.6
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img,(1,2,0)))
    
#  HSV pixel의 lower과 upper 경계값 정의/스킨 색 경계값 설정
# 'skin'의 범위 값 설정 
lower = np.array([0, 133, 77], dtype = "uint8")
upper = np.array([255, 173, 127], dtype = "uint8")

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

camera = cv2.VideoCapture(0)

while True:

    (grabbed, frame) = camera.read()

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinMask = cv2.inRange(converted, lower, upper)
 
    # 경계선 찾아주는애
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))#타원 모양으로 매트리스 생성
    skinMask = cv2.dilate(skinMask, kernel, iterations = 1)#SkinMask의 iterations를 두번 반복(잡힌 범위 주변 margin이 뚱뚱해진다)
 
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    
    contours, _ = cv2.findContours(skinMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)#(라인을 그릴 이미지, 검출된 컨투어, 음수로 지정할경우 모든 컨투어,색상지정(현재는 초록),선두께)

    rects = [cv2.boundingRect(contour) for contour in contours]#시작점의 X,y좌표와 너비와 높이를 배열에 저장.

    for rect in rects:
        if (rect[2] * rect[3] < 100000) or (rect[2] * rect[3] > 360000):
            continue
        
        val_H = int((rect[0] + rect[2]) / 16)
        val_V = int((rect[1] + rect[3]) / 16)
        
        HL = (rect[0] - val_H) 
        HR = (rect[0] + rect[2] + val_H)
        VU = (rect[1] - val_V)
        VD = (rect[1] + rect[3] + val_V)
        
        if HL <= 0:
            HL = 0
        elif HR >= 1280:
            HR = 1280
        elif VU <= 0:
            VU = 0
        elif VD >= 960:
            VD = 960
            
        cv2.rectangle(frame, (HL, VU), (HR, VD), (0, 0, 255), 3)

        if (HL >= 0 and HR <= 1280 and VU >= 0 and VD <= 960): 
            image = frame[HR:VD, HL:VU]
            image_data = preprocess(image)
            image_test = image_data.cpu()
            imshow(torchvision.utils.make_grid(image_test,nrow=1))
        
    cv2.imshow("images",frame)
    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()