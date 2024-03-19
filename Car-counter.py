from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
cap=cv2.VideoCapture("/Users/poojithamm/PycharmProjects/Object-Detection-Yolo/pythonProject/Chapter-1/Videos/cars.mp4")
model=YOLO('../Yolo-Weights/yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask=cv2.imread("Object-Detection-Yolo/pythonProject/Chapter-1/images/mask.png")
#TRACKING
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limit=[398,297,673,297]
totalCount=0
while True:
    success,img=cap.read()
    imgRegion=cv2.bitwise_and(img,mask)
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    for i in results:
        boxes=i.boxes
        for j in boxes:
            #idhu bounding box kaaga
            x1,y1,x2,y2=j.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            #idhu confidence podradhuku
            conf=math.ceil(j.conf[0]*100)/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))
            #idhu class Name kaga
            cls=int(j.cls[0])
            currentclass=classNames[cls]
            if (currentclass=='car' or currentclass=='truck' or currentclass=='bus' or currentclass=="motorbike") and conf>0.5:
                #cvzone.putTextRect(img,f'{currentclass} {conf}',(max(0,x1),max(35,y1)))
                cvzone.cornerRect(img,(x1,y1,w,h),10,rt=5)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))
    resultsTracker=tracker.update(detections)
    cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),5)
    for i in resultsTracker:
        x1,y1,x2,y2,id=i
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        w,h=x2-x1,y2-y1
        print(i)
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img,f"Id:{int(id)}",(max(0,x1),max(35,y1)))
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        if limit[0]<cx<limit[2] and limit[1]-20<cy<limit[1]+20:
            totalCount+=1
    cvzone.putTextRect(img,f"Count:{totalCount}",(50,50))
    cv2.imshow("Pooji",img)
    cv2.waitKey(1)