import cv2
import time
import os
import numpy as np
#======================PATHS=======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg_path = os.path.join(BASE_DIR,"models/yolov4-tiny.cfg")
weights_path = os.path.join(BASE_DIR,"models/yolov4-tiny.weights")
coco_path = os.path.join(BASE_DIR,"models/coco.names")
image_path = os.path.join(BASE_DIR,"images/minnu.jpg")


#CLASSES AND YOLO MEMORY
net = cv2.dnn.readNetFromDarknet(cfg_path,weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(coco_path,"r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
ouput_layers = net.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture(0,cv2.CAP_V4L2)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
time.sleep(0.5)
#smaller resolution so less computation


if not cap.isOpened():
    print("camera crashed")
    raise SystemExit

backsub = cv2.createBackgroundSubtractorMOG2(

        history = 300,
        varThreshold = 25,
        detectShadows = True
)

kernel =  cv2.getStructuringElement(
            cv2.MORPH_RECT,#rectangular matrices
            (5,5)
)

#==================YOLO CONTROL======================

yolo_cooldown = 1.5 #seconds
last_yolo = 0

print("YOLO is LIVE now")

#=====================MAIN LOOP=========================
while True:
    status,frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not status:
        raise SystemExit
    
    #Initial declaration
    height, width, _ = frame.shape
    current_time = time.time()
    
    #===============MOTION DETECTION====================
    Mask_img = backsub.apply(frame, learningRate = 0.001)#if rate is zero then it basically just detects the motion no more learning
    #current prefered rate = 0.001 in dim lit room(or almost dark room)

    #Kernel tool config(remember kernel is a tool cant make any changes on its own)
    kernel =  cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5,5)
     )

        #Morphology config
        
    clean = cv2.morphologyEx(
            Mask_img,
            cv2.MORPH_OPEN,
            kernel,
            iterations = 1
    )

    clean1 = cv2.morphologyEx(
            clean,
            cv2.MORPH_DILATE,
            kernel,
            iterations =2 
    )
        

    clean2 =  cv2.morphologyEx(
            clean1,
            cv2.MORPH_CLOSE,
            kernel,
            iterations = 2
    )
        

        #Contour config
    contour, _ = cv2.findContours(
            clean2,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detection_status = False
    Min_Area = 1500

    for c in contour:
        if cv2.contourArea(c) > Min_Area:
            motion_detection_status = True
            break
    #===========CONDITIONAL YOLO==============
    if motion_detection_status & (current_time-last_yolo>yolo_cooldown):
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor = 1/255.0,
            size = (416,416),
            swapRB = True,
            crop = False
        )

        net.setInput(blob)
        output_by_layer= net.forward(ouput_layers)

        #Extracting boxes
        boxes = []
        class_confidence_list = []
        class_id_list = []

        for output in output_by_layer:
            for detection in output:
                '''
                    a single output looks likes:
                    [d0,d1,d2,d3,d4,d5,....]
                    each detection having
                    [x,y,w,h,confidence,clas probabilites....]
                '''
                scores = detection[5:]#we only consider the class probabilities
                class_id_index = scores.argmax()
                #basically .argmax() gives the index output of the maximum value in the list
                class_confidence = scores[class_id_index]

                if class_confidence > 0.5:
                    centre_x = int(detection[0]*width)
                    centre_y = int(detection[1]*height)

                    det_width = int(detection[2]*width)
                    det_height = int(detection[3]*height)
                    
                    #for bottom left point:
                    BL_x = int(centre_x - det_width/2)
                    BL_y = int(centre_y - det_height/2)

                    boxes.append([BL_x,BL_y,det_width,det_height])
                    class_confidence_list.append(float(class_confidence))
                    class_id_list.append(class_id_index)
        
        #NMS
        indexes = cv2.dnn.NMSBoxes(
            boxes,class_confidence_list,
            score_threshold=0.5,
            nms_threshold = 0.4
        )
        #Initialising preceeding values by appending them to the next time stamp\
        pre_boxes = []
        pre_class_id_list = []
        pre_class_confidence_list = []

        if len(indexes) > 0:
            for i in indexes.flatten():
                pre_boxes.append(boxes[i])
                pre_class_id_list.append(class_id_list[i])
                pre_class_confidence_list.append(class_confidence_list[i])
        
        last_yolo = current_time


        #======================YOLO GEOMERTRY INDICATORS========================
        
        for i in indexes.flatten():
            x,y,w,h = pre_boxes[i]
            class_label = classes[pre_class_id_list[i]]
            confidence = pre_class_confidence_list[i]

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(
                frame,
                f"{class_label}{confidence:.2f}",
                (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )
        cv2.imshow("YOLO Live",frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

cap.release()
cv2.detroyAllWinodows()