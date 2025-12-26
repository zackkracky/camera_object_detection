import cv2
import os

#BASIC YOLO setup

#try to understand how the os lubrbary is being used here as it will help later on alot for other projects
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg_path = os.path.join(BASE_DIR,"models/yolov4-tiny.cfg")
weights_path = os.path.join(BASE_DIR,"models/yolov4-tiny.weights")
coco_path = os.path.join(BASE_DIR,"models/coco.names")
image_path = os.path.join(BASE_DIR,"images/minnu.jpg")

#loading YOLO onto memory
net = cv2.dnn.readNetFromDarknet(cfg_path,weights_path)
#COMPUTATION parameters
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(coco_path,"r") as f:
    classes = [line.strip() for line in f.readlines()]

ouput_layers = net.getUnconnectedOutLayersNames()


#==============CAMERA SETUP===================
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

if not cap.isOpened():
    print("camera failed")
    raise SystemExit

print("YOLO detection is LIVE")

#main loop
while True:
    status, frame = cap.read()
    if not status:
        print("Frame failed")
        break  
    height, width = frame.shape[:2]

    #blob formating:
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

    if len(indexes) > 0:#checks if the number of items is not zero
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
        class_label = classes[class_id_list[i]]
        confidence = class_confidence_list[i]

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
#