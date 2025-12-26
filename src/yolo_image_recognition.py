import cv2
import os


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
net.setPreferableTarget(cv2.dnn.DNN_TARGER_CPU)

#loading class labels

with open(coco.names,"r") as f:
    classes = [lines.strip() for line in f.readlines()]

print(classes)