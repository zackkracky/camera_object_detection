import cv2

cfg_path = "models/yolo4-tiny.cfg"
weight_path = "models/yolo4-tiny.weights"

net = cv2.dnn.readNetFromDarknet(cfg_path,weight_path)

print("model loaded succesfully")