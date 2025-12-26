import cv2

#here we are using ../ in the start so it goes back to the predecessor folder then comes to model folder

cfg_path = "../models/yolov4-tiny.cfg"
weight_path = "../models/yolov  4-tiny.weights"

net = cv2.dnn.readNetFromDarknet(cfg_path,weight_path)

print("model loaded succesfully")