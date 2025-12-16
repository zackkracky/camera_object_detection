import cv2

cam = cv2.VideoCapture(0,cv2.CAP_V4L2)

if cam.isOpened():
    print("ON")
else:
    print("OFF")
    raise SystemExit

status, frame = cam.read()

if status:
    print("frame OK",frame.shape)
    '''
    frame.shape returns the dimensions of the frame
    which you can double check with
    '''
    cam.imwrite("test_frame.jpg",frame)
else:
    print("no frame detected")

cam.release()