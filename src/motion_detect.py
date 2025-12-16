import cv2
import time



cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)
time.sleep(0.5)

if not cap.isOpened():
    print("error")
else:
    print("Video Live")

    #Background Subtractor config:
    backsub = cv2.createBackgroundSubtractorMOG2(

        history = 500,
        varThreshold = 25,
        detectShadows = True

    )

    while True:
    
        status, frame = cap.read()

        if not status:
            continue
        
        Mask_img = backsub.apply(frame)
        
        #threshold config for Mask
        _, threshold_img = cv2.threshold(
            Mask_img,
            250,#choose either 250 or 252
            255,
            cv2.THRESH_BINARY
        )


        #Kernel tool config(remember kernel is a tool cant make any changes on its own)
        kernel =  cv2.getStructureElement(
            cv2.MORPH_ELEMENT,
            (3,3)
        )

        #Morphology config
        clean1 = cv2.morphologyEx(
            threshold_img,
            cv2.MORPH_OPEN,
            kernel,
            iteration = 1
        )

        clean2 = cv2.morphologyEx(
            clean1,
            cv2.MORPH_DILATE,
            kernel,
            iteration =1 
        )

        cv2.imshow("background Mask",threshold_img)
        cv2.imshow("Post Morphology",clean2)
        cv2.imshow("live",frame)

        

        if cv2.     waitKey(1) & 0xFF == ord('q'):
            break




cap.release()
cv2.destroyAllWinodws()