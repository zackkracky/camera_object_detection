import cv2
import time



cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
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
        
        frame = cv2.flip(frame, 1)
        #0 for vertical mirror and -1 for both horrizontal and veritcal mirror
        
        
        if not status:
            continue
            
        Mask_img = backsub.apply(frame, learningRate = 0.0001)#if rate is zero then it basically just detects the motion no more learning
        
        #threshold config for Mask
        _, threshold_img = cv2.threshold(
            Mask_img,
            250,#choose either 250 or 252
            255,
            cv2.THRESH_BINARY
        )


        #Kernel tool config(remember kernel is a tool cant make any changes on its own)
        kernel =  cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5,5)
        )

        #Morphology config
        
        clean = cv2.morphologyEx(
            threshold_img,
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

        clean3 = cv2.morphologyEx(
            clean2,
            cv2.MORPH_DILATE,
            kernel,
            iterations =2 
        )
        

        #Contour config
        contour, _ = cv2.findContours(
            clean3,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        #minimum area of pixels for it to be considered as an object
        Minimum_area = 800

        for c  in contour:
            area = cv2.contourArea(c)

            if area < Minimum_area:
                continue


            #Bounded region of object    
            x,y,w,h = cv2.boundingRect(c)
            
            #Centre
            centre_x = x + (w//2)
            centre_y = y + (h//2)

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

            cv2.circle(
                frame,
                (centre_x,centre_y),
                4,
                (0,0,255),
                -1

            )
            

        cv2.imshow("background Mask",threshold_img)
        cv2.imshow("Post Morphology",clean2)
        cv2.imshow("live",frame)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()