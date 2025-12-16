import cv2

cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

if not cap.isOpened():
    print("error in launching cam")
    raise SystemExit
else:
    prints("sucess")
    
    while True:
        status , frame = cap.read()

        if not status:
            continue

        cv2.imshow("live",frame)

        if waitKey(1) & 0xFF == ord('q'):
            break
cv2.release()
cv2.destroyAllWindows()

