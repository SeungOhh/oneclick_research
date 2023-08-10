import cv2


cap = cv2.VideoCapture(0)
count = 0



while True:
    sc, frame = cap.read()
    
    if sc==True:
        # frame resize
        frame = cv2.resize(frame, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        
        # image show
        cv2.imshow("Frame", frame)
        count += 1
        print(count)
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# close everything
cap.release()
k = cv2.waitKey(1)
cv2.destroyAllWindows()