import cv2




cap = cv2.VideoCapture(0)


sc, frame = cap.read()

count = 0

while True:
    cv2.imshow("Frame", frame)
    count += 1
    print(count)


    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# close everything
cap.release()
k = cv2.waitKey(1)
# cv2.destroyAllWindows()