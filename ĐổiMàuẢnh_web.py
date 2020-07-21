import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    h, w, _ = frame.shape
    ret1, frame1 = cap.read()
    #Red Brown Video
    frame1 = np.full((h, w, 3), (0, 0, 255), np.uint8)
    used_frame = cv2.addWeighted(frame, 0.7, frame1, 0.3, 0)

    # Gray Video
    gray = (cv2.cvtColosframe, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',used_frame)
    #cv2.imshow('gray',gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()