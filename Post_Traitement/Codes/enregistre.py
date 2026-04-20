import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
fourcc2 = cv.VideoWriter_fourcc(*'XVID')

out = cv.VideoWriter('Balade_Air1_3.avi', fourcc, 20.0, (640,  480))
out2 = cv.VideoWriter('Balade_Air2_3.avi', fourcc2, 20.0, (640,  480))

while cap.isOpened():
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # write the flipped frame
    out.write(frame)
    out2.write(frame2)

    cv.imshow('frame', frame)
    cv.imshow('frame2', frame2)
    if cv.waitKey(1) == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cap2.release()
out2.release()
cv.destroyAllWindows()