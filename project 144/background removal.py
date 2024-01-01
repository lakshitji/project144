import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mount_everest.jfif')
mountain = cv2.resize(mountain, (640, 480))  # Resize mountain image to match camera feed

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # resizing the frame to match the mountain image
        frame = cv2.resize(frame, (640, 480))

        # convert the mountain image to grayscale for masking
        mountain_gray = cv2.cvtColor(mountain, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mountain_gray, 10, 255, cv2.THRESH_BINARY)

        # invert the mask
        mask_inv = cv2.bitwise_not(mask)

        # bitwise and operation to extract foreground / person
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # bitwise and operation to extract the mountain background
        mountain_fg = cv2.bitwise_and(mountain, mountain, mask=mask)

        # final image
        final_frame = cv2.add(frame_bg, mountain_fg)

        # show it
        cv2.imshow('frame', final_frame)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
