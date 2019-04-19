import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from InverseLK import*


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [[x, y]]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append([x, y])
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, tuple(refPt[0]), tuple(refPt[1]), (0, 255, 0), 2)
        cv2.imshow("test", image)

frame_increase = 0
frame_num = str(140 + frame_increase).zfill(4) + ".jpg"
cap = cv2.VideoCapture("human/" + frame_num)
success, image = cap.read()
out = cv2.VideoWriter('car_tracking.avi', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (720, 480))

count = 0
p = np.zeros(6)
while success:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.bilateralFilter(image,9,75,75)
    clone = image.copy()
    if count == 0:
        refPt = []
        cropping = False
        # load the image, clone it, and setup the mouse callback function
        cv2.namedWindow('template', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('template', 320*5, 240*5)
        cv2.setMouseCallback("template", click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("template", image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                image = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if len(refPt) == 2:
            template = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            template = cv2.equalizeHist(template)
            cv2.imwrite('template.jpg', template)
            print("top_left(x,y) and bottom_right(x,y) is")
            print(refPt)
            cv2.waitKey(0)
        
        count += 1

    else:
        print(frame_increase)
        p = InverseLK(clone, template, refPt, p)

        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        warp_mat = cv2.invertAffineTransform(warp_mat)
        newRefPt = np.hstack((refPt, [[1], [1]]))
        newRefPt = np.dot(warp_mat, newRefPt.T).astype(int)
        print('new refpt', tuple(newRefPt.T[0]), tuple(newRefPt.T[1]))

        cv2.rectangle(image, tuple(newRefPt.T[0]), tuple(newRefPt.T[1]), (0, 255, 0), 2)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 320*5, 240*5)
        cv2.imshow("image", image)
        cv2.waitKey(1)

    out.write(image)
    frame_increase += 1
    frame_num = str(140 + frame_increase).zfill(4) + ".jpg"
    cap = cv2.VideoCapture("human/" + frame_num)
    success, image = cap.read()
cap.release()
out.release()
cv2.destroyAllWindows()
