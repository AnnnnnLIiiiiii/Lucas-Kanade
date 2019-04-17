import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from affineLKtracker import*


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
        cv2.imshow("image", image)

frame_increase = 0
frame_num = str(140+frame_increase).zfill(4) + ".jpg"
cap = cv2.VideoCapture("./human/" + frame_num)
success, image = cap.read()

count = 0
p = np.zeros(6)
while success:
    rgb = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clone = image.copy()
    if count == 0:
        refPt = []
        cropping = False
        # load the image, clone it, and setup the mouse callback function
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", image)
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
            crop = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            template = crop.copy()
            cv2.imwrite('./template.jpg', crop)
            print("top_left(x,y) and bottom_right(x,y) is")
            print(refPt)
            cv2.waitKey(0)
        count += 1

    else:
        for k in range(50):
            p = affineLKtracker(clone, template, refPt, p)
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        # newRefPt = []
        # for i in range(2):
        #     new = refPt[i].copy()
        #     new.append(1)
        #     newRefPt.append(new)
        # newRefPt = np.array(newRefPt)
        # newRefPt = np.dot(warp_mat, newRefPt.T).astype(int).T
        newRefPt = np.zeros((2,2))
        for i in range(2):
            newRefPt[0][i] = int(warp_mat[i][0] * refPt[0][0] + warp_mat[i][1] * refPt[0][1] + warp_mat[i][2])
            newRefPt[1][i] = int(warp_mat[i][0] * refPt[1][0] + warp_mat[i][1] * refPt[1][1] + warp_mat[i][2])
        newRefPt = newRefPt.astype(int)
        cv2.rectangle(rgb, tuple(newRefPt[0]), tuple(newRefPt[1]), (0, 0, 255), 2)
        cv2.rectangle(rgb, tuple(refPt[0]), tuple(refPt[1]), (0, 255, 0), 2)
        cv2.imshow("", rgb)
        cv2.waitKey(1)


    frame_increase += 1
    frame_num = str(140 + frame_increase).zfill(4) + ".jpg"
    cap = cv2.VideoCapture("./human/" + frame_num)
    success, image = cap.read()

cv2.destroyAllWindows()