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

row, col = image.shape[0], image.shape[1]
count = 0
p = np.zeros(6)
p_r = np.zeros(6)
p_g = np.zeros(6)
p_b = np.zeros(6)
video_frames = []

while success:
    rgb = image.copy()
    r, g, b = cv2.split(image)
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
            r_crop = r[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            g_crop = g[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            b_crop = b[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
            r_temp = r_crop.copy()
            g_temp = g_crop.copy()
            b_temp = b_crop.copy()
            template = crop.copy()
            cv2.imwrite('/home/an/Desktop/673/Project 4/template.jpg', crop)
            print("top_left(x,y) and bottom_right(x,y) is")
            print(refPt)
            cv2.waitKey(0)
        count += 1

    else:
        for k in range(100):
            p = affineLKtracker(clone, template, refPt, p)
            p_r = affineLKtracker(r, r_temp, refPt, p_r)
            p_g = affineLKtracker(g, g_temp, refPt, p_g)
            p_b = affineLKtracker(b, b_temp, refPt, p_b)
        # p = (p_r + p_g + p_b)/3
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        warp_mat_r = np.array([[1 + p_r[0], p_r[2], p_r[4]], [p_r[1], 1 + p_r[3], p_r[5]]])
        warp_mat_g = np.array([[1 + p_g[0], p_g[2], p_g[4]], [p_g[1], 1 + p_g[3], p_g[5]]])
        warp_mat_b = np.array([[1 + p_b[0], p_b[2], p_b[4]], [p_b[1], 1 + p_b[3], p_b[5]]])
        # newRefPt = []
        # for i in range(2):
        #     new = refPt[i].copy()
        #     new.append(1)
        #     newRefPt.append(new)
        # newRefPt = np.array(newRefPt)
        # newRefPt = np.dot(warp_mat, newRefPt.T).astype(int).T
        newRefPt = np.zeros((2,2))
        newRefPt_r = np.zeros((2, 2))
        newRefPt_g = np.zeros((2, 2))
        newRefPt_b = np.zeros((2, 2))
        for i in range(2):
            newRefPt[0][i] = int(warp_mat[i][0] * refPt[0][0] + warp_mat[i][1] * refPt[0][1] + warp_mat[i][2])
            newRefPt[1][i] = int(warp_mat[i][0] * refPt[1][0] + warp_mat[i][1] * refPt[1][1] + warp_mat[i][2])
            newRefPt_r[0][i] = int(warp_mat_r[i][0] * refPt[0][0] + warp_mat_r[i][1] * refPt[0][1] + warp_mat_r[i][2])
            newRefPt_r[1][i] = int(warp_mat_r[i][0] * refPt[1][0] + warp_mat_r[i][1] * refPt[1][1] + warp_mat_r[i][2])
            newRefPt_g[0][i] = int(warp_mat_g[i][0] * refPt[0][0] + warp_mat_g[i][1] * refPt[0][1] + warp_mat_g[i][2])
            newRefPt_g[1][i] = int(warp_mat_g[i][0] * refPt[1][0] + warp_mat_g[i][1] * refPt[1][1] + warp_mat_g[i][2])
            newRefPt_b[0][i] = int(warp_mat_b[i][0] * refPt[0][0] + warp_mat_b[i][1] * refPt[0][1] + warp_mat_b[i][2])
            newRefPt_b[1][i] = int(warp_mat_b[i][0] * refPt[1][0] + warp_mat_b[i][1] * refPt[1][1] + warp_mat_b[i][2])
        newRefPt = newRefPt.astype(int)
        newRefPt_b = newRefPt_b.astype(int)
        newRefPt_g = newRefPt_g.astype(int)
        newRefPt_r = newRefPt_r.astype(int)
        cv2.rectangle(rgb, tuple(newRefPt[0]), tuple(newRefPt[1]), (0, 255, 255), 2)
        cv2.rectangle(rgb, tuple(newRefPt_r[0]), tuple(newRefPt_r[1]), (0, 0, 255), 2)
        cv2.rectangle(rgb, tuple(newRefPt_g[0]), tuple(newRefPt_g[1]), (0, 255, 0), 2)
        cv2.rectangle(rgb, tuple(newRefPt_b[0]), tuple(newRefPt_b[1]), (255, 0, 0), 2)
        cv2.rectangle(rgb, tuple(refPt[0]), tuple(refPt[1]), (255, 0, 255), 2)
        cv2.imshow("", rgb)
        video_frames.append(rgb)
        cv2.waitKey(1)


    frame_increase += 1
    frame_num = str(140 + frame_increase).zfill(4) + ".jpg"
    cap = cv2.VideoCapture("./human/" + frame_num)
    success, image = cap.read()

out = cv2.VideoWriter('human_output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10.0, (col, row))
for i in range(len(video_frames)):
    out.write(video_frames[i])
out.release()


cv2.destroyAllWindows()