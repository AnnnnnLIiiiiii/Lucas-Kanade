import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os




def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
frame_increase = 0
frame_num = str(20+frame_increase).zfill(4) + ".jpg"
cap = cv2.VideoCapture("/home/an/Desktop/673/Project 4/car/frame" + frame_num)
success, image = cap.read()


def lucas_kanade(template, current):
    # Initialization
    err_threshold = 1
    p = np.zeros((6,1))
    warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    d_p = 10 * np.ones((6,1))

    rows, cols = template.shape

    # Calculate error image

    # while np.linalg.norm(d_p, 2) > err_threshold:
    for k in range(10):

        # Calculate error image
        warp_current = cv2.warpAffine(current, warp_mat, (cols, rows))
        error_img = template - warp_current

        # Calculate gradient of the image, and warp them by current warp_mat
        grad_x = cv2.Sobel(current, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(current, cv2.CV_64F, 0, 1, ksize=5)
        warp_grad_x = cv2.warpAffine(grad_x, warp_mat, (cols, rows))
        warp_grad_y = cv2.warpAffine(grad_y, warp_mat, (cols, rows))

        # Calculate the steepest gradient descent and Hessian matrix

        steepest_descent_1 = np.zeros(template.shape)
        steepest_descent_2 = np.zeros(template.shape)
        steepest_descent_3 = np.zeros(template.shape)
        steepest_descent_4 = np.zeros(template.shape)
        steepest_descent_5 = np.zeros(template.shape)
        steepest_descent_6 = np.zeros(template.shape)

        hessian_matrix = np.zeros((6,6))
        update = np.zeros((6,1))
        for d_y in range(current.shape[0]):
            for d_x in range(current.shape[1]):
                x = refPt[0][0] + d_x
                y = refPt[0][1] + d_y
                # Calculate Jacobian
                jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])

                # Set gradient of a pixel into 1 by 2 vector
                grad = np.array([warp_grad_x[d_y, d_x], warp_grad_y[d_y, d_x]])

                # steepest_descent is a 1 by 6 vector (grad * jacobian), yet for convenience reshape it to 6 by 1
                steepest_descents = np.dot(grad, jacobian).reshape(6,1)

                # Compute Hessian matrix
                hessian_matrix += np.dot(steepest_descents, steepest_descents.T)

                # Compute steepest-gradient-descent update
                update += steepest_descents * (template[d_y, d_x] - warp_current[d_y, d_x])
                print(update)
                d_p = np.dot(np.linalg.pinv(hessian_matrix), update)

                # Seperate 6 steepest descent images
                steepest_descent_1[d_y, d_x] = steepest_descents[0]
                steepest_descent_2[d_y, d_x] = steepest_descents[1]
                steepest_descent_3[d_y, d_x] = steepest_descents[2]
                steepest_descent_4[d_y, d_x] = steepest_descents[3]
                steepest_descent_5[d_y, d_x] = steepest_descents[4]
                steepest_descent_6[d_y, d_x] = steepest_descents[5]

        p += d_p

    return p



count = 0
while success:
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
            cv2.imwrite('/home/an/Desktop/673/Project 4/template.jpg', crop)
            print("top_left(x,y) and bottom_right(x,y) is")
            print(refPt)
            cv2.waitKey(0)

        count += 1

    else:
        crop = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        current = crop.copy()

        # p = lucas_kanade(template, current)
        p = np.zeros((6,1))
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        rows,cols = template.shape

        # Calculate error image
        warp_current = cv2.warpAffine(current, warp_mat, (cols, rows))
        error_img = template - warp_current

        grad_x = cv2.Sobel(current, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(current, cv2.CV_64F, 0, 1, ksize=5)
        warp_grad_x = cv2.warpAffine(grad_x, warp_mat, (cols, rows))
        warp_grad_y = cv2.warpAffine(grad_y, warp_mat, (cols, rows))


        while True:
            cv2.imshow("video", image)
            cv2.imshow("original crop area", crop)
            cv2.imshow("warp current by lucas", warp_current)
            cv2.imshow("error image", error_img)
            k = cv2.waitKey(33)
            if k == 32:  # Esc key to stop
                break

    frame_increase += 1
    frame_num = str(20 + frame_increase).zfill(4) + ".jpg"
    cap = cv2.VideoCapture("/home/an/Desktop/673/Project 4/car/frame" + frame_num)
    success, image = cap.read()

cv2.destroyAllWindows()