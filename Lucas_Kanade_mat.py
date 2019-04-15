import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def jacobian(x_shape, y_shape):
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y) 
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)

    return jacob


def lucas_kanade(template, current):
    # Initialization
    err_threshold = 1
    p = np.zeros((6,))
    warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    print(warp_mat.shape)

    rows, cols = template.shape

    # Calculate error image

    # while np.linalg.norm(d_p, 2) > err_threshold:
    for k in range(1000):

        # Calculate error image
        warp_current = cv2.warpAffine(current, warp_mat, (cols, rows))
        error_img = template - warp_current

        # Calculate gradient of the image, and warp them by current warp_mat
        grad_x = cv2.Sobel(current, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(current, cv2.CV_64F, 0, 1, ksize=5)
        warp_grad_x = cv2.warpAffine(grad_x, warp_mat, (cols, rows))
        warp_grad_y = cv2.warpAffine(grad_y, warp_mat, (cols, rows))

        # Calculate Jacobian
        jacob = jacobian(current.shape[1], current.shape[0])

        # Set gradient of a pixel into 1 by 2 vector
        grad = np.stack((warp_grad_x, warp_grad_y), axis=2)
        grad = np.expand_dims((grad), axis=2)
        steepest_descents = np.matmul(grad, jacob)
        steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))

        # Compute Hessian matrix
        hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))

        # Compute steepest-gradient-descent update
        diff = (template - warp_current).reshape((current.shape[0], current.shape[1], 1, 1))
        update = (steepest_descents_trans * diff).sum((0,1))
        d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape(-1)
        p += d_p

    return p
