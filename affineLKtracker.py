import numpy as np
import cv2


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


def affineLKtracker(img, tmp, rect, p):

    # Initialization
    rows, cols = tmp.shape
    learning_rate = 50
    warp_mat = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])

    # Calculate warp image
    warp_img = cv2.warpAffine(img, warp_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    crop_warp_img = warp_img.astype(np.uint8)[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    histEqual_crop_warp_img = cv2.equalizeHist(crop_warp_img)
    diff = tmp.astype(int) - histEqual_crop_warp_img
    # diff = tmp.astype(int) - crop_warp_img

    # Calculate warp gradient of image
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad_x_warp = cv2.warpAffine(grad_x, warp_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    grad_x_warp = grad_x_warp[rect[0][1]: rect[1][1], rect[0][0]: rect[1][0]]
    grad_y_warp = cv2.warpAffine(grad_y, warp_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    grad_y_warp = grad_y_warp[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    # Calculate Jacobian
    jacob = jacobian(cols, rows)

    # Set gradient of a pixel into 1 by 2 vector
    grad = np.stack((grad_x_warp, grad_y_warp), axis=2)
    grad = np.expand_dims((grad), axis=2)
    steepest_descents = np.matmul(grad, jacob)
    steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))

    # Compute Hessian matrix
    hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))

    # Compute steepest-gradient-descent update
    diff = diff.reshape((rows, cols, 1, 1))
    update = (steepest_descents_trans * diff).sum((0,1))
    d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
    p += learning_rate * d_p

    return p, histEqual_crop_warp_img
