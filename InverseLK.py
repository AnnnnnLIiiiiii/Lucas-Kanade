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

def eigen_mat(gradX, gradY):
    I_xx = gradX * gradX
    I_yy = gradY * gradY
    I_xy = gradX * gradY
    row1 = np.stack((I_xx, I_xy), axis=2)
    row2 = np.stack((I_xy, I_yy), axis=2)
    eigen_mat = np.stack((row1, row2), axis=2)

    return eigen_mat


def InverseLK(img, tmp, rect, p):

    # Initialization
    rows, cols = tmp.shape
    lr = 1
    #threshold = 15  # threshold for error
    threshold = 0.5*0.01  # threshold for delta_p
    d_p_norm = 1
    #error_mean = 100
    iteration = 2000
    i = 0

    # Calculate gradient of template
    # blur = cv2.GaussianBlur(tmp,(3,3),0)
    grad_x = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(tmp, cv2.CV_64F, 0, 1, ksize=5)


    # 18/04/18 Calculate weight
    # grad_T = (grad_x**2 + grad_y**2)**0.5
    E_mat = eigen_mat(grad_x, grad_y)
    _, s, _ = np.linalg.svd(E_mat)
    weight_comp = s[:,:,0]*s[:,:,1]
    weight = s[:,:,0]**0.5 + s[:,:,1]**0.5

    # Calculate Jacobian
    jacob = jacobian(cols, rows)

    # Set gradient of a pixel into 1 by 2 vector
    grad = np.stack((grad_x, grad_y), axis=2)
    grad = np.expand_dims((grad), axis=2)
    steepest_descents = np.matmul(grad, jacob)
    steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))

    # apply weight
    # grad_T = grad_T.reshape((rows, cols, 1, 1))
    # weight_sdt_1 = grad_T * steepest_descents_trans
    # weight = weight.reshape((rows, cols, 1, 1))
    # weight_sdt_2 = weight * steepest_descents_trans

         
    # Compute Hessian matrix
    hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))
    # hessian_matrix = np.matmul(weight_sdt_1, steepest_descents).sum((0, 1))
    # hessian_matrix = np.matmul(weight_sdt_2, steepest_descents).sum((0, 1))

    #while error_mean > threshold:
    while i<=iteration:
        # Calculate warp image
        warp_mat = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
        warp_img = cv2.warpAffine(img, warp_mat, (0, 0))[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        
        # Equalize the image
        warp_img = cv2.equalizeHist(warp_img)

        # Compute the error term
        error = tmp.astype(float) - warp_img.astype(float)
     
        # Compute steepest-gradient-descent update
        error = error.reshape((rows, cols, 1, 1))
        # error_mean = np.mean(np.absolute(error))
        update = (steepest_descents_trans * error).sum((0,1))
        # update = (weight_sdt_1 * error).sum((0,1))
        # update = (weight_sdt_2 * error).sum((0,1))
        d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
        
        #Update p
        d_p_deno = (1+d_p[0]) * (1+d_p[3])- d_p[1]*d_p[2]
        d_p_0 = (-d_p[0] - d_p[0]*d_p[3] + d_p[1]*d_p[2]) / d_p_deno 
        d_p_1 = (-d_p[1]) / d_p_deno
        d_p_2 = (-d_p[2]) / d_p_deno
        d_p_3 = (-d_p[3] - d_p[0]*d_p[3] + d_p[1]*d_p[2]) / d_p_deno
        d_p_4 = (-d_p[4] - d_p[3]*d_p[4] + d_p[2]*d_p[5]) / d_p_deno
        d_p_5 = (-d_p[5] - d_p[0]*d_p[5] + d_p[1]*d_p[4]) / d_p_deno

        p[0] += lr * (d_p_0 + p[0]*d_p_0 + p[2]*d_p_1)
        p[1] += lr * (d_p_1 + p[1]*d_p_0 + p[3]*d_p_1)
        p[2] += lr * (d_p_2 + p[0]*d_p_2 + p[2]*d_p_3)
        p[3] += lr * (d_p_3 + p[1]*d_p_2 + p[3]*d_p_3)
        p[4] += lr * (d_p_4 + p[0]*d_p_4 + p[2]*d_p_5)
        p[5] += lr * (d_p_5 + p[1]*d_p_4 + p[3]*d_p_5)

        d_p_norm = (d_p_0**2 + d_p_1**2 + d_p_2**2 + d_p_3**2 + d_p_4**2 + d_p_5**2)**0.5 
        i += 1
    cv2.imshow('equalize_img', warp_img)


    return p



