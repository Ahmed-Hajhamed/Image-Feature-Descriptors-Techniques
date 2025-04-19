import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def harris_detector(orignal_image):
    # convert to grayscale
    image = orignal_image.copy()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = np.float32(gray)  #we must convert it to float for better results

    # start the time
    start_time = time.time()

    ##################### *Harris Corner Detection and lambda_min* ######
    # R= det(H) - k*(Trace(H))^2

    # calc gradients using soble edge detection
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # calc the matrices
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # blurring with gaussian
    blockSize = 5
    Sigma = 1
    Sxx = cv2.GaussianBlur(Ixx, (blockSize, blockSize), sigmaX=Sigma)
    Syy = cv2.GaussianBlur(Iyy, (blockSize, blockSize), sigmaX=Sigma)
    Sxy = cv2.GaussianBlur(Ixy, (blockSize, blockSize), sigmaX=Sigma)

    # harris equation: R= det(H) - k*(Trace(H))^2
    k = 0.04
    detH = Sxx * Syy - Sxy ** 2
    traceH = Sxx + Syy
    corners = detH - k * (traceH ** 2)

    # calc lambda minus: using equation that is big
    term = np.sqrt(((Sxx - Syy) ** 2) / 4 + Sxy ** 2)
    lambda1 = (traceH / 2) + term
    lambda2 = (traceH / 2) - term
    lambda_min = np.minimum(lambda1, lambda2)

    # calc the time and print it
    elapsed_time = time.time() - start_time
    print(f"Computation Time: {elapsed_time:.4f} seconds")

    # for better visualization
    corners = cv2.dilate(corners, None)
    # lambda_min = cv2.dilate(lambda_min, None)

    # threshold and highlight the corners with red dots and lambda_min with green dots
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    image[lambda_min > 0.1 * lambda_min.max()] = [0, 255, 0]

    
    return image
