import cv2
import numpy as np


def sum_white(img):

    # get total number of pixels in image
    dimensions = img.shape
    total_pix = dimensions[0]*dimensions[1]

    n_white_pix = np.sum(img == 255)
    percent = (n_white_pix/total_pix)*100

    # print('Total: ', total_pix)
    # print('Number of white pixels:', n_white_pix)
    # print('Yellow percentage: ', str(percent)+'%')
    return percent
