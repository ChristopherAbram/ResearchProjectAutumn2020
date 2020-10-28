import numpy as np
import cv2
from utils.image import resize, convolve2D
import sklearn.metrics


def confusion_matrix(array1, array2, threshold, sfactor=1):
    """
    Compares two matrices in order to obtain confusion matrix.
    Parameters:
        array1 (ndarray): 2D array, predicted values (binary image - 0s and 1s),
        array2 (ndarray): 2D array, ground truth values (binary image - 0s and 1s),
        sfactor (int): by how much scale the internal product matrix.
    """
    height1, width1 = array1.shape
    height2, width2 = array2.shape
    cwidth, cheight = sfactor * np.lcm(width1, width2), sfactor * np.lcm(height1, height2)
    # Bring to the same size:
    ra1 = resize(array1, (cwidth, cheight), interpolation=cv2.INTER_AREA)
    ra2 = resize(array2, (cwidth, cheight), interpolation=cv2.INTER_AREA)

    kw, kh = int(cwidth / (width2 * sfactor)), int(cheight / (height2 * sfactor))

    kernel = 1. / (kw * kh) * np.ones((kw, kh), dtype=np.float32)
    result = convolve2D(ra1, kernel, strides=(kh, kw))
    result_th = cv2.threshold(result, threshold, 1.0, cv2.THRESH_BINARY)[1]

    result_flat = result_th.ravel() 
    ground_truth_flat = array2.ravel()
    cmatrix = sklearn.metrics.confusion_matrix(ground_truth_flat, result_flat)
    return result, result_th, cmatrix, ra1, ra2

