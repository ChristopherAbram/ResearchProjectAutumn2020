import numpy as np
import cv2
from utils.image import convolve2D
from sklearn import metrics


def make_comparable(predicted, truth):
    """
    Bring two images to same dimension by scaling each dimension by corresponding least-common-multiple and
    get shape of eventual kernel.
    :param predicted: 2D ndarray, predicted binary values
    :param truth: 2D ndarray, ground truth binary values
    :return: rescaled predicted, rescaled truth, kernel shape
    """
    predicted_height, predicted_width = predicted.shape
    truth_height, truth_width = truth.shape
    # Account for small differences in sizes
    # This implementation assumes square kernels and square inputs
    if predicted_height > predicted_width:
        padded = np.zeros((predicted_height, predicted_height), dtype=predicted.dtype)
        padded[:,:predicted_width] = predicted
        predicted_height, predicted_width = padded.shape
    elif predicted_height < predicted_width:
        padded = np.zeros((predicted_width, predicted_width), dtype=predicted.dtype)
        padded[:predicted_height,:] = predicted
        predicted_height, predicted_width = padded.shape
    else:
        padded = predicted
        
    common_height = np.lcm(predicted_height, truth_height)
    common_width = np.lcm(predicted_width, truth_width)
    predicted_resized = cv2.resize(padded, (common_width, common_height), interpolation=cv2.INTER_AREA)
    truth_resized = cv2.resize(truth, (common_width, common_height), interpolation=cv2.INTER_AREA)
    kernel_shape = (int(common_height / truth_height), int(common_width / truth_width))
    return predicted_resized, truth_resized, kernel_shape


def make_kernel(kernel_shape):
    """
    Create kernel as (identity matrix * (1 / matrix size)).
    :param kernel_shape: tuple, (height, width)
    :return: ndarray with shape kernel_shape
    """
    kernel_height, kernel_width = kernel_shape
    kernel = 1. / (kernel_height * kernel_width) * np.ones((kernel_width, kernel_height), dtype=np.float32)
    return kernel


def confusion_matrix(hrsl_binary, grid3_binary, threshold, products=True):
    hrsl_resized, grid3_resized, kernel_shape = make_comparable(hrsl_binary, grid3_binary)
    kernel = make_kernel(kernel_shape)
    convolved = convolve2D(hrsl_resized, kernel, strides=kernel_shape)
    hrsl_resized_thresholded = cv2.threshold(\
        convolved, thresh=threshold, maxval=1.0, type=cv2.THRESH_BINARY)[1]
    cm = metrics.confusion_matrix(grid3_binary.ravel(), hrsl_resized_thresholded.ravel())
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]]) # just for interpretability and debugging..
    if products:
        return cm, convolved, (hrsl_resized_thresholded, grid3_resized)
    else:
        return cm

def compute_metrics(hrsl_binary, grid3_binary, threshold):
    hrsl_resized, grid3_resized, kernel_shape = make_comparable(hrsl_binary, grid3_binary)
    kernel = make_kernel(kernel_shape)
    convolved = convolve2D(hrsl_resized, kernel, strides=kernel_shape)
    hrsl_resized_thresholded = cv2.threshold(\
        convolved, thresh=threshold, maxval=1.0, type=cv2.THRESH_BINARY)[1]
    g2r, h2r = grid3_binary.ravel(), hrsl_resized_thresholded.ravel()
    cm = metrics.confusion_matrix(g2r, h2r)
    cm = np.array([[cm[1,1], cm[1,0]], [cm[0,1], cm[0,0]]]) # just for interpretability and debugging..
    accuracy = metrics.accuracy_score(g2r, h2r)
    recall = metrics.recall_score(g2r, h2r)
    precision = metrics.precision_score(g2r, h2r)
    f1 = metrics.f1_score(g2r, h2r)
    return cm, accuracy, recall, precision, f1
