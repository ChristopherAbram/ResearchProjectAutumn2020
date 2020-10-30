import numpy as np
import cv2


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
    common_height = np.lcm(predicted_height, truth_height)
    common_width = np.lcm(predicted_width, truth_width)

    predicted_resized = cv2.resize(predicted, (common_width, common_height), interpolation=cv2.INTER_AREA)
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
