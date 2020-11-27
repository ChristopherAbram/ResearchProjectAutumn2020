from pathlib import Path
import numpy as np


def get_project_path():
    return Path(__file__).parent.parent.parent.absolute()

def prepare_data(data):
    """
    Replace nans with zeros and map all elements greater than zero to one.
    Assumes all entries are already positive.

    :param data: numpy array
    :return: numpy array of same shape
    """
    data = np.nan_to_num(data) # replace nan with zero
    data[data > 0] = 1 # make binary
    return data.astype(np.uint8)


# https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def get_pixels(window):
    """
    Construct indices for all pixels in the window.

    :param window: ((row_upper, row_lower), (col_left, col_right))
    :return: numpy array of shape (X,2)
    """
    return np.array(np.meshgrid(
        np.arange(window[0][0], window[0][1]),
        np.arange(window[1][0], window[1][1]))).T.reshape(-1,2)

def filter_bounds(array, bounds):
    """
    Return only those array entries that are within bounds.
    Lower bounds is inclusive, upper bound is inclusive.
    Bounds are provided as ((lower, upper) for first column, (lower, upper) for second column).

    :param array: numpy array of shape (N,2)
    :param bounds: ((int, int), (int, int))
    :return: numpy array of shape (M,2), where M <= N
    """
    check_first = np.logical_and(array[:,0] >= bounds[0][0], array[:,0] <  bounds[0][1])
    check_second = np.logical_and(array[:,1] >= bounds[1][0], array[:,1] <  bounds[1][1])
    return array[np.logical_and(check_first, check_second)]
