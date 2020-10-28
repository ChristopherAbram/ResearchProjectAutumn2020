import numpy as np
import cv2
from utils.image import resize


def confusion_matrix(array1, array2, sfactor=1):
    height1, width1 = array1.shape
    height2, width2 = array2.shape
    cwidth, cheight = sfactor * np.lcm(width1, width2), sfactor * np.lcm(height1, height2)
    # Bring to the same size:
    ra1 = resize(array1, (cwidth, cheight), interpolation=cv2.INTER_AREA)
    ra2 = resize(array2, (cwidth, cheight), interpolation=cv2.INTER_AREA)

    return ra1, ra2
