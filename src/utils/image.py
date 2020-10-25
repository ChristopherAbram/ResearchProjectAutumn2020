import numpy as np
import cv2


def convolve2D(image, kernel, padding=(0,0), strides=(1,1)):
    # Cross correlation
    kernel = np.flipud(np.fliplr(kernel))

    xKernShape, yKernShape = kernel.shape
    xImgShape, yImgShape = image.shape
    xPadding, yPadding = padding
    xStride, yStride = strides

    xOutput = int(((xImgShape - xKernShape + 2 * xPadding) / xStride) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * yPadding) / yStride) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply padding
    if padding != (0, 0):
        imagePadded = np.zeros((xImgShape + xPadding * 2, yImgShape + yPadding * 2))
        imagePadded[int(xPadding):int(-1 * xPadding), int(yPadding):int(-1 * yPadding)] = image
    else:
        imagePadded = image

    for x in range(xOutput):
        for y in range(yOutput):
            output[x, y] = (kernel * imagePadded[x * xStride:x * xStride + xKernShape, y * yStride:y * yStride + yKernShape]).sum()
    return output

def resize(image, size, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, size, interpolation=interpolation)

def humdata2binary(image):
    """Takes a raw humdata frame and returns a binary representation, 
    i.e. ones for places where there is nonzero value in original array, zero otherwise"""
    img = image.copy()
    img = np.nan_to_num(img)
    img[np.where(img > 0)] = 1
    return img.astype(np.uint8)

def humdata2visualization(image, binary=True):
    """
    Converts a raw humdata frame to be visualized as an image.\n
    Parameters:\n
        image (ndarray): a raw humdata array,
        binary (boolean): a flag whether the array should be converted to binary black and white or normalized to gray scale
    """
    if binary:
        img = image.copy()
        img = humdata2binary(img)
        img = img * 255
        return img
    else:
        img = np.zeros(image.shape, image.dtype)
        img = cv2.normalize(image, img, 0, 255, cv2.NORM_MINMAX)
        return img.astype(np.uint8)

def grid2binary(image):
    """Takes a raw grid3 frame and returns a binary representation"""
    return image.astype(np.uint8)

def grid2visualization(image, binary=True):
    """
    Converts a raw GRID3 frame to be visualized as an image.\n
    Parameters:\n
        image (ndarray): a raw grid3 array,
        binary (boolean): a flag whether the array should be converted to binary black and white or normalized to gray scale.
                          Note: GRID3 are 0s and 1s, then the convertion like this is equal to binary.
    """
    img = image.copy()
    img = grid2binary(img)
    img = img * 255
    return img

def googlemaps2visualization(g):
    img = g.get_vardata()
    img = img * 255
    return img.astype(np.uint8)
