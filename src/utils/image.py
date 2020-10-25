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
    image = np.nan_to_num(image)
    image[np.where(image > 0)] = 1
    image = image.astype(np.uint8)
    return image

# def grid2binary(image):
