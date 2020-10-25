import unittest
import numpy as np

from src.utils.image import *


class ImageUtilsTest(unittest.TestCase):

    def setUp(self):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def get_convolution_size(self, in_shape, k_shape, padding, strides):
        height = int(((in_shape[0] - k_shape[0] + 2 * padding[0]) / strides[0]) + 1)
        width = int(((in_shape[1] - k_shape[1] + 2 * padding[1]) / strides[1]) + 1)
        return height, width

    def test_convolve2D_unit_kernel(self):
        img = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [1, 7, 8]])
        kernel = np.array([[1]])
        out = convolve2D(img, kernel)
        self.assertEqual(
            out.shape, self.get_convolution_size(img.shape, kernel.shape, (0,0), (1,1)))
        self.assertTrue(np.allclose(out, img))

    def test_convolve2D_kernel_same_size(self):
        img = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [1, 7, 8]])
        kernel = 1/9 * np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
        out = convolve2D(img, kernel)
        self.assertEqual(
            out.shape, self.get_convolution_size(img.shape, kernel.shape, (0,0), (1,1)))
        self.assertAlmostEqual(out[0,0], img.mean())

    def test_convolve2D_kernel_same_size_and_padding(self):
        img = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [1, 7, 8]])
        kernel = 1/9 * np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
        out = convolve2D(img, kernel, padding=(1, 1))
        self.assertEqual(
            out.shape, self.get_convolution_size(img.shape, kernel.shape, (1,1), (1,1)))
        self.assertTrue(
            np.allclose(out, 
            np.array([
                [10./9, 18./9, 14./9], 
                [18./9, 34./9, 29./9], 
                [15./9, 28./9, 24./9]])))

    def test_convolve2D_kernel_and_stride(self):
        img = np.array([
            [1, 2, 3, 1, 2, 3],
            [3, 4, 5, 3, 4, 5],
            [1, 7, 8, 1, 7, 8],
            [1, 2, 3, 1, 2, 3],
            [3, 4, 5, 3, 4, 5],
            [1, 7, 8, 1, 7, 8]])
        kernel = 1/9 * np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])
        out = convolve2D(img, kernel, strides=(3, 3))
        self.assertEqual(
            out.shape, self.get_convolution_size(img.shape, kernel.shape, (0,0), (3,3)))
        self.assertTrue(
            np.allclose(out, 
            np.array([
                [34./9, 34./9], 
                [34./9, 34./9]])))

    


if __name__ == "__main__":
    unittest.main()
