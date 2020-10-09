import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import itertools


class RasterWindow:
    def __init__(self, filepath, x_slices = 3, y_slices = 3):
        self.filepath = filepath
        self.num_xslices = x_slices
        self.num_yslices = y_slices

        self.__raster = rasterio.open(self.filepath)


    def __iter__(self):
        self.__w_pairs = itertools.product(
            range(self.num_xslices), 
            range(self.num_yslices))
        return self


    def __next__(self):
        try:
            xi, yi = next(self.__w_pairs)
            si_width, si_height = \
                int(self.__raster.width / self.num_xslices), \
                int(self.__raster.height / self.num_yslices)
            w = Window.from_slices(
                (yi * si_height, yi * si_height + si_height), 
                (xi * si_width, xi * si_width + si_width))
            return self.__raster.read(1, window=w)
        except StopIteration:
            raise
