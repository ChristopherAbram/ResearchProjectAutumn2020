import sys
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import itertools
from shapely.geometry import box


class RasterWindow:
    def __init__(self, filepath, x_slices = 3, y_slices = 3):
        self.filepath = filepath
        self.num_xslices = x_slices
        self.num_yslices = y_slices
        self._raster = rasterio.open(self.filepath)
        self._w_pairs = itertools.product(
            range(self.num_yslices),
            range(self.num_xslices))
        self.si_width, self.si_height = \
            self._raster.width // self.num_xslices, \
            self._raster.height // self.num_yslices
        self.xi, self.yi = 0, 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.xi, self.yi = next(self._w_pairs)
            return self.__get(self.xi, self.yi)
        except StopIteration:
            self._raster.close()
            raise

    def __get(self, xi, yi):
        w = Window(yi * self.si_width, xi * self.si_height, self.si_width, self.si_height)
        w_ = self._raster.width - yi * self.si_width
        h_ = self._raster.height - xi * self.si_height
        
        if yi == (self.num_xslices - 1) and w_ > 0:
            w = Window(yi * self.si_width, xi * self.si_height, w_, self.si_height)

        if xi == (self.num_yslices - 1) and h_ > 0:
            w = Window(yi * self.si_width, xi * self.si_height, self.si_width, h_)

        if yi == (self.num_xslices - 1) and (xi == self.num_yslices - 1) and h_ > 0:
            w = Window(yi * self.si_width, xi * self.si_height, self.si_width, h_)

        if yi == (self.num_xslices - 1) and (xi == self.num_yslices - 1) and w_ > 0:
            w = Window(yi * self.si_width, xi * self.si_height, w_, self.si_height)

        if yi == (self.num_xslices - 1) and (xi == self.num_yslices - 1) and w_ > 0 and h_ > 0:
            w = Window(yi * self.si_width, xi * self.si_height, w_, h_)

        X = self._raster.read(1, window=w)
        return X, (xi, yi), (X.shape[1], X.shape[0])

    def get(self, x, y):
        """
        Get (x, y) window from (N, M) possible windows.
        Remeber that windows are indexed the same as arrays, that is starting form 0.
        """
        if x >= self.num_yslices or y >= self.num_xslices:
            raise IndexError(
                "Window indexes out of range. Given ({}, {}) and maximum are ({}, {})".format(
                    x, y, self.num_yslices - 1, self.num_xslices - 1))

        self.xi, self.yi = x, y
        X, _, (w, h) = self.__get(x, y)
        return X, (w, h)

    def get_coords(self, row, col):
        """
        Find geo coordinate by row and col given with respect to patch.
        row - the pixel row,
        col - the pixel column
        """
        col_ = self.yi * self.si_width + col
        row_ = self.xi * self.si_height + row
        return self._raster.xy(row_, col_)


    def get_raster(self):
        return self._raster

    def get_raster_bbox(self):
        """return spatial bbox in form [[x_low, x_high], [y_low, y_high]]"""
        upper_left = self._raster.transform * (0,0)
        lower_right = self._raster.transform * (self._raster.width, self._raster.height)
        bbox = [[upper_left[0], lower_right[0]],[lower_right[1], upper_left[1]]]
        return bbox


class RasterWindowSize(RasterWindow):
    def __init__(self, filepath, width=1024, height=1024):
        super(RasterWindowSize, self).__init__(filepath)
        self.width = width
        self.height = height
        self.num_xslices = self._raster.width // self.width
        self.num_yslices = self._raster.height // self.height
        self._w_pairs = itertools.product(
            range(self.num_yslices),
            range(self.num_xslices))
        self.si_width, self.si_height = self.width, self.height


def get_window_px(raster, x, y, width, height):
    return raster.read(1, window=Window.from_slices((x, x + height), (y, y + width)))


def get_window_geo(raster, bbox):
    [lonl, latb, lonr, latu] = bbox.bounds
    rx, ry = raster.index(lonr, latb)
    lx, ly = raster.index(lonl, latu)
    w, h = np.abs(ry - ly), np.abs(lx - rx)
    return get_window_px(raster, lx, ly, w, h)
