import sys, os, itertools
import numpy as np
import rasterio
from rasterio.windows import Window as rWindow
from shapely.geometry import box


class Window:
    """
    Stores one entry of data retrieved from RasterTable instance.

    Attributes:
        data (ndarray): numpy 2D array with data,
        pos (tuple): (x, y) pair telling where the window is located in RasterTable,
        size (tuple): the actual size of the window (width, height),
        window (rasterio.windows.Window): an instance of corresponding rasterio window.
    """
    def __init__(self, data, pos, window=None):
        self.data = data
        self.pos = pos
        self.window = window
        self.size = (self.data.shape[1], self.data.shape[0])


class RasterTable:
    """
    The RasterTable class defines helper operations on raster data. 
    It splits the entire dataset into several, creating abstract table of 'height_slices' rows and 'width_slices' columns. 
    The class defines operations which helps to request given window (entry of the table) and perform some operation with it.
    """

    def __init__(self, filepath, width_slices = 3, height_slices = 3):
        self.filepath = filepath
        self._raster = rasterio.open(self.filepath)
        self.width_slices = width_slices
        self.height_slices = height_slices
        self.si_width, self.si_height = \
            self._raster.width // self.width_slices, \
            self._raster.height // self.height_slices

    def __del__(self):
        pass
        # self._raster.close()
        # del self._raster

    def __len__(self):
        return self.width_slices * self.height_slices

    def __iter__(self):
        return self.iterator()

    def __get(self, xi, yi):
        """
        Get window (xi, yi)
        """
        w = rWindow(yi * self.si_width, xi * self.si_height, self.si_width, self.si_height)
        w_ = self._raster.width - yi * self.si_width
        h_ = self._raster.height - xi * self.si_height
        
        if yi == (self.width_slices - 1) and w_ > 0:
            w = rWindow(yi * self.si_width, xi * self.si_height, w_, self.si_height)

        if xi == (self.height_slices - 1) and h_ > 0:
            w = rWindow(yi * self.si_width, xi * self.si_height, self.si_width, h_)

        if yi == (self.width_slices - 1) and (xi == self.height_slices - 1) and h_ > 0:
            w = rWindow(yi * self.si_width, xi * self.si_height, self.si_width, h_)

        if yi == (self.width_slices - 1) and (xi == self.height_slices - 1) and w_ > 0:
            w = rWindow(yi * self.si_width, xi * self.si_height, w_, self.si_height)

        if yi == (self.width_slices - 1) and (xi == self.height_slices - 1) and w_ > 0 and h_ > 0:
            w = rWindow(yi * self.si_width, xi * self.si_height, w_, h_)

        return Window(self._raster.read(1, window=w), (xi, yi), window=w)

    def get(self, x, y):
        """
        Get (x, y) window from (width_slices - 1, height_slices - 1) possible windows.
        Remeber that windows are indexed the same as arrays, that is starting form 0.
        x - the row index of table,
        y - the column index of table.
        """
        if x >= self.height_slices or y >= self.width_slices:
            raise IndexError(
                "Window indexes out of range. Given ({}, {}) and maximum are ({}, {})".format(
                    x, y, self.height_slices - 1, self.width_slices - 1))
        
        return self.__get(x, y)

    def iterator(self):
        return RasterTableIter(self)

    def get_raster(self):
        """
        Returns underying rasterio object.
        """
        return self._raster

    def get_raster_bbox(self):
        """
        Returns spatial bbox in form [[x_low, x_high], [y_low, y_high]]
        """
        # TODO: ...
        upper_left = self._raster.transform * (0,0)
        lower_right = self._raster.transform * (self._raster.width, self._raster.height)
        bbox = [[upper_left[0], lower_right[0]],[lower_right[1], upper_left[1]]]
        return bbox

    def find_geo_coords(self, x, y, pxrow, pxcol):
        """
        Find geo coordinate by pxrow and pxcol given with respect to window (x, y).
        x - the row index of window table,
        y - the column index of window table,
        pxrow - the pixel row,
        pxcol - the pixel column,
        """
        if x >= self.width_slices or y >= self.height_slices:
            raise IndexError(
                "Window index out of range. Given ({}, {}) and maximum are ({}, {})".format(
                    x, y, self.height_slices - 1, self.width_slices - 1))
        col_ = y * self.si_width + pxcol
        row_ = x * self.si_height + pxrow
        return self._raster.xy(row_, col_)

    def find_geo_bbox(self, x=None, y=None):
        """
        Finds the bounding box of the window (x, y) expressed in geo coordinates.
        If x or y is None, it will take the current window of the iterator.
        x - the row index of window table,
        y - the column index of window table
        """
        up_left = self.find_geo_coords(x, y, 0, 0)
        bottom_right = self.find_geo_coords(x, y, self.si_height - 1, self.si_width - 1)
        return box(up_left[0], up_left[1], bottom_right[0], bottom_right[1])

    def get_by_geo_bbox(self, bbox):
        pass


class RasterTableSize(RasterTable):
    """
    Does the same thing as RasterTable, except that it defines row and columns of the table by size of an entry.
    Note that windows at the edges of dataset will have different size, because window 
    might not be a perfect multiple of the entire dataset
    """

    def __init__(self, filepath, width=1024, height=1024):
        super(RasterTableSize, self).__init__(filepath)
        self.si_width, self.si_height = width, height
        self.width_slices = self._raster.width // self.si_width
        self.height_slices = self._raster.height // self.si_height


class RasterTableAligned(RasterTable):
    """
    The class creates a table, same as RasterTable and provides a functionality to align two GeoTiff files.
    Allows to iterate over pair of two datasets, which are geografically aligned.

    Attributes:
        filepath1 (string): a path to first GeoTiff file,
        filepath2 (string): a path to second GeoTiff file,
        width_slices (uint): number of horizontal splits,
        height_slices (uint): number of vertical splits.
    """
    
    def __init__(self, filepath1, filepath2, width_slices=3, height_slices=3):
        super(RasterTableAligned, self).__init__(filepath1, width_slices, height_slices)
        self.aligned_filepath = filepath2
        self._raster_2 = rasterio.open(self.aligned_filepath)

    def get(self, x, y):
        """
        Returns a pair of window, first window and the second which is aligned to the first one.
        Remeber that windows are indexed the same as arrays, that is starting form 0.
        x - the row index of table,
        y - the column index of table.
        """
        w1 = super(RasterTableAligned, self).get(x, y)
        w2 = self.__get_aligned(w1)
        w2.pos = (x, y)
        return w1, w2

    def get_raster(self):
        return self._raster, self._raster_2

    def __get_aligned(self, window):
        bounds = self._raster.window_bounds(window.window)
        rw = self._raster_2.window(*bounds)
        xi, yi = window.pos
        w_ = self._raster_2.width - rw.col_off
        h_ = self._raster_2.height - rw.row_off
        
        if yi == (self.width_slices - 1) and w_ > 0:
            rw = rWindow(rw.col_off, rw.row_off, w_, rw.height)
        if xi == (self.height_slices - 1) and h_ > 0:
            rw = rWindow(rw.col_off, rw.row_off, rw.width, h_)
        if yi == (self.width_slices - 1) and (xi == self.height_slices - 1) and h_ > 0:
            rw = rWindow(rw.col_off, rw.row_off, rw.width, h_)
        if yi == (self.width_slices - 1) and (xi == self.height_slices - 1) and w_ > 0:
            rw = rWindow(rw.col_off, rw.row_off, w_, rw.height)
        if yi == (self.width_slices - 1) and (xi == self.height_slices - 1) and w_ > 0 and h_ > 0:
            rw = rWindow(rw.col_off, rw.row_off, w_, h_)
        return Window(
            self._raster_2.read(1, window=rw, boundless=True, fill_value=0.), 
            (0, 0), rw)


class RasterTableSizeAligned(RasterTableAligned):
    """
    The same as RasterTableAligned but it defines splits by size (like RasterTableSize).

    Attributes:
        filepath1 (string): a path to first GeoTiff file,
        filepath2 (string): a path to second GeoTiff file,
        width (uint): a width of column in the table,
        height (uint): a height of row in the table.
    """
    # TODO: use multiple inheritance...

    def __init__(self, filepath1, filepath2, width=1024, height=1024):
        super(RasterTableSizeAligned, self).__init__(filepath1, filepath2)
        self.si_width, self.si_height = width, height
        self.width_slices = self._raster.width // self.si_width
        self.height_slices = self._raster.height // self.si_height


class RasterTableIter:
    """
    Allows to iterate over the RasterTable entries. It iterates by rows and by columns.
    To get an instance of this class, call method iterator() on RasterTable object.
    """

    def __init__(self, raster_table):
        self.raster_table = raster_table
        self._w_pairs = itertools.product(
            range(self.raster_table.height_slices),
            range(self.raster_table.width_slices))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            xi, yi = next(self._w_pairs)
            return self.raster_table.get(xi, yi)
        except StopIteration:
            raise


def get_window_px(raster, x, y, width, height):
    w = rWindow.from_slices((x, x + height), (y, y + width))
    return raster.read(1, window=w), w


def get_window_geo(raster, bbox):
    [lonl, latb, lonr, latu] = bbox.bounds
    rx, ry = raster.index(lonr, latb)
    lx, ly = raster.index(lonl, latu)
    w, h = np.abs(ry - ly), np.abs(lx - rx)
    return get_window_px(raster, lx, ly, w, h)
