import rasterio
import numpy as np

from utils.helpers import prepare_data, get_pixels, get_project_path, filter_bounds
from utils.iterator import ArrayIterator


class Pipeline():

    def __init__(self, raster1_path, raster2_path, window_height, window_width):
        """

        reaster assume channel 1
        :param raster1_path:
        :param raster2_path:
        :param window_height:
        :param window_width:
        """
        self.raster1 = rasterio.open(raster1_path)
        self.raster2 = rasterio.open(raster2_path)
        self.result = np.zeros(self.raster2.shape, dtype=np.uint8)
        self.window_iterator = ArrayIterator(self.raster1, window_height, window_width)


    def run(self):
        self.window_iterator.reset()
        while not self.window_iterator.has_reached_end():

            # get next window
            window = self.window_iterator.pop_window()

            # read data from FB raster using current window
            data = self.raster1.read(1,window=window)

            # replace nan with 0 and >0 with 1
            data = prepare_data(data)

            # get all pixels contained in window
            window_pixels = get_pixels(window)

            # keep only those pixels for which data has a 1 entry
            window_pixels = window_pixels[data.ravel() > 0]

            # check if there are any pixels, if continue from next iteration
            if window_pixels.size > 0:

                # use raster1 to get coordinates for each pixel
                raster1_vxy = np.vectorize(self.raster1.xy) # gets center
                xcoords, ycoords = raster1_vxy(window_pixels[:,0], window_pixels[:,1])

                # for each coordinate get corresponding pixel in raster2
                raster2_vindex = np.vectorize(self.raster2.index)
                raster2_pixels = np.vstack(raster2_vindex(xcoords, ycoords, op=round, precision=15)).T

                # make sure all raster2_pixels are in bounds of raster2.shape
                raster2_pixels = filter_bounds(raster2_pixels, ((0,self.raster2.height),(0,self.raster2.width)))

                # get unique counts for pixels in GRID raster
                raster2_pixels_unique, counts = np.unique(raster2_pixels, return_counts=True, axis=0)

                # update result
                self.result[raster2_pixels_unique[:,0], raster2_pixels_unique[:,1]] += counts.astype(np.uint8)


    def write_to_tif(self):
        project_path = get_project_path()
        with rasterio.open(
                project_path / 'data/metrics/pipeline-counts.tif', \
                'w', \
                driver='GTiff', \
                width=self.raster2.shape[1], \
                height=self.raster2.shape[0], \
                count=1, \
                dtype=np.uint8, \
                crs=self.raster2.crs, \
                transform=self.raster2.transform \
                ) as dst:
            dst.write(self.result, indexes=1)

