import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import rasterio
from shapely.geometry import box
from matplotlib.colors import LogNorm
from utils.location import GeoLocation

from utils.definitions import get_project_path
from utils.raster import RasterWindow, RasterWindowSize, get_window_px, get_window_geo


def main(argc, argv):

    country = 'NGA'
    # in_file = os.path.join(
    #     get_project_path(), "data", "humdata", '%s' % country, 'population_%s_2018-10-01.tif' % country.lower())

    in_file = os.path.join(
        get_project_path(), "data", "grid", '%s' % country, '%s_population.tif' % country.lower())

    # cmap = plt.cm.magma
    geo = GeoLocation()
    with rasterio.open(in_file) as dataset:
        # Loop through your list of coords
        # for i, (lon, lat) in enumerate(coordinates):

        # Get pixel coordinates from map 
        coords = geo.get_coordinates('Lagos, Nigeria')
        lon, lat = coords['lon'], coords['lat']
        px, py = dataset.index(lon, lat)
        print('Pixel Y, X coords: {}, {}'.format(py, px))

        # window = get_window_px(dataset, px - 500, py - 500, 1000, 1000)
        # rows, cols = rasterio.transform.rowcol(dataset.transform, [3.35-0.1, 3.35+0.1], [6.5-0.1, 6.5+0.1])
        window = get_window_geo(dataset, box(3.35-0.1, 6.5-0.1, 3.35+0.1, 6.5+0.1))
        window = window * 255
        fig, ax = plt.subplots()
        ax.imshow(window)
        plt.show()

            # Build an NxN window
            # window = rasterio.windows.Window(px - N // 2, py - N // 2, N, N)
            # print(window)

            # # Read the data in the window
            # # clip is a nbands * N * N numpy array
            # clip = dataset.read(window=window)

            # # You can then write out a new file
            # meta = dataset.meta
            # meta['width'], meta['height'] = N, N
            # meta['transform'] = rio.windows.transform(window, dataset.transform)

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
