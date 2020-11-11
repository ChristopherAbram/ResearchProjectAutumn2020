import numpy as np
import cv2, rasterio, salem, os
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map, GoogleCenterMap
from shapely.geometry import box

from humset.utils.definitions import get_project_path
from humset.utils.raster import get_window_geo

country = 'NGA'
in_files = [
    os.path.join(get_project_path(), "data", "humdata", 'population_%s_2018-10-01.tif' % country.lower()),
    os.path.join(get_project_path(), "data", "worldpop", '%s_ppp_2015.tif' % country.lower()),
    os.path.join(get_project_path(), "data", "grid3", '%s - population - v1.2 - mastergrid.tif' % country)
]

# lat, lon = (6.541456, 3.312719)  # Lagos 2
lat, lon = (8.499714, 3.423570) # Ago-Are
# lat, lon = (7.382932, 3.929635) # Ibadan
# lat, lon = (4.850891, 6.993961) # Port Harcourt

box_spread = 0.5
with rasterio.open(in_files[0]) as humdata, rasterio.open(in_files[2]) as grid3:

    h_data, h_window = get_window_geo(
        humdata, box(lon - box_spread, lat - box_spread, 
                        lon + box_spread, lat + box_spread))
    
    g_data, g_window = get_window_geo(
        grid3, box(lon - box_spread, lat - box_spread, 
                        lon + box_spread, lat + box_spread))

    with rasterio.open(
        os.path.join(get_project_path(), 'data', 'out', 'example_humdata.tif'), 'w',
        driver='GTiff', width=h_data.shape[1], height=h_data.shape[0], count=1,
        dtype=h_data.dtype, crs=humdata.crs, transform=humdata.transform) as dst:

        dst.write(h_data, indexes=1)

    with rasterio.open(
        os.path.join(get_project_path(), 'data', 'out', 'example_grid3.tif'), 'w',
        driver='GTiff', width=g_data.shape[1], height=g_data.shape[0], count=1,
        dtype=g_data.dtype, crs=grid3.crs, transform=grid3.transform) as dst:

        dst.write(g_data, indexes=1)
    