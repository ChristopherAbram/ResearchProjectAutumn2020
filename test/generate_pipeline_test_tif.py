from pathlib import Path
import rasterio
import numpy as np

def get_project_path():
    return Path(__file__).parent.parent.absolute()

project_path = get_project_path()
fb_path = project_path / 'data/humdata/population_nga_2018-10-01.tif'
grid_path = project_path / 'data/grid3/NGA - population - v1.2 - mastergrid.tif'

fb_raster = rasterio.open(fb_path)
grid_raster = rasterio.open(grid_path)

data_eye = np.eye(90, dtype=np.uint8)
data_zeros = np.zeros(90*90, dtype=np.uint8).reshape(90,90)
data_ones = np.ones(90*90, dtype=np.uint8).reshape(90,90)
data_left_edge = np.copy(data_zeros)
data_left_edge[:,0] = 1
data_top_edge = np.copy(data_zeros)
data_top_edge[0,:] = 1
data_right_edge = np.copy(data_zeros)
data_right_edge[:,-1] = 1
data_bottom_edge = np.copy(data_zeros)
data_bottom_edge[-1,:] = 1

files = {
    "eye": data_eye,
    "zeros": data_zeros,
    "ones": data_ones,
    "left_edge": data_left_edge,
    "top_edge": data_top_edge,
    "right_edge": data_right_edge,
    "bottom_edge": data_bottom_edge
}

for name,data in files.items():

    with rasterio.open( \
            project_path / f'test/data/pipeline/{name}.tif', \
            'w', \
            driver='GTiff', \
            width=data.shape[1], \
            height=data.shape[0], \
            count=1, \
            dtype=data.dtype, \
            crs=fb_raster.crs, \
            transform=fb_raster.transform \
            ) as dst:
        dst.write(data, indexes=1)


# with this window, not all pixels in the test raster will have corresponding pixel here
grid_data = grid_raster.read(1, window=((0,30),(0,30)))
with rasterio.open( \
        project_path / 'test/data/pipeline/grid3-30.tif', \
        'w', \
        driver='GTiff', \
        width=grid_data.shape[1], \
        height=grid_data.shape[0], \
        count=1, \
        dtype=grid_data.dtype, \
        crs=grid_raster.crs, \
        transform=grid_raster.transform \
        ) as dst:
    dst.write(grid_data, indexes=1)


# with this window, all pixels in the test raster will have corresponding pixel here
grid_data = grid_raster.read(1, window=((0,60),(0,60)))
with rasterio.open( \
        project_path / 'test/data/pipeline/grid3-60.tif', \
        'w', \
        driver='GTiff', \
        width=grid_data.shape[1], \
        height=grid_data.shape[0], \
        count=1, \
        dtype=grid_data.dtype, \
        crs=grid_raster.crs, \
        transform=grid_raster.transform \
        ) as dst:
    dst.write(grid_data, indexes=1)
