import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg
from humset.utils.definitions import get_project_path
import os

country = 'NGA'
in_file = os.path.join(get_project_path(), "data", "humdata", 'population_%s_2018-10-01.tif' % country.lower())
out_file = os.path.join(get_project_path(), "data", "humdata", "out", 'population_%s_2018-10-01.tif' % country.lower())

with rasterio.Env():
    gdal_data = os.environ['GDAL_DATA']
    print(gdal_data)
    proj_lib = os.environ['PROJ_LIB']
    print(proj_lib)

    dst_crs = {'init': 'EPSG:3857'}

    with rasterio.open(in_file) as src:
        # transform, width, height = calculate_default_transform(src.crs, dst_crs, 
        #                                                     src.width, 
        #                                                     src.height, 
        #                                                     *src.bounds)
        kwargs = src.meta.copy()
        # kwargs.update({'crs': dst_crs,'transform': transform, 'width': width,'height': height})
        print(kwargs['transform'])

        calculate_default_transform

        with rasterio.open(out_file, 'w', **kwargs) as dst:
                reproject(source=rasterio.band(src, 1),destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
