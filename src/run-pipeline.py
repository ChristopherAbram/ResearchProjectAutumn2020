import rasterio

from utils.helpers import get_project_path
from pipeline import Pipeline

fb_raster_path = get_project_path() / 'data/humdata/population_nga_2018-10-01.tif'
grid_raster_path = get_project_path() / 'data/grid3/NGA - population - v1.2 - mastergrid.tif'

# prime factorization of fb_raster.height=34558: 2 * 37 * 467
# prime factorization of fb_raster.width=43172: 2 * 2 * 43 * 251
# choose window shape: (467,251)
window_height = 467
window_width = 251

pipeline = Pipeline(fb_raster_path,
                    grid_raster_path,
                    window_height,
                    window_width,
                    log=True)

pipeline.run()

pipeline.write_to_tif()
