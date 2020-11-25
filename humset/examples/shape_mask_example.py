import os
import numpy as np
import rasterio
import rasterio.mask as mask
import geopandas as gpd
import humset.shapefiles_by_country as sbc
from fiona.crs import from_epsg
from shapely.geometry import box
import matplotlib.pyplot as plt

from humset.utils.definitions import get_project_path

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

optimistic = 'nga_metrics_p30_t20.tif'
pesimistic = 'nga_metrics_p30_t80.tif'
medium = 'nga_metrics_p30_t50.tif'

metrics_tif = os.path.join(get_project_path(), 'data/results', medium)

raster_metrics = rasterio.open(metrics_tif)

regions = sbc.get_shapes('Nigeria')
# getting shape of Anambra
shape = regions.get('NGAr102')['coordinates']
# lon and lat of Anambra
lat, lon = 6.221173, 6.971196

tam = 0.2
bbox=box(lon-tam, lat-tam, lon+tam, lat+tam)

geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0])

coords = getFeatures(geo)

# use rasterstats instead
out_img, out_transform = mask.mask(raster_metrics, coords)
metrics_shaped=np.float32(out_img[0].copy())
# metrics_shaped[np.where(metrics_shaped!=metrics_shaped)] = 0

plt.figure(figsize=(8,7))
ax = plt.subplot(111)
cmap = plt.cm.magma

plt.imshow(metrics_shaped,cmap=cmap,vmin=1)

cbar = plt.colorbar(pad=0.005,fraction=0.1)
cbar.set_label('Metrics', rotation=90)
ax.axis('on')
plt.xticks([])
plt.yticks([])
plt.show()