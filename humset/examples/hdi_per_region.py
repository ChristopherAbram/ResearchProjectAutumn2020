import os
import numpy as np
import rasterio
import rasterio.mask as mask
import geopandas as gpd
import pandas as pd
import humset.shapefiles_by_country as sbc
from fiona.crs import from_epsg
from shapely.geometry import box
import matplotlib.pyplot as plt
import rasterstats

from humset.utils.definitions import get_project_path
from rasterstats import zonal_stats

hdi_path = os.path.join(get_project_path(), 'data/hdi', 'GDL-Sub-national-HDI-data.csv')
hdi_nigeria = pd.read_csv(hdi_path, usecols=['GDLCODE','2018'])
region_codes = np.array(hdi_nigeria)[1:,0]
ix = np.argsort(region_codes)
region_codes = region_codes[ix]
region_hdis =  np.array(hdi_nigeria)[1:,1]
region_hdis = region_hdis[ix]
code_hdi = list(zip(region_codes, region_hdis))

optimistic = 'nga_metrics_p30_t20.tif'
pesimistic = 'nga_metrics_p30_t80.tif'
medium = 'nga_metrics_p30_t50.tif'

metrics_path = os.path.join(get_project_path(), 'data/metrics', optimistic)
with rasterio.open(metrics_path) as src:
    affine = src.transform
    tp_layer = src.read(1)
    fp_layer = src.read(2)
    fn_layer = src.read(3)
    tn_layer = src.read(4)
    acc_layer = src.read(5)
    recall_layer = src.read(6)
    prec_layer = src.read(7)
    f1_layer = src.read(8)

region_shapes = sbc.get_shapes_rasterstats('Nigeria')

true_positive = zonal_stats(region_shapes, tp_layer, affine=affine, stats="sum")
false_positive = zonal_stats(region_shapes, fp_layer, affine=affine, stats="sum")
false_negative = zonal_stats(region_shapes, fn_layer, affine=affine, stats="sum")
true_negative = zonal_stats(region_shapes, tn_layer, affine=affine, stats="sum")
# metrics
accuracy = zonal_stats(region_shapes, acc_layer, affine=affine, stats="mean")
recall = zonal_stats(region_shapes, recall_layer, affine=affine, stats="mean")
precision = zonal_stats(region_shapes, prec_layer, affine=affine, stats="mean")
f1 = zonal_stats(region_shapes, f1_layer, affine=affine, stats="mean")

chosen_metric = accuracy

np_chosen_metric = np.array(chosen_metric)
metric_per_region = [d['mean'] for d in np_chosen_metric]
plt.scatter(region_hdis, metric_per_region)
plt.show()
