import os
import numpy as np
import rasterio
import rasterio.mask as mask
import geopandas as gpd
import humset.shapefiles_by_country as sbc
from fiona.crs import from_epsg
from shapely.geometry import box
import matplotlib.pyplot as plt
import rasterstats

from humset.utils.definitions import get_project_path
from rasterstats import zonal_stats

optimistic = 'nga_metrics_p30_t20.tif'
pesimistic = 'nga_metrics_p30_t80.tif'
medium = 'nga_metrics_p30_t50.tif'

metrics_path = os.path.join(get_project_path(), 'data/results', optimistic)
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
accuracy = zonal_stats(region_shapes, acc_layer, affine=affine, stats="mean")
recall = zonal_stats(region_shapes, recall_layer, affine=affine, stats="mean")
precision = zonal_stats(region_shapes, prec_layer, affine=affine, stats="mean")
f1 = zonal_stats(region_shapes, f1_layer, affine=affine, stats="mean")

print('First region')
print(f'TP: {true_positive[0]["sum"]}')
print(f'FP: {false_positive[0]["sum"]}')
print(f'FN: {false_negative[0]["sum"]}')
print(f'TN: {true_negative[0]["sum"]}')
print(f'ACC: {accuracy[0]["mean"]}')
print(f'RECALL: {recall[0]["mean"]}')
print(f'PREC: {precision[0]["mean"]}')
print(f'F1: {f1[0]["mean"]}')

print('\n')
print('Second region')
print(f'TP: {true_positive[1]["sum"]}')
print(f'FP: {false_positive[1]["sum"]}')
print(f'FN: {false_negative[1]["sum"]}')
print(f'TN: {true_negative[1]["sum"]}')
print(f'ACC: {accuracy[1]["mean"]}')
print(f'RECALL: {recall[1]["mean"]}')
print(f'PREC: {precision[1]["mean"]}')
print(f'F1: {f1[1]["mean"]}')