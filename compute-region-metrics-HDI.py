import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import fiona
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time

### Get dataframe for HDI

hdi_path = 'data/shdi/SHDI Complete 4.0 (1).csv'
df = pd.read_csv(hdi_path, usecols=['iso_code', 'year', 'level', 'GDLCODE', 'shdi'], low_memory=False)
# get HDI for Nigeria at subnational level for year 2018
df = df.loc[
        (df['iso_code'] == 'NGA') &
        (df['year'] == 2018) &
        (df['level'] == 'Subnat')]
df.drop(columns=['iso_code', 'year', 'level'], inplace=True)
df['shdi'] = df['shdi'].astype(float)
df.set_index('GDLCODE', inplace=True)
df_hdi = df

### Get level 1 admin units (federal states) shapefile

shapefile_path = 'data/shapefiles/GDL Shapefiles V4.shp'
shapes_NGA = {}

with fiona.open(shapefile_path) as shapefile:
    for feature in shapefile:
        if feature['properties']['country'] == 'Nigeria':
            shapes_NGA[feature['properties']['GDLcode']] = feature['geometry']

### Set threshold and compute balanced accuracy for each federal state
# https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

dfs = {}

thresholds = np.arange(9)
gdl_codes = df_hdi.index.to_numpy()
print(gdl_codes)

for threshold in thresholds:
    bal_accuracy = {}
    f1 = {}
    precision = {}
    recall = {}

    start_time = time.time()
    for region, shape in shapes_NGA.items():

        # get truth
        with rasterio.open('data/grid3/NGA - population - v1.2 - mastergrid.tif') as dataset:
            truth = rasterio.mask.mask(dataset, [shape], all_touched=False, nodata=255)[0]
            truth = truth.flatten()
            truth = truth[truth != 255]

        # get prediction
        with rasterio.open('data/metrics/pipeline-counts.tif') as dataset:
            counts = rasterio.mask.mask(dataset, [shape], all_touched=False, nodata=255)[0]
            counts = counts.flatten()
            counts = counts[counts != 255]
            prediction = np.zeros(counts.size, dtype=np.uint8)
            prediction[counts > threshold] = 1

        bal_accuracy[region] = balanced_accuracy_score(truth, prediction)
        f1[region] = f1_score(truth, prediction)
        precision[region] = precision_score(truth, prediction)
        recall[region] = recall_score(truth, prediction)

    # make dataframe
    df_score = pd.DataFrame(index=bal_accuracy.keys())
    df_score['balanced_accuracy'], df_score['f1_score'], df_score['precision'], df_score['recall'] = [
        bal_accuracy.values(), f1.values(), precision.values(), recall.values()]
    df_score.index.rename('GDLCODE', inplace=True)
    df = df_hdi.join(df_score)

    dfs[threshold] = df
    print(f'threshold {threshold} taken: {time.time()-start_time} seconds')


# make dataframe with multiindex
data = np.vstack([df.to_numpy() for df in dfs.values()])
index = pd.MultiIndex.from_arrays([np.repeat(thresholds, gdl_codes.size), np.tile(gdl_codes,thresholds.size)])
final_df = pd.DataFrame(data=data, index=index)

final_df.to_csv('data/results/hdi-threshold-metrics.csv', sep=',')
