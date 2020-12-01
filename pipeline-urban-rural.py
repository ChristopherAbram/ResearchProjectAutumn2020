import rasterio
import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Helpers

def prepare_data(data):
    # aggregate rural classes: 10-11-12-13 to 1
    data[np.logical_and(data > 10, data < 14)] = 1
    # aggregate urban classes: 21-22-23-30 to 2
    data[np.logical_and(data > 20, data < 31)] = 2
    # map -200,10 to 255
    data[np.logical_and(data == -200, data==10] = 0
    # cast to np.uint8 to reduce memory
    return data.astype(np.uint8)

class ArrayIterator:

    def __init__(self, raster, window_height, window_width):
        self.raster = raster
        self.window_height = window_height
        self.window_width = window_width
        self.current_window = ((0, window_height), (0, window_width))
        self.reached_end = False

    def go_to_next(self, log=False):
        # if not yet reached end of row
        if self.current_window[1][1] < self.raster.width:
            self.current_window = ( \
                self.current_window[0], \
                (self.current_window[1][1], self.current_window[1][1] + self.window_width) \
                )
        # if reached end of the row, but not end of table
        elif self.current_window[0][1] < self.raster.height:
            if log:
                print(f'progress {round(self.current_window[0][1] / self.raster.height, 4) * 100} %')
            self.current_window = ( \
                (self.current_window[0][1], self.current_window[0][1] + self.window_height), \
                (0, self.window_width) \
                )
        # if reached end of table
        else:
            self.reached_end = True

    def pop_window(self, log=False):
        current_window = self.current_window
        self.go_to_next(log)
        return current_window

    def has_reached_end(self):
        return self.reached_end

    def reset(self):
        self.current_window = ((0, window_height), (0, window_width))
        self.reached_end = False

### For a window=((row_upper, row_lower), (col_left, col_right)), get all pixels contained in window

# https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def get_pixels(window):
    return np.array(np.meshgrid(
        np.arange(window[0][0], window[0][1]),
        np.arange(window[1][0], window[1][1]))).T.reshape(-1, 2)


def filter_bounds(array, bounds):
    check_first = np.logical_and(array[:, 0] >= bounds[0][0], array[:, 0] < bounds[0][1])
    check_second = np.logical_and(array[:, 1] >= bounds[1][0], array[:, 1] < bounds[1][1])
    return array[np.logical_and(check_first, check_second)]

print('START READING DATA')

ghs_raster = rasterio.open('prepared-classes.tif')

counts_raster = rasterio.open('pipeline-counts.tif')

# initialize iterator
window_iterator = ArrayIterator(counts_raster, 251, 71)

# initialize zeros array of same shape as COUNTS raster
classes = np.zeros(counts_raster.shape, dtype=np.uint8)


#### rasterio + numpy go brrr

# read entire data from GHS raster
start_time = time.time()
data = ghs_raster.read(1)
data = prepare_data(data)
print(np.unique(data))
print(f'reading and preparing data took: {round(time.time() - start_time)} seconds')


start_time = time.time()
print('START PIPELINE')
while not window_iterator.has_reached_end():

    window = window_iterator.pop_window(log=True)

    # read data from COUNTS raster using current window
    counts = counts_raster.read(1, window=window)

    # get all pixels contained in window
    pixels = get_pixels(window)

    # keep only those pixels for which data has a nonzero entry
    pixels = pixels[counts.ravel() > 0]

    # check if there are any pixels, if continue from next iteration
    if pixels.size > 0:
        # use COUNTS raster to get coordinates for each pixel
        counts_raster_vxy = np.vectorize(counts_raster.xy)  # gets center
        xcoords, ycoords = counts_raster_vxy(pixels[:, 0], pixels[:, 1])

        # for each coordinate get corresponding pixel in the GHS raster
        ghs_raster_vindex = np.vectorize(ghs_raster.index)
        ghs_pixels = np.vstack(ghs_raster_vindex(xcoords, ycoords, op=round, precision=15)).T

        # make sure all GHS pixels are in bounds of GHS shape
        ghs_pixels = filter_bounds(ghs_pixels, ((0, ghs_raster.height), (0, ghs_raster.width)))

        # update result
        classes[pixels[:, 0], pixels[:, 1]] = data[ghs_pixels[:, 0], ghs_pixels[:, 1]]

print('END PIPELINE')
print(f'running pipeline took: {round((time.time() - start_time) / 60, 2)} minutes\n')

print('unique classes:')
print(np.unique(data))
print(np.unique(classes))


# Compute balanced accuracy

for threshold in range(9):

    start_time = time.time()

    cm = {}
    bal_accuracy = {}
    f1 = {}
    precision = {}
    recall = {}

    truth = rasterio.open('grid3.tif').read(1)

    counts = counts_raster.read(1)
    prediction = np.zeros(counts.shape, dtype=np.uint8)
    prediction[counts > threshold] = 1

    for _class in [0,1,2]:

        mask = classes == _class

        tn, fp, fn, tp = confusion_matrix(truth[mask], prediction[mask]).ravel()
        cm[_class] = {}
        cm[_class]['TN'] = tn
        cm[_class]['FP'] = fp
        cm[_class]['FN'] = fn
        cm[_class]['TP'] = tp

        bal_accuracy[_class] = balanced_accuracy_score(truth[mask], prediction[mask])
        f1[_class] = f1_score(truth[mask], prediction[mask])
        precision[_class] = precision_score(truth[mask], prediction[mask])
        recall[_class] = recall_score(truth[mask], prediction[mask])

        print(f'metric computation took: {time.time() - start_time} seconds')

        print(f'\n\nRESULTS for THRESHOLD = {threshold}')
        print('----------\nconfusion matrix')
        print(cm)
        print('----------\nbalanced accuracy')
        print(bal_accuracy)
        print('----------\nf1')
        print(f1)
        print('----------\nprecision')
        print(precision)
        print('----------\nrecall')
        print(recall)

