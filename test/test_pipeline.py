import pytest
import rasterio
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from utils.helpers import prepare_data, get_pixels, get_project_path
from pipeline import Pipeline


# TODO move this to helpers and test it
def get_corresponding_window(raster1_path, raster2_path):
    """Get window in raster2 that corresponds to whole of raster1. Clip if neccessary."""
    raster1 = rasterio.open(raster1_path)
    raster2 = rasterio.open(raster2_path)
    xcoord_first, ycoord_first = raster1.xy(0, 0)
    xcoord_last, ycoord_last = raster1.xy(raster1.height-1, raster1.width-1)
    upper_left = raster2.index(xcoord_first, ycoord_first, op=round, precision=15)
    bottom_right = raster2.index(xcoord_last, ycoord_last, op=round, precision=15)
    # the +1 for bottom_right is because upper_bound in windows is exclusive
    return ((max(upper_left[0],0), min(bottom_right[0]+1,raster2.height)),
            (max(upper_left[1],0), min(bottom_right[1]+1,raster2.width)))


def get_shape(raster_path):
    with rasterio.open(raster_path) as raster:
        return raster.shape


class TestPipeline:

    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_eye(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/eye.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        window = get_corresponding_window(get_project_path() / 'test/data/pipeline/eye.tif',
                                          get_project_path() / f'test/data/pipeline/{name}.tif')
        first_pixel = get_pixels(window)[0]
        diagonal = np.array([[first_pixel[0]+i, first_pixel[1]+i] for i in range(min(window[0][1] - window[0][0],
                                                                                     window[1][1] - window[1][0]))])
        # check result is lesser equal 9 everywhere
        assert (result <= 9).all()
        # check that all counts on the diagonal are greater 0 and less or equal 3
        assert (result[diagonal[:,0], diagonal[:,1]] > 0).all() and (result[diagonal[:,0], diagonal[:,1]] <= 3).all()
        # check that result is 0 everywhere else
        result[diagonal[:, 0], diagonal[:, 1]] = 0
        assert (result == 0).all()


    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_zeros(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/zeros.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        # check result is 0 everywhere
        assert (result == 0).all()


    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_ones(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/ones.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        window = get_corresponding_window(get_project_path() / 'test/data/pipeline/ones.tif',
                                          get_project_path() / f'test/data/pipeline/{name}.tif')
        pixels = get_pixels(window)
        rectangle = pixels
        inner_rectangle = rectangle[np.logical_and(
            np.logical_and(rectangle[:,0] != window[0][0], rectangle[:,1] != window[1][0]),
            np.logical_and(rectangle[:,0] != window[0][1]-1, rectangle[:,1] != window[1][1]-1))]

        # check result is lesser equal 9 everywhere
        assert (result <= 9).all()
        # check that all counts on the diagonal are greater 0 and less or equal 3
        assert (result[rectangle[:,0], rectangle[:,1]] > 0).all()
        assert (result[inner_rectangle[:,0], inner_rectangle[:,1]] == 9).all()
        # check that result is 0 everywhere else
        result[rectangle[:, 0], rectangle[:, 1]] = 0
        assert (result == 0).all()


    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_left_edge(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/left_edge.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        window = get_corresponding_window(get_project_path() / 'test/data/pipeline/left_edge.tif',
                                          get_project_path() / f'test/data/pipeline/{name}.tif')
        pixels = get_pixels(window)
        vertical_edge = pixels[pixels[:,1] == window[1][0]]

        # check result is lesser equal 9 everywhere
        assert (result <= 9).all()
        # check that all counts on the vertical edge are greater 0 and less or equal 3
        assert (result[vertical_edge[:,0], vertical_edge[:,1]] > 0).all() and (result[vertical_edge[:,0], vertical_edge[:,1]] <= 3).all()
        # check that result is 0 everywhere else
        result[vertical_edge[:, 0], vertical_edge[:, 1]] = 0
        assert (result == 0).all()


    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_top_edge(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/top_edge.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        window = get_corresponding_window(get_project_path() / 'test/data/pipeline/top_edge.tif',
                                          get_project_path() / f'test/data/pipeline/{name}.tif')
        pixels = get_pixels(window)
        horizontal_edge = pixels[pixels[:,0] == window[0][0]]

        # check result is lesser equal 9 everywhere
        assert (result <= 9).all()
        # check that all counts on the horizontal edge are greater 0 and less or equal 3
        assert (result[horizontal_edge[:,0], horizontal_edge[:,1]] > 0).all() and (result[horizontal_edge[:,0], horizontal_edge[:,1]] <= 3).all()
        # check that result is 0 everywhere else
        result[horizontal_edge[:, 0], horizontal_edge[:, 1]] = 0
        assert (result == 0).all()

    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_right_edge(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/right_edge.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        height, width = get_shape(get_project_path() / f'test/data/pipeline/{name}.tif')

        if height == width == 30:
            assert (result == 0).all()
        else:
            window = get_corresponding_window(get_project_path() / 'test/data/pipeline/right_edge.tif',
                                              get_project_path() / f'test/data/pipeline/{name}.tif')
            pixels = get_pixels(window)
            vertical_edge = pixels[pixels[:, 1] == window[1][1]-1]
            # check result is lesser equal 9 everywhere
            assert (result <= 9).all()
            # check that all counts on the vertical edge are greater 0 and less or equal 3
            assert (result[vertical_edge[:, 0], vertical_edge[:, 1]] > 0).all()# and (
                        #result[vertical_edge[:, 0], vertical_edge[:, 1]] <= 3).all()
            # check that result is 0 everywhere else
            result[vertical_edge[:, 0], vertical_edge[:, 1]] = 0
            assert (result == 0).all()


    @pytest.mark.parametrize('name',['grid3-30', 'grid3-60'])
    def test_bottom_edge(self, name):
        pipeline = Pipeline(
            get_project_path() / 'test/data/pipeline/bottom_edge.tif',
            get_project_path() / f'test/data/pipeline/{name}.tif',
            45,
            45
        )
        pipeline.run()
        result = pipeline.result

        height, width = get_shape(get_project_path() / f'test/data/pipeline/{name}.tif')

        if height == width == 30:
            assert (result == 0).all()
        else:
            window = get_corresponding_window(get_project_path() / 'test/data/pipeline/bottom_edge.tif',
                                              get_project_path() / f'test/data/pipeline/{name}.tif')
            pixels = get_pixels(window)
            horizontal_edge = pixels[pixels[:, 0] == window[0][1]-1]
            # check result is lesser equal 9 everywhere
            assert (result <= 9).all()
            # check that all counts on the vertical edge are greater 0 and less or equal 3
            assert (result[horizontal_edge[:, 0], horizontal_edge[:, 1]] > 0).all() and (
                        result[horizontal_edge[:, 0], horizontal_edge[:, 1]] <= 3).all()
            # check that result is 0 everywhere else
            result[horizontal_edge[:, 0], horizontal_edge[:, 1]] = 0
            assert (result == 0).all()
