import pytest
from pathlib import Path
import numpy as np

from utils.helpers import get_project_path, prepare_data, get_pixels, filter_bounds

class TestHelpers:

    def test_get_project_path(self):
        assert get_project_path() == Path(__file__).parent.parent.absolute()


    def test_prepare_data(self):
        data = np.tile(np.array([np.nan, 0, 0.1, 1, 2]),2).reshape((2,5))
        expected = np.array([[0,0,1,1,1],
                             [0,0,1,1,1]],dtype=np.uint8)
        assert np.all(prepare_data(data) == expected)


    def test_get_pixels(self):
        window = ((0,3),(5,8))
        expected = np.array([[0,5],
                             [0,6],
                             [0,7],
                             [1,5],
                             [1,6],
                             [1,7],
                             [2,5],
                             [2,6],
                             [2,7]])
        assert np.all(get_pixels(window) == expected)


    def test_filter_bounds(self):
        assert filter_bounds(np.ones((10,2)), ((0,1),(0,1))).size == 0
        assert filter_bounds(np.ones((10,2)), ((0,2),(0,1))).size == 0
        assert filter_bounds(np.ones((10,2)), ((0,2),(0,2))).size == 20
        assert (filter_bounds(np.array([[1,2],[2,2],[2,1],[1,1]]), ((2,3),(2,3))).size == np.array([[2,2]])).all()
