import pytest
import numpy as np

from utils.iterator import ArrayIterator


class FakeRaster:

    def __init__(self, array):
        self.height, self.width = array.shape


class TestArrayIterator:

    def test__init__(self):
        iterator = ArrayIterator(None, 10, 15)
        assert not iterator.reached_end
        assert iterator.current_window[0][0] == 0
        assert iterator.current_window[0][1] == 10
        assert iterator.current_window[1][0] == 0
        assert iterator.current_window[1][1] == 15


    def test_go_to_next_same_row(self):
        iterator = ArrayIterator(FakeRaster(np.arange(100).reshape(10,10)), 5, 5)
        iterator.go_to_next()
        assert iterator.current_window[0][0] == 0
        assert iterator.current_window[0][1] == 5
        assert iterator.current_window[1][0] == 5
        assert iterator.current_window[1][1] == 10


    def test_go_to_next_next_row(self):
        iterator = ArrayIterator(FakeRaster(np.arange(100).reshape(10,10)), 5, 5)
        iterator.go_to_next()
        iterator.go_to_next()
        assert iterator.current_window[0][0] == 5
        assert iterator.current_window[0][1] == 10
        assert iterator.current_window[1][0] == 0
        assert iterator.current_window[1][1] == 5


    def test_go_to_next_reached_end(self):
        iterator = ArrayIterator(FakeRaster(np.arange(100).reshape(10,10)), 5, 5)
        iterator.go_to_next()
        iterator.go_to_next()
        iterator.go_to_next()
        iterator.go_to_next()
        assert iterator.has_reached_end()


    def test_pop_window(self):
        iterator = ArrayIterator(FakeRaster(np.arange(100).reshape(10,10)), 5, 5)
        window = iterator.pop_window()
        assert iterator.current_window[0][0] == 0
        assert iterator.current_window[0][1] == 5
        assert iterator.current_window[1][0] != 0
        assert iterator.current_window[1][1] != 5
        assert window[0][0] == 0
        assert window[0][1] == 5
        assert window[1][0] == 0
        assert window[1][1] == 5


    def test_reset(self):
        iterator = ArrayIterator(FakeRaster(np.arange(100).reshape(10,10)), 5, 5)
        iterator.go_to_next()
        iterator.go_to_next()
        iterator.go_to_next()
        iterator.go_to_next()
        iterator.reset()
        assert not iterator.has_reached_end()
        assert iterator.current_window[0][0] == 0
        assert iterator.current_window[0][1] == 5
        assert iterator.current_window[1][0] == 0
        assert iterator.current_window[1][1] == 5
