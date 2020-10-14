import unittest
from unittest import mock
from unittest.mock import patch

import os.path as path
import json

from src.utils.definitions import get_project_path
from src.utils.location import GeoLocation


class GeoLocationTest(unittest.TestCase):

    def setUp(self):
        with open(path.join(get_project_path(), "test", "data", "ny_coord.json")) as file:
            self.gt_coords_single = json.load(file)
        with open(path.join(get_project_path(), "test", "data", "list_coord.json")) as file:
            self.gt_coords_list = json.load(file)
        return


    def tearDown(self):
        pass


    def compare_coords(self, coord, gt_coord):
        self.assertIsInstance(coord['lat'], float)
        self.assertIsInstance(coord['lon'], float)
        self.assertAlmostEqual(coord['lat'], float(gt_coord['lat']), delta=0.01)
        self.assertAlmostEqual(coord['lon'], float(gt_coord['lon']), delta=0.01)
        for i in range(4):
            self.assertIsInstance(coord['bbox'][i], float)
            self.assertAlmostEqual(coord['bbox'][i], float(gt_coord['boundingbox'][i]), delta=0.01)


    @patch('src.utils.location.Nominatim.geocode')
    def test_get_coordinates_single_name(self, mocked):
        mocked.return_value.raw = self.gt_coords_single
        geo = GeoLocation('testagent1')
        coord = geo.get_coordinates('New York')
        mocked.assert_called_once()
        self.compare_coords(coord, self.gt_coords_single)


    def give_coords(self, name):
        m = mock.Mock()
        m.raw = self.gt_coords_list[name]
        return m


    @patch('src.utils.location.Nominatim.geocode')
    def test_get_coordinates_list(self, mocked):
        mocked.side_effect = self.give_coords
        geo = GeoLocation('testagent2')
        coords = geo.get_coordinates(['New York', 'Hong Kong', 'Warszawa', 'Berlin'])
        self.assertEqual(mocked.call_count, 4)
        for place, coord in coords.items():
            self.compare_coords(coord, self.gt_coords_list[place])


if __name__ == "__main__":
    unittest.main()
