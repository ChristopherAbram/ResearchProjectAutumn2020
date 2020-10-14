import unittest

import test.utils.location as location
import test.utils.raster as raster


def get_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(location))
    suite.addTests(loader.loadTestsFromModule(raster))
    return suite
