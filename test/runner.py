import unittest

import test.utils.suite
import test.metrics


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(test.utils.suite.get_suite())
    result = runner.run(test.metrics.get_suite())
