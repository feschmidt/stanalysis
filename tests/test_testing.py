import sys
import os
# sys.path.append(os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)),'src')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import unittest
import stlabutils


class TestTesting(unittest.TestCase):
    def test_testing(self):
        # Do some stuff (with stlabutils)
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
