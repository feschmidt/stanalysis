import unittest
import stlabutils


class TestTesting(unittest.TestCase):
    def test_testing(self):
        # Do some stuff (with stlabutils)
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
