import sys
from os.path import join, dirname
sys.path.insert(0,dirname(dirname(dirname(__file__))))


import unittest
import stlabutils


class TestReaddata(unittest.TestCase):
    def test_readdata_file(self):
        # Do some stuff (with stlabutils)
        mydata = stlabutils.readdata.readdat(
            join(dirname(dirname(__file__)),'examples/data/M59_2017_06_26_16.58.40_RF_vs_power_m60dbmatt_2amp_ref_sample.dat')
        )
        expected_result = [
            'Frequency (Hz)', 'S21re ()', 'S21im ()', 'S21dB (dB)',
            'S21Ph (rad)', 'Power (dBm)', 'Temperature (K)', 'Power (K)'
        ]
        self.assertEqual(
            list(mydata[-1]), expected_result, msg='Unexpected column titles')

    def test_readdata_shape(self):
        # Do some stuff (with stlabutils)
        mydata = stlabutils.readdata.readdat(
            join(dirname(dirname(__file__)),'examples/data/M59_2017_06_26_16.58.40_RF_vs_power_m60dbmatt_2amp_ref_sample.dat')
        )
        expected_result = (1001, 8)
        self.assertEqual(
            mydata[-1].shape, expected_result, msg='Unexpected shape')


if __name__ == '__main__':
    unittest.main(verbosity=2)
