import sys
from os.path import join, dirname
# sys.path.insert(0,join(join(dirname(dirname(__file__)),'utils')))
sys.path.insert(0,dirname(dirname(dirname(__file__))))

import unittest
import stlabutils
import numpy as np
 

class TestS11fit(unittest.TestCase):
    def test_S11fit_f0(self):
        mydata = stlabutils.readdata.readdat(
            'examples/data/M59_2017_06_26_16.58.40_RF_vs_power_m60dbmatt_2amp_ref_sample.dat'
        )
        # Convert S parameter data from Re,Im to complex array
        z = mydata[-1]['S21re ()'] + 1j * mydata[-1]['S21im ()']
        x = mydata[-1]['Frequency (Hz)']
        # Do fit with some given parameters.  More options available.
        fitres, _, _, _ = stlabutils.S11fit(
            x, z, ftype='A'
        )  # , doplots=False, trimwidth=3., fitwidth=5., margin=51)
        f0 = fitres['f0'].value
        expected_f0 = 8108901094
        self.assertAlmostEqual(
            f0, expected_f0, delta=1,
            msg='Unexpected resonance frequency')  # delta=1 seems arbitrary?

    def test_S11fit_deviation(self):
        mydata = stlabutils.readdata.readdat(
            'examples/data/M59_2017_06_26_16.58.40_RF_vs_power_m60dbmatt_2amp_ref_sample.dat'
        )
        # Convert S parameter data from Re,Im to complex array
        z = mydata[-1]['S21re ()'] + 1j * mydata[-1]['S21im ()']
        x = mydata[-1]['Frequency (Hz)']
        # Do fit with some given parameters.  More options available.
        fitres, _, _, _ = stlabutils.S11fit(
            x, z, ftype='A'
        )  # , doplots=False, trimwidth=3., fitwidth=5., margin=51)
        zfit = stlabutils.S11func(
            mydata[-1]['Frequency (Hz)'], fitres, ftype='A')
        thedata = mydata[-1]['S21dB (dB)']
        thefit = 20 * np.log10(np.abs(zfit))
        rmse = np.sqrt(np.sum((thedata - thefit)**2))
        # delta=1 seems arbitrary?
        self.assertLess(rmse, 1, msg='Fit and data deviate significantly')


if __name__ == '__main__':
    unittest.main(verbosity=2)
