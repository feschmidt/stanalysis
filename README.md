[![Build Status](https://travis-ci.com/feschmidt/stanalysis.svg?branch=ci)](https://travis-ci.com/feschmidt/stanalysis)
[![GitHub Issues](https://img.shields.io/github/issues/feschmidt/stanalysis.svg)](https://github.com/feschmidt/stanalysis/issues)
[![DOCS](https://img.shields.io/badge/read%20-thedocs-ff66b4.svg)](https://steelelab-delft.github.io/stlab/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1299278.svg)](https://doi.org/10.5281/zenodo.1299278)

# stanalysis

Utilities for analysis scripts developed in the [SteeleLab at TU Delft](http://steelelab.tudelft.nl).

Due to regular changes of the code (stanalysis is work in progress), it is recommended to git clone this repository to your local working directory (preferably `C:\libs\stlab` on windows) and add a link to the `PYTHONPATH` instead of directly placing it in `\site-packages`.

Example data can be found in `examples/data`:

- `M59_2017_06_26_16.58.40_RF_vs_power_m60dbmatt_2amp_ref_sample.dat` is a microwave reflection measurement using a vector network analyzer, published in [Schmidt and Jenkins _et al., Nature Communications **9**, 4069 (2018)](https://www.nature.com/articles/s41467-018-06595-2)