[![GitHub Issues](https://img.shields.io/github/issues/steelelab-delft/stlabutils.svg)](https://github.com/steelelab-delft/stlabutils/issues)
[![DOCS](https://img.shields.io/badge/read%20-thedocs-ff66b4.svg)](https://steelelab-delft.github.io/stlabutils/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1299278.svg)](https://doi.org/10.5281/zenodo.1299278)

# stlabutils

Utilities for analysis scripts developed in the [SteeleLab at TU Delft](http://steelelab.tudelft.nl).

Due to regular changes of the code (stanalysis is work in progress), it is recommended to git clone this repository to your local computer. See instructions below.

Example data can be found in `examples/data`:

- `M59_2017_06_26_16.58.40_RF_vs_power_m60dbmatt_2amp_ref_sample.dat` is a microwave reflection measurement using a vector network analyzer, published in [Schmidt and Jenkins _et al., Nature Communications **9**, 4069 (2018)](https://www.nature.com/articles/s41467-018-06595-2)

# Installation instructions

Clone the repository to your computer using, for example, the GitHub desktop client. 

Then add the directory you cloned it to (or any upper folder in the folder tree it is stored in) to your python path. On Gary's mac, my GitHub desktop client stores my local repositories in:

`/Users/gsteele/Documents/GitHub`

This means that we should add this directory to the PYTHONPATH environment variable. On my mac, I added the following to my `.profile` file:

`export PYTHONPATH="$PYTHONPATH:/Users/gsteele/Documents/GitHub"`

Restarting the jupter notebook server in a shell where this environment variable is defined, I can then directly import the `stlabutils` library (along with any other libraries stored in my GitHub folder). 

For detailed instructions on setting the python path variable on different platforms, see this stack exchange post:

https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath
