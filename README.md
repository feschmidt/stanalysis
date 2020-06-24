[![GitHub Issues](https://img.shields.io/github/issues/steelelab-delft/stlabutils.svg)](https://github.com/steelelab-delft/stlabutils/issues)
[![DOCS](https://img.shields.io/badge/read%20-thedocs-ff66b4.svg)](https://steelelab-delft.github.io/stlabutils/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/181363447.svg)](https://zenodo.org/badge/latestdoi/181363447)

# stlabutils

Utilities for analysis scripts developed in the [SteeleLab at TU Delft](http://steelelab.tudelft.nl).

Documentation can be found [here](https://steelelab-delft.github.io/stlabutils/index.html).

## Easy (pip) installation

Open a command prompt and run ```pip install git+git://github.com/steelelab-delft/stlabutils.git```

## Using stlabutils from the cloned repository (recommended)

Clone the repository to your computer using, for example, the GitHub desktop client or git bash.

Install the requirements by opening a command prompt in the repository and running ```pip install -r requirements.txt```

Then add the directory you cloned it to (or any upper folder in the folder tree it is stored in) to your PYTHONPATH, using one of the following methods.

### Windows

After anaconda installation, there should be a ```PYTHONPATH``` variable in ```My Computer > Properties > Advanced System Settings > Environment Variables > ```

Add the directory in which the git repos are to this library, for example ```;C:\libs```

Taken from [here](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows) and [here](https://stackoverflow.com/questions/7054424/python-not-recognized-as-a-command).

### macOS

On Gary's mac, my GitHub desktop client stores my local repositories in:

`/Users/gsteele/Documents/GitHub`

This means that we should add this directory to the PYTHONPATH environment variable. On my mac, I added the following to my `.profile` file:

`export PYTHONPATH="$PYTHONPATH:/Users/gsteele/Documents/GitHub"`

Restarting the jupter notebook server in a shell where this environment variable is defined, I can then directly import the `stlabutils` library (along with any other libraries stored in my GitHub folder).

### Linux

Same as for macOS (both are UNIX based).

For detailed instructions on setting the python path variable on different platforms, see [this stack exchange post](https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath).

# Development

[![Build Status](https://travis-ci.com/steelelab-delft/stlabutils.svg?branch=master)](https://travis-ci.com/steelelab-delft/stlabutils)