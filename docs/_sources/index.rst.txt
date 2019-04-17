.. Test documentation master file, created by
   sphinx-quickstart on Mon Sep 10 12:04:37 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for STLabutils
=======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:

   utils/*

Introduction
============

STLabutils is a collection of tool scripts used for file creation, reading and equipment control and analysis.
It is used together with STLab for measurements performed in the `SteeleLab at TU Delft <http://steelelab.tudelft.nl>`_.
While STLab is necessary for performing measurements and communication with instruments, STLabutils is a standalone
library which can be used by itself to read and analyse datafiles, and perform certain simulations.
STLabutils were formerly contained within :code:`stlab.utils`.

The basic structure of the package is as follows:

| stlab
| ├── __init__.py
| ├── LICENCE.md
| ├── README.md
| ├── utils
| │   ├── 
| │   └── ...
| ├── examples
| │   ├── ...
| │   └── ...
| ├── docs
| │   ├── ...
| │   └── ...
| ├── doc_gen
| │   ├── ...
| │   └── ...

* The "utils" folder contains modules for reading and writing files, resonance fitting, data structure management (stlabmtx for example).
  These packages were formely contained within :code:`stlab.utils`.
* "examples" contains a collection of basic examples such as example data and Q factor fits.
* "docs" contains this documentation and "doc_gen" contains the sphynx scripts for generating it.
* The __init__.py file contains the modules and names imported when running "import stlabutils".  Note that some modules and functions are renamed for (in?)convenience.

The imports done when doing :code:`import stlabutils` are:

.. literalinclude:: ../__init__.py
  :language: python

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

stlabutils is licensed under the `GNU General Public License v3.0 
<https://www.gnu.org/licenses/gpl-3.0.en.html>`_.