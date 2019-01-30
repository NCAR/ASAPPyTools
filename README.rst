=======================
The ASAP Python Toolbox
=======================

|Circle| |Codecov| |Docs| |PyPI|

The ASAP Python Toolbox is a collection of stand-alone tools for doing simple
tasks, from managing print messages with a set verbosity level, to
keeping timing information, to managing simple MPI communication.

:AUTHORS: Kevin Paul, John Dennis, Sheri Mickelson, Haiying Xu
:COPYRIGHT: 2016-2019, University Corporation for Atmospheric Research
:LICENSE: See the LICENSE.rst file for details

Send questions and comments to Kevin Paul (kpaul@ucar.edu).

Overview
--------

The ASAP (Application Scalability And Performance) group at the National
Center for Atmospheric Research maintains this collection of simple
Python tools for managing tasks commonly used with its Python software.
The modules contained in this package include:

:vprinter: For managing print messages with verbosity-level specification
:timekeeper: For managing multiple "stop watches" for timing metrics
:partition: For various data partitioning algorithms
:simplecomm: For simple MPI communication

Only the simplecomm module depends on anything beyond the basic built-in
Python packages.

Dependencies
------------

All of the ASAP Python Toolbox tools are written to work with Python 2.6+ (including
Python 3+). The vprinter, timekeeper, and partition modules are pure
Python. The simplecomm module depends on mpi4py (>=1.3).

This implies the dependency:

- mpi4py depends on numpy (>-1.4) and MPI

Easy Installation
-----------------

The easiest way to install the ASAP Python Toolbox is from the Python
Package Index (PyPI) with the pip package manager::

    $  pip install [--user] asaptools
    
The optional '--user' argument can be used to install the package in the
local user's directory, which is useful if the user doesn't have root
privileges.

Obtaining the Source Code
-------------------------

Currently, the most up-to-date source code is available via git from the
site::

    https://github.com/NCAR/ASAPPyTools

Check out the most recent tag.  The source is available in read-only
mode to everyone, but special permissions can be given to those to make
changes to the source.

Building & Installation
-----------------------

Installation of the ASAP Python Toolbox is very simple. After checking out the
source from the above svn link, via::

    $  git clone https://github.com/NCAR/ASAPPyTools

change into the top-level source directory, check out the most recent tag,
and run the Python distutils setup. On unix, this involves::

    $  cd ASAPPyTools
    $  python setup.py install [--prefix-/path/to/install/location]

The prefix is optional, as the default prefix is typically /usr/local on
linux machines. However, you must have permissions to write to the
prefix location, so you may want to choose a prefix location where you
have write permissions. Like most distutils installations, you can
alternatively install the pyTools with the --user option, which will
automatically select (and create if it does not exist) the $HOME/.local
directory in which to install. To do this, type (on unix machines)::

    $  python setup.py install --user

This can be handy since the site-packages directory will be common for
all user installs, and therefore only needs to be added to the
PYTHONPATH once.

Instructions & Use
------------------

For instructions on how to use the ASAP Python Toolbox, see the
documentation_.


.. _documentation: https://asappytools.readthedocs.io/en/latest/

.. |Circle| image:: https://img.shields.io/circleci/project/github/NCAR/ASAPPyTools.svg?style=for-the-badge&logo=circleci
    :target: https://circleci.com/gh/NCAR/ASAPPyTools

.. |Codecov| image:: https://img.shields.io/codecov/c/github/NCAR/ASAPPyTools.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/ASAPPyTools

.. |Docs| image:: https://readthedocs.org/projects/asappytools/badge/?version=latest&style=for-the-badge
    :target: https://asappytools.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://img.shields.io/pypi/v/asaptools.svg?style=for-the-badge
    :target: https://pypi.org/project/asaptools/
    :alt: Python Package Index
