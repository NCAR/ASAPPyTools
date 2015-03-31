=======================
The ASAP Python Toolbox
=======================

The ASAP Python Toolbox is a collection of stand-alone tools for doing simple
tasks, from managing print messages with a set verbosity level, to
keeping timing information, to managing simple MPI communication.

:AUTHORS: Kevin Paul, John Dennis, Sheri Mickelson, Haiying Xu
:VERSION: 0.3
:COPYRIGHT: See the document entitled LICENSE.txt

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

All of the ASAP PyTools are written to work with Python 2.6+ (but not
Python 3.0+). The vprinter, timekeeper, and partition modules are pure
Python. The simplecomm module depends on mpi4py (>-1.3).

This implies the dependency:

- mpi4py depends on numpy (>-1.4) and MPI

Obtaining the Source Code
-------------------------

Currently, the most up-to-date source code is available via svn from the
site::

    https://proxy.subversion.ucar.edu/pubasap/pyTools/tags/v0.3

The source is available in read-only mode to everyone, but special
permissions can be given to those to make changes to the source.

Building & Installation
-----------------------

Installation of the PyTools is very simple. After checking out the
source from the above svn link, via::

    svn co https://proxy.subversion.ucar.edu/pubasap/pyTools/tags/v0.3 ASAPTools

change into the top-level source directory and run the Python distutils
setup. On unix, this involves::

    $  cd ASAPTools
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

To install API documentation for developer use, you must run doxygen
with the command (on unix machines)::

    $  doxygen Doxyfile

If doxypypy is found on the system, then the Doxypypy filter is used to
generate the API documentation. Otherwise, the raw Python files will be
interpretted directly by Doxygen. The resulting API documentation will
be placed in the docs/api directory.

Before Using the ASAP Python Toolbox
------------------------------------

Before the PyTools package can be used, you must make sure that the
site-packages directory containing the 'pytools' source directory is in
your PYTHONPATH. Depending on the PREFIX used during installation, this
path will be::

    $PREFIX/lib/python2.X/site-packages

where X will be 6 or 7 (or other) depending on the version of Python
that you are using to install the package.

Instructions & Use
------------------

For instructions on how to use the PyTools, see the additional documents
found in the docs/api and docs/user directories.

If you are a developer wanting to use the PyTools API directly from your
own Python code, please read the 'Building & Installation' section above
for instructions on how to build the API documentation. Once built, you
will be able to open the 'docs/api/html/index.html' page in any browser.

The docs/user directory contains user manual describing how to use the
different modules in the PyTools package. Both this README and the User
Manual are written in reStructuredText, and can easily be converted to HTML or
many other formats with the help of a tool such as pandoc or Sphinx.
