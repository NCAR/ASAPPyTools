Change Log
==========

Copyright 2019 University Corporation for Atmospheric Research

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Version 0.6.0
-------------

- Allowing for support of all Python 2.6+ (including Python 3+)

Version 0.5.4
-------------

- Bugfix: Special catch for dtype='c' (C-type char arrays) in check for 
  Numpy arrays being bufferable

Version 0.5.3
-------------

- Updates just for PyPI release

Version 0.5.2
-------------

- Improved testing for send/recv data types
- Backwards compatability with mpi4py version 1.3.1 

Version 0.5.1
-------------

- Checking dtype of Numpy NDArrays before determing if buffered send/recv
  calls can be used.
 
Version 0.5.0
-------------

- Now requires Python >=2.7 and <3.0
- Using more advanced features of Python 2.7 (over 2.6)
- Changed Numpy NDArray type-checking to allow for masked arrays, instead of
  just NDArrays

Version 0.4.2
-------------

- Update setup script to setuptools (instead of distutils)

Version 0.4.1
-------------

- Bugfixes

Version 0.4
-----------

- Updating install to include LICENSE
- Restructured source directory
- Upload to PyPI

Version 0.3
-----------

- Repackaging the pyTools repo into a Python package with
  installation software and Sphinx-style documentation
