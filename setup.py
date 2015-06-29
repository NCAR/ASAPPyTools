#!/usr/bin/env python
"""
ASAP Python Toolbox -- Setup Script

Copyright 2015, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from setuptools import setup

setup(name='ASAPTools',
      version='0.4.2',
      description='A collection of useful Python modules from the '
                  'Application Scalability And Performance (ASAP) group '
                  'at the National Center for Atmospheric Research',
      author='Kevin Paul',
      author_email='kpaul@ucar.edu',
      url='https://github.com/NCAR-CISL-ASAP/ASAPPyTools',
      download_url='https://github.com/NCAR-CISL-ASAP/ASAPPyTools/tarball/v0.4.1',
      license='https://github.com/NCAR-CISL-ASAP/ASAPPyTools/blob/master/LICENSE.txt',
      packages=['asaptools'],
      package_dir={'asaptools': 'source/asaptools'},
      package_data={'asaptools': ['LICENSE.txt']},
      install_requires=['mpi4py>=1.3']
      )
