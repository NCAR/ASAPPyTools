#!/usr/bin/env python

from distutils.core import setup
import shutil

# Copy license file into package directory
shutil.copy('LICENSE.txt', 'source/asaptools')

setup(name='ASAPTools',
      version='0.4',
      description='A collection of useful Python modules from the '
                  'Application Scalability And Performance (ASAP) group '
                  'at the National Center for Atmospheric Research',
      author='Kevin Paul',
      author_email='kpaul@ucar.edu',
      url='https://github.com/NCAR-CISL-ASAP/ASAPPyTools',
      download_url='https://github.com/NCAR-CISL-ASAP/ASAPPyTools/tarball/v0.4',
      license='https://github.com/NCAR-CISL-ASAP/ASAPPyTools/blob/master/LICENSE.txt',
      packages=['asaptools'],
      package_dir={'asaptools': 'source/asaptools'},
      package_data={'asaptools': ['LICENSE.txt']},
      requires=['mpi4py']
      )
