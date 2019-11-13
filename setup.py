#!/usr/bin/env python
"""
ASAP Python Toolbox -- Setup Script


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
"""

from setuptools import setup
import versioneer

setup(name='ASAPTools',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A collection of useful Python modules from the '
                  'Application Scalability And Performance (ASAP) group '
                  'at the National Center for Atmospheric Research',
      author='Kevin Paul',
      author_email='kpaul@ucar.edu',
      url='https://github.com/NCAR/ASAPPyTools',
      license='https://github.com/NCAR/ASAPPyTools/blob/master/LICENSE.txt',
      packages=['asaptools'],
      package_dir={'asaptools': 'asaptools'},
      package_data={'asaptools': ['LICENSE.txt']},
      install_requires=['mpi4py>=1.3']
      )
