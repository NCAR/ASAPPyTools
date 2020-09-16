#!/usr/bin/env python
"""
ASAP Python Toolbox -- Setup Script


Copyright 2020 University Corporation for Atmospheric Research

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

from os.path import exists

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''

CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
]

setup(
    name='asaptools',
    description='A collection of useful Python modules from the '
    'Application Scalability And Performance (ASAP) group '
    'at the National Center for Atmospheric Research',
    long_description=long_description,
    python_requires='>=3.6',
    maintainer='Kevin Paul',
    maintainer_email='kpaul@ucar.edu',
    classifiers=CLASSIFIERS,
    url='https://asappytools.readthedocs.io',
    project_urls={
        'Documentation': 'https://asappytools.readthedocs.io',
        'Source': 'https://github.com/NCAR/ASAPPyTools',
        'Tracker': 'https://github.com/NCAR/ASAPPyTools/issues',
    },
    packages=find_packages(),
    package_dir={'asaptools': 'asaptools'},
    include_package_data=True,
    install_requires=install_requires,
    license='Apache 2.0',
    zip_safe=False,
    keywords='mpi',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
)
