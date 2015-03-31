#!/usr/bin/env python

from distutils.core import setup

setup(name='ASAPTools',
      version='0.3',
      description='A collection of useful Python modules from the '
                  'Application Scalability And Performance (ASAP) group '
                  'at the National Center for Atmospheric Research',
      author='Kevin Paul',
      author_email='kpaul@ucar.edu',
      packages=['asaptools'],
      requires=['mpi4py']
      )
