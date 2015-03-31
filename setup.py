#!/usr/bin/env python

from distutils.core import setup

setup(name='ASAPTools',
      version='0.3',
      description='A collection of useful Python modules',
      author='Kevin Paul',
      author_email='kpaul@ucar.edu',
      packages=['asaptools'],
      requires=['mpi4py']
      )
