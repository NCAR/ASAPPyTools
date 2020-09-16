"""
Parallel-1-Serial Tests for the SimpleComm class

The 'P1S' Test Suite specificially tests whether the serial behavior is the
same as the 1-rank parallel behavior.  If the 'Par' test suite passes with
various communicator sizes (1, 2, ...), then this suite should be run to make
sure that serial communication behaves consistently.

Copyright 2017, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from __future__ import print_function

import unittest

import numpy as np
from mpi4py import MPI

from asaptools import simplecomm
from asaptools.partition import Duplicate, EqualStride

MPI_COMM_WORLD = MPI.COMM_WORLD


class SimpleCommP1STests(unittest.TestCase):
    def setUp(self):
        self.scomm = simplecomm.create_comm(serial=True)
        self.pcomm = simplecomm.create_comm(serial=False)
        self.size = MPI_COMM_WORLD.Get_size()
        self.rank = MPI_COMM_WORLD.Get_rank()

    def testIsSerialLike(self):
        self.assertEqual(self.rank, 0, 'Rank not consistent with serial-like operation')
        self.assertEqual(self.size, 1, 'Size not consistent with serial-like operation')

    def testGetSize(self):
        sresult = self.scomm.get_size()
        presult = self.pcomm.get_size()
        self.assertEqual(sresult, presult)

    def testIsManager(self):
        sresult = self.scomm.is_manager()
        presult = self.pcomm.is_manager()
        self.assertEqual(sresult, presult)

    def testSumInt(self):
        data = 5
        sresult = self.scomm.allreduce(data, 'sum')
        presult = self.pcomm.allreduce(data, 'sum')
        self.assertEqual(sresult, presult)

    def testSumList(self):
        data = range(5)
        sresult = self.scomm.allreduce(data, 'sum')
        presult = self.pcomm.allreduce(data, 'sum')
        self.assertEqual(sresult, presult)

    def testSumDict(self):
        data = {'rank': self.rank, 'range': range(3 + self.rank)}
        sresult = self.scomm.allreduce(data, 'sum')
        presult = self.pcomm.allreduce(data, 'sum')
        self.assertEqual(sresult, presult)

    def testSumArray(self):
        data = np.arange(5)
        sresult = self.scomm.allreduce(data, 'sum')
        presult = self.pcomm.allreduce(data, 'sum')
        self.assertEqual(sresult, presult)

    def testMaxInt(self):
        data = 13 + self.rank
        sresult = self.scomm.allreduce(data, 'max')
        presult = self.pcomm.allreduce(data, 'max')
        self.assertEqual(sresult, presult)

    def testMaxList(self):
        data = range(5 + self.rank)
        sresult = self.scomm.allreduce(data, 'max')
        presult = self.pcomm.allreduce(data, 'max')
        self.assertEqual(sresult, presult)

    def testMaxDict(self):
        data = {'rank': self.rank, 'range': range(3 + self.rank)}
        sresult = self.scomm.allreduce(data, 'max')
        presult = self.pcomm.allreduce(data, 'max')
        self.assertEqual(sresult, presult)

    def testMaxArray(self):
        data = np.arange(5 + self.rank)
        sresult = self.scomm.allreduce(data, 'max')
        presult = self.pcomm.allreduce(data, 'max')
        self.assertEqual(sresult, presult)

    def testPartitionInt(self):
        data = 13 + self.rank
        sresult = self.scomm.partition(data, func=Duplicate())
        presult = self.pcomm.partition(data, func=Duplicate())
        self.assertEqual(sresult, presult)

    def testPartitionIntInvolved(self):
        data = 13 + self.rank
        sresult = self.scomm.partition(data, func=Duplicate(), involved=True)
        presult = self.pcomm.partition(data, func=Duplicate(), involved=True)
        self.assertEqual(sresult, presult)

    def testPartitionList(self):
        data = range(5 + self.rank)
        sresult = self.scomm.partition(data, func=EqualStride())
        presult = self.pcomm.partition(data, func=EqualStride())
        self.assertEqual(sresult, presult)

    def testPartitionListInvolved(self):
        data = range(5 + self.rank)
        sresult = self.scomm.partition(data, func=EqualStride(), involved=True)
        presult = self.pcomm.partition(data, func=EqualStride(), involved=True)
        self.assertEqual(sresult, presult)

    def testPartitionArray(self):
        data = np.arange(2 + self.rank)
        sresult = self.scomm.partition(data)
        presult = self.pcomm.partition(data)
        self.assertEqual(sresult, presult)

    def testPartitionArrayInvolved(self):
        data = np.arange(2 + self.rank)
        sresult = self.scomm.partition(data, involved=True)
        presult = self.pcomm.partition(data, involved=True)
        np.testing.assert_array_equal(sresult, presult)

    def testPartitionStrArray(self):
        data = np.array([c for c in 'abcdefghijklmnopqrstuvwxyz'])
        sresult = self.scomm.partition(data)
        presult = self.pcomm.partition(data)
        self.assertEqual(sresult, presult)

    def testPartitionStrArrayInvolved(self):
        data = np.array([c for c in 'abcdefghijklmnopqrstuvwxyz'])
        sresult = self.scomm.partition(data, involved=True)
        presult = self.pcomm.partition(data, involved=True)
        np.testing.assert_array_equal(sresult, presult)

    def testRationError(self):
        data = 10
        self.assertRaises(RuntimeError, self.scomm.ration, data)
        self.assertRaises(RuntimeError, self.pcomm.ration, data)

    def testCollectError(self):
        data = 10
        self.assertRaises(RuntimeError, self.scomm.collect, data)
        self.assertRaises(RuntimeError, self.pcomm.collect, data)


if __name__ == '__main__':
    try:
        from cStringIO import StringIO
    except ImportError:
        from io import StringIO

    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(SimpleCommP1STests)
    unittest.TextTestRunner(stream=mystream).run(tests)
    MPI_COMM_WORLD.Barrier()

    results = MPI_COMM_WORLD.gather(mystream.getvalue())
    if MPI_COMM_WORLD.Get_rank() == 0:
        for rank, result in enumerate(results):
            print('RESULTS FOR RANK ' + str(rank) + ':')
            print(str(result))
