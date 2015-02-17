'''
Parallel-1-Serial Tests for the SimpleComm class

The 'P1S' Test Suite specificially tests whether the serial behavior is the
same as the 1-rank parallel behavior.  If the 'Par' test suite passes with
various communicator sizes (1, 2, ...), then this suite should be run to make
sure that serial communication behaves consistently.

_______________________________________________________________________________
Created on Feb 17, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''
import unittest
import simplecomm
import numpy as np
from partfunc import equal_stride
from os import linesep
from mpi4py import MPI
MPI_COMM_WORLD = MPI.COMM_WORLD

def test_info_msg(name, data, sresult, presult):
    spcr = ' ' * len(name)
    msg = ''.join([linesep,
                   name, ' - Input: ', str(data), linesep,
                   spcr, ' - Serial Result:   ', str(sresult), linesep,
                   spcr, ' - Parallel Result: ', str(presult)])
    return msg


class SimpleCommP1STests(unittest.TestCase):

    def setUp(self):
        self.scomm = simplecomm.SimpleComm(mpi=False)
        self.pcomm = simplecomm.SimpleComm(mpi=True)
        self.size = MPI_COMM_WORLD.Get_size()
        self.rank = MPI_COMM_WORLD.Get_rank()

    def tearDown(self):
        pass

    def testIsSerialLike(self):
        self.assertEqual(self.rank, 0, 'Rank not consistent with serial-like operation')
        self.assertEqual(self.size, 1, 'Size not consistent with serial-like operation')

    def testGetSize(self):
        sresult = self.scomm.get_size()
        presult = self.pcomm.get_size()
        msg = test_info_msg('get_size()', None, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testIsMaster(self):
        sresult = self.scomm.is_master()
        presult = self.pcomm.is_master()
        msg = test_info_msg('is_master()', None, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testSumInt(self):
        data = 5
        sresult = self.scomm.reduce(data)
        presult = self.pcomm.reduce(data)
        msg = test_info_msg('sum(int)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testSumList(self):
        data = range(5)
        sresult = self.scomm.reduce(data)
        presult = self.pcomm.reduce(data)
        msg = test_info_msg('sum(list)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testSumArray(self):
        data = np.arange(5)
        sresult = self.scomm.reduce(data)
        presult = self.pcomm.reduce(data)
        msg = test_info_msg('sum(array)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testMaxInt(self):
        data = 13 + self.rank
        sresult = self.scomm.reduce(data, op=max)
        presult = self.pcomm.reduce(data, op=max)
        msg = test_info_msg('max(int)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testMaxList(self):
        data = range(5 + self.rank)
        sresult = self.scomm.reduce(data, op=max)
        presult = self.pcomm.reduce(data, op=max)
        msg = test_info_msg('max(list)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testMaxArray(self):
        data = np.arange(5 + self.rank)
        sresult = self.scomm.reduce(data, op=max)
        presult = self.pcomm.reduce(data, op=max)
        msg = test_info_msg('max(array)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testGatherInt(self):
        if self.pcomm.is_master():
            data = None
            sresult = self.scomm.gather()
            presult = self.pcomm.gather()
        else:
            data = 10
            sresult = self.scomm.gather(data)
            presult = self.pcomm.gather(data)
        msg = test_info_msg('Gather(int)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testGatherList(self):
        data = range(3 + self.rank)
        sresult = self.scomm.gather(data)
        presult = self.pcomm.gather(data)
        msg = test_info_msg('Gather(list)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testScatterInt(self):
        if self.pcomm.is_master():
            data = 100
            sresult = self.scomm.scatter(data)
            presult = self.pcomm.scatter(data)
        else:
            data = None
            sresult = self.scomm.scatter()
            presult = self.pcomm.scatter()
        msg = test_info_msg('Scatter(int)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testScatterList(self):
        if self.pcomm.is_master():
            data = range(10)
            sresult = self.scomm.scatter(data, part=equal_stride)
            presult = self.pcomm.scatter(data, part=equal_stride)
        else:
            data = None
            sresult = self.scomm.scatter()
            presult = self.pcomm.scatter()
        msg = test_info_msg('Scatter(list)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)

    def testScatterListSkip(self):
        if self.pcomm.is_master():
            data = range(10)
            sresult = self.scomm.scatter(data, part=equal_stride, skip=True)
            presult = self.pcomm.scatter(data, part=equal_stride, skip=True)
        else:
            data = None
            sresult = self.scomm.scatter()
            presult = self.pcomm.scatter()
        msg = test_info_msg('Scatter(list, skip)', data, sresult, presult)
        print msg
        self.assertEqual(sresult, presult, msg)


if __name__ == "__main__":
    hline = '=' * 70
    if MPI_COMM_WORLD.Get_rank() == 0:
        print hline
        print 'STANDARD OUTPUT FROM ALL TESTS:'
        print hline
    MPI_COMM_WORLD.Barrier()

    from cStringIO import StringIO
    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(SimpleCommP1STests)
    unittest.TextTestRunner(stream=mystream).run(tests)
    MPI_COMM_WORLD.Barrier()

    results = MPI_COMM_WORLD.gather(mystream.getvalue())
    if MPI_COMM_WORLD.Get_rank() == 0:
        for rank, result in enumerate(results):
            print hline
            print 'TESTS RESULTS FOR RANK ' + str(rank) + ':'
            print hline
            print str(result)
