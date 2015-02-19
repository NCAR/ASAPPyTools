'''
Parallel Tests for the SimpleComm class

_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''
import unittest
import simplecomm
import numpy as np
from partfunc import equal_stride
from os import linesep as eol
from mpi4py import MPI
MPI_COMM_WORLD = MPI.COMM_WORLD

def test_info_msg(rank, size, name, data, actual, expected):
    rknm = ''.join(['[', str(rank), '/', str(size), '] ', str(name)])
    spcr = ' ' * len(rknm)
    msg = ''.join([eol,
                   rknm, ' - Input: ', str(data), eol,
                   spcr, ' - Actual:   ', str(actual), eol,
                   spcr, ' - Expected: ', str(expected)])
    return msg


class SimpleCommParTests(unittest.TestCase):

    def setUp(self):
        self.gcomm = simplecomm.create_comm()
        self.size = MPI_COMM_WORLD.Get_size()
        self.rank = MPI_COMM_WORLD.Get_rank()

    def tearDown(self):
        pass

    def testGetSize(self):
        actual = self.gcomm.get_size()
        expected = self.size
        msg = test_info_msg(self.rank, self.size, 'get_size()', None, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testIsMaster(self):
        actual = self.gcomm.is_master()
        expected = (self.rank == 0)
        msg = test_info_msg(self.rank, self.size, 'is_master()', None, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSumInt(self):
        data = 5
        actual = self.gcomm.reduce(data)
        expected = self.size * 5
        msg = test_info_msg(self.rank, self.size, 'sum(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSumList(self):
        data = range(5)
        actual = self.gcomm.reduce(data)
        expected = self.size * sum(data)
        msg = test_info_msg(self.rank, self.size, 'sum(list)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSumDict(self):
        data = {'rank': self.rank, 'range': range(3 + self.rank)}
        actual = self.gcomm.reduce(data, op=sum)
        expected = {'rank': sum(range(self.size)),
                    'range': sum([sum(range(3 + i)) for i in xrange(self.size)])}
        msg = test_info_msg(self.rank, self.size, 'sum(dict)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSumArray(self):
        data = np.arange(5)
        actual = self.gcomm.reduce(data)
        expected = self.size * sum(data)
        msg = test_info_msg(self.rank, self.size, 'sum(array)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testMaxInt(self):
        data = 13 + self.rank
        actual = self.gcomm.reduce(data, op=max)
        expected = 12 + self.size
        msg = test_info_msg(self.rank, self.size, 'max(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testMaxList(self):
        data = range(5 + self.rank)
        actual = self.gcomm.reduce(data, op=max)
        expected = (self.size - 1) + max(range(5))
        msg = test_info_msg(self.rank, self.size, 'max(list)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testMaxDict(self):
        data = {'rank': self.rank, 'range': range(3 + self.rank)}
        actual = self.gcomm.reduce(data, op=max)
        expected = {'rank': self.size - 1, 'range': self.size + 1}
        msg = test_info_msg(self.rank, self.size, 'max(dict)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testMaxArray(self):
        data = np.arange(5 + self.rank)
        actual = self.gcomm.reduce(data, op=max)
        expected = (self.size - 1) + max(range(5))
        msg = test_info_msg(self.rank, self.size, 'max(array)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testGatherInt(self):
        data = 10 + self.rank
        actual = self.gcomm.gather(data)
        if self.gcomm.is_master():
            expected = [10 + i for i in xrange(self.size)]
        else:
            expected = None
        msg = test_info_msg(self.rank, self.size, 'Gather(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testGatherList(self):
        data = range(2 + self.rank)
        actual = self.gcomm.gather(data)
        if self.gcomm.is_master():
            expected = [range(2 + i) for i in xrange(self.size)]
        else:
            expected = None
        msg = test_info_msg(self.rank, self.size, 'Gather(list)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testScatterInt(self):
        if self.gcomm.is_master():
            data = 10
            actual = self.gcomm.scatter(data)
            expected = 10
        else:
            data = None
            actual = self.gcomm.scatter()
            expected = 10
        msg = test_info_msg(self.rank, self.size, 'Scatter(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testScatterList(self):
        if self.gcomm.is_master():
            data = range(10)
            actual = self.gcomm.scatter(data, part=equal_stride)
            expected = range(10)[0::self.size]
        else:
            data = None
            actual = self.gcomm.scatter()
            expected = range(10)[self.rank::self.size]
        msg = test_info_msg(self.rank, self.size, 'Scatter(list)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testScatterListSkip(self):
        if self.gcomm.is_master():
            data = range(10)
            actual = self.gcomm.scatter(data, part=equal_stride, skip=True)
            expected = None
        else:
            data = None
            actual = self.gcomm.scatter()
            expected = range(10)[self.rank - 1::self.size - 1]
        msg = test_info_msg(self.rank, self.size, 'Scatter(list, skip)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSplitSumInt(self):
        intracomm, intercomm = self.gcomm.split([1, 1], minsize=1)
        data = 1
        data2 = intracomm.reduce(data, op=sum)
        actual = intercomm.reduce(data2, op=sum)
        if intracomm.is_master():
            expected = self.gcomm.get_size()
        else:
            expected = None
        msg = test_info_msg(self.rank, self.size, 'SplitSum(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSplitGatherInt(self):
        intracomm, intercomm = self.gcomm.split([1, 1], minsize=1)
        data = self.rank
        ranks = intracomm.gather(data)
        actual = intercomm.gather(ranks)
        if self.gcomm.is_master():
            if self.size == 1:
                expected = [[0]]
            else:
                expected = [range(0, self.size / 2 + (self.size % 2)),
                            range(self.size / 2 + (self.size % 2), self.size)]
        else:
            expected = None
        msg = test_info_msg(self.rank, self.size, 'SplitGather(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSplitScatterInt(self):
        intracomm, intercomm = self.gcomm.split([1, 1], minsize=1)
        if self.gcomm.is_master():
            data = 10
        else:
            data = None
        data2 = intercomm.scatter(data)
        actual = intracomm.scatter(data2)
        expected = 10
        msg = test_info_msg(self.rank, self.size, 'SplitScatter(int)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSplitScatterList(self):
        intracomm, intercomm = self.gcomm.split([1, 1], minsize=1)
        if self.gcomm.is_master():
            data = range(10)
        else:
            data = None
        data2 = intercomm.scatter(data, part=equal_stride)
        actual = intracomm.scatter(data2, part=equal_stride)
        if self.size == 1:
            expected = range(0, 10)
        else:
            expected = range(intracomm.get_color(), 10, 2)
            expected = expected[intracomm.comm.Get_rank()::intracomm.get_size()]
        msg = test_info_msg(self.rank, self.size, 'SplitScatter(list)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)

    def testSplitScatterListSkip(self):
        intracomm, intercomm = self.gcomm.split([1, 1], minsize=1)
        if self.gcomm.is_master():
            data = range(10)
        else:
            data = None
        data2 = intercomm.scatter(data, part=equal_stride, skip=True)
        actual = intracomm.scatter(data2, part=equal_stride, skip=True)
        if intracomm.is_master():
            expected = None
        else:
            expected = range(intracomm.get_color(), 10, 2)
            expected = expected[intracomm.comm.Get_rank() - 1::intracomm.get_size() - 1]
        msg = test_info_msg(self.rank, self.size, 'SplitScatter(list, skip)', data, actual, expected)
        print msg
        self.assertEqual(actual, expected, msg)


if __name__ == "__main__":
    hline = '=' * 70
    if MPI_COMM_WORLD.Get_rank() == 0:
        print hline
        print 'STANDARD OUTPUT FROM ALL TESTS:'
        print hline
    MPI_COMM_WORLD.Barrier()

    from cStringIO import StringIO
    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(SimpleCommParTests)
    unittest.TextTestRunner(stream=mystream).run(tests)
    MPI_COMM_WORLD.Barrier()

    results = MPI_COMM_WORLD.gather(mystream.getvalue())
    if MPI_COMM_WORLD.Get_rank() == 0:
        for rank, result in enumerate(results):
            print hline
            print 'TESTS RESULTS FOR RANK ' + str(rank) + ':'
            print hline
            print str(result)
