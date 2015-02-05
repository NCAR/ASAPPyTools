'''
Parallel Tests for the SimpleComm class

_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''
import unittest
import simplecomm
import numpy as np
from mpi4py import MPI


class SimpleCommParTests(unittest.TestCase):

    def setUp(self):
        self.gcomm = simplecomm.SimpleComm()
        self.size = MPI.COMM_WORLD.Get_size()
        self.rank = MPI.COMM_WORLD.Get_rank()

    def tearDown(self):
        pass

    def testGetSize(self):
        actual = self.gcomm.get_size()
        expected = self.size
        err_msg = 'Communicator size (' + str(actual) + \
                  ') is not what was expected (' + str(expected) + ')'
        self.assertEqual(actual, expected, err_msg)

    def testIsMaster(self):
        actual = self.gcomm.is_master()
        expected = (self.rank == 0)
        print ''.join(['RANK: ', str(self.rank), ' - is_master() = ',
                       str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['is_master() - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testSumInt(self):
        data = 5
        actual = self.gcomm.reduce(data)
        expected = self.size * 5
        print ''.join(['RANK: ', str(self.rank), ' - sum(int) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['sum(int) - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testSumList(self):
        data = range(5)
        actual = self.gcomm.reduce(data)
        expected = self.size * sum(data)
        print ''.join(['RANK: ', str(self.rank), ' - sum(list) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['sum(list) - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testSumArray(self):
        data = np.arange(5)
        actual = self.gcomm.reduce(data)
        expected = self.size * sum(data)
        print ''.join(['RANK: ', str(self.rank), ' - sum(array) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['sum(array) - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testMaxInt(self):
        data = 13 + self.rank
        actual = self.gcomm.reduce(data, op=max)
        expected = 12 + self.size
        print ''.join(['RANK: ', str(self.rank), ' - max(int) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['max(int) - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testMaxList(self):
        data = range(5 + self.rank)
        actual = self.gcomm.reduce(data, op=max)
        expected = (self.size - 1) + max(range(5))
        print ''.join(['RANK: ', str(self.rank), ' - max(list) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['max(list) - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testMaxArray(self):
        data = np.arange(5 + self.rank)
        actual = self.gcomm.reduce(data, op=max)
        expected = (self.size - 1) + max(range(5))
        print ''.join(['RANK: ', str(self.rank), ' - max(array) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['max(array) - Expected: ', str(expected),
                                  ', Actual: ', str(actual)]))

    def testTwoWayInt(self):
        data = 10 + self.rank
        self.gcomm.send(data)
        actual = self.gcomm.receive()
        if self.gcomm.is_master():
            expected = [10 + i for i in xrange(self.size)]
        else:
            expected = 10
        print ''.join(['RANK: ', str(self.rank), ' - TwoWay(int) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['TwoWay(int) - Expected: ',
                                  str(expected), ', Actual: ', str(actual)]))

    def testTwoWayList(self):
        data = range(1 + self.rank)
        self.gcomm.send(data)
        actual = self.gcomm.receive()
        if self.gcomm.is_master():
            expected = [range(1 + i) for i in xrange(self.size)]
        else:
            expected = [0]
        print ''.join(['RANK: ', str(self.rank), ' - TwoWay(list) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['TwoWay(list) - Expected: ',
                                  str(expected), ', Actual: ', str(actual)]))

    def testSendOutInt(self):
        if self.gcomm.is_master():
            data = 100
            self.gcomm.send(data)
            actual = None
            expected = None
        else:
            data = None
            actual = self.gcomm.receive()
            expected = 100
        print ''.join(['RANK: ', str(self.rank), ' - SendOut(int) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['SendOut(int) - Expected: ',
                                  str(expected), ', Actual: ', str(actual)]))

    def testSendInInt(self):
        if not self.gcomm.is_master():
            data = 100 + self.rank
            self.gcomm.send(data)
            actual = None
            expected = None
        else:
            data = None
            actual = self.gcomm.receive()
            expected = [100 + i for i in xrange(self.size)]
            expected[0] = None
        print ''.join(['RANK: ', str(self.rank), ' - SendIn(int) - '
                       'Input: ', str(data), ' - Result: ', str(actual)])
        self.assertEqual(actual, expected,
                         ''.join(['SendIn(int) - Expected: ',
                                  str(expected), ', Actual: ', str(actual)]))


if __name__ == "__main__":
    hline = '=' * 70
    if MPI.COMM_WORLD.Get_rank() == 0:
        print hline
        print 'STANDARD OUTPUT FROM ALL TESTS:'
        print hline
    MPI.COMM_WORLD.Barrier()

    from cStringIO import StringIO
    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(SimpleCommParTests)
    unittest.TextTestRunner(stream=mystream).run(tests)
    MPI.COMM_WORLD.Barrier()

    results = MPI.COMM_WORLD.gather(mystream.getvalue())
    if MPI.COMM_WORLD.Get_rank() == 0:
        for rank, result in enumerate(results):
            print hline
            print 'TESTS RESULTS FOR RANK ' + str(rank) + ':'
            print hline
            print str(result)
