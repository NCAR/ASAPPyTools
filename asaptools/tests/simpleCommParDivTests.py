"""
Parallel Tests with communicator division for the SimpleComm class

Copyright 2017, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from __future__ import print_function

import unittest

from mpi4py import MPI

from asaptools import simplecomm
from asaptools.partition import Duplicate, EqualStride

MPI_COMM_WORLD = MPI.COMM_WORLD


class SimpleCommParDivTests(unittest.TestCase):
    def setUp(self):
        self.gcomm = simplecomm.create_comm()
        self.gsize = MPI_COMM_WORLD.Get_size()
        self.grank = MPI_COMM_WORLD.Get_rank()

        self.groups = ['a', 'b', 'c']

        self.rank = int(self.grank // len(self.groups))
        self.color = int(self.grank % len(self.groups))
        self.group = self.groups[self.color]

        self.monocomm, self.multicomm = self.gcomm.divide(self.group)

        self.all_colors = [i % len(self.groups) for i in range(self.gsize)]
        self.all_groups = [self.groups[i] for i in self.all_colors]
        self.all_ranks = [int(i // len(self.groups)) for i in range(self.gsize)]

    def testGlobalRanksMatch(self):
        actual = self.gcomm.get_rank()
        expected = self.grank
        self.assertEqual(actual, expected)

    def testMonoGetRank(self):
        actual = self.monocomm.get_rank()
        expected = self.rank
        self.assertEqual(actual, expected)

    def testMultiGetRank(self):
        actual = self.multicomm.get_rank()
        expected = self.color
        self.assertEqual(actual, expected)

    def testMonoGetSize(self):
        actual = self.monocomm.get_size()
        expected = self.all_colors.count(self.color)
        self.assertEqual(actual, expected)

    def testMultiGetSize(self):
        actual = self.multicomm.get_size()
        expected = self.all_ranks.count(self.rank)
        self.assertEqual(actual, expected)

    def testMonoIsManager(self):
        actual = self.monocomm.is_manager()
        expected = self.rank == 0
        self.assertEqual(actual, expected)

    def testMultiIsManager(self):
        actual = self.multicomm.is_manager()
        expected = self.color == 0
        self.assertEqual(actual, expected)

    def testMonoSumInt(self):
        data = self.color + 1
        actual = self.monocomm.allreduce(data, 'sum')
        expected = self.monocomm.get_size() * data
        self.assertEqual(actual, expected)

    def testMultiSumInt(self):
        data = self.rank + 1
        actual = self.multicomm.allreduce(data, 'sum')
        expected = self.multicomm.get_size() * data
        self.assertEqual(actual, expected)

    def testMonoSumList(self):
        data = list(range(5))
        actual = self.monocomm.allreduce(data, 'sum')
        expected = self.monocomm.get_size() * sum(data)
        self.assertEqual(actual, expected)

    def testMultiSumList(self):
        data = list(range(5))
        actual = self.multicomm.allreduce(data, 'sum')
        expected = self.multicomm.get_size() * sum(data)
        self.assertEqual(actual, expected)

    def testMonoSumDict(self):
        data = {'a': list(range(3)), 'b': [5, 7]}
        actual = self.monocomm.allreduce(data, 'sum')
        expected = {
            'a': self.monocomm.get_size() * sum(range(3)),
            'b': self.monocomm.get_size() * sum([5, 7]),
        }
        self.assertEqual(actual, expected)

    def testMultiSumDict(self):
        data = {'a': list(range(3)), 'b': [5, 7]}
        actual = self.multicomm.allreduce(data, 'sum')
        expected = {
            'a': self.multicomm.get_size() * sum(range(3)),
            'b': self.multicomm.get_size() * sum([5, 7]),
        }
        self.assertEqual(actual, expected)

    def testMonoPartitionInt(self):
        data = self.grank
        actual = self.monocomm.partition(data, func=Duplicate())
        if self.monocomm.is_manager():
            expected = None
        else:
            expected = self.color  # By chance!
        self.assertEqual(actual, expected)

    def testMultiPartitionInt(self):
        data = self.grank
        actual = self.multicomm.partition(data, func=Duplicate())
        if self.multicomm.is_manager():
            expected = None
        else:
            expected = self.rank * len(self.groups)
        self.assertEqual(actual, expected)

    def testMonoPartitionIntInvolved(self):
        data = self.grank
        actual = self.monocomm.partition(data, func=Duplicate(), involved=True)
        expected = self.color  # By chance!
        self.assertEqual(actual, expected)

    def testMultiPartitionIntInvolved(self):
        data = self.grank
        actual = self.multicomm.partition(data, func=Duplicate(), involved=True)
        expected = self.rank * len(self.groups)
        self.assertEqual(actual, expected)

    def testMonoPartitionList(self):
        if self.monocomm.is_manager():
            data = list(range(10 + self.grank))
        else:
            data = None
        actual = self.monocomm.partition(data)
        if self.monocomm.is_manager():
            expected = None
        else:
            expected = list(range(self.rank - 1, 10 + self.color, self.monocomm.get_size() - 1))
        self.assertEqual(actual, expected)

    def testMultiPartitionList(self):
        if self.multicomm.is_manager():
            data = list(range(10 + self.grank))
        else:
            data = None
        actual = self.multicomm.partition(data)
        if self.multicomm.is_manager():
            expected = None
        else:
            expected = list(
                range(
                    self.color - 1,
                    10 + self.rank * len(self.groups),
                    self.multicomm.get_size() - 1,
                )
            )
        self.assertEqual(actual, expected)

    def testMonoPartitionListInvolved(self):
        if self.monocomm.is_manager():
            data = list(range(10 + self.grank))
        else:
            data = None
        actual = self.monocomm.partition(data, func=EqualStride(), involved=True)
        expected = list(range(self.rank, 10 + self.color, self.monocomm.get_size()))
        self.assertEqual(actual, expected)

    def testMultiPartitionListInvolved(self):
        if self.multicomm.is_manager():
            data = list(range(10 + self.grank))
        else:
            data = None
        actual = self.multicomm.partition(data, func=EqualStride(), involved=True)
        expected = list(
            range(self.color, 10 + self.rank * len(self.groups), self.multicomm.get_size())
        )
        self.assertEqual(actual, expected)

    def testMonoCollectInt(self):
        if self.monocomm.is_manager():
            data = None
            actual = [self.monocomm.collect() for _ in range(1, self.monocomm.get_size())]
            expected = [
                i
                for i in enumerate(
                    range(len(self.groups) + self.color, self.gsize, len(self.groups)),
                    1,
                )
            ]
        else:
            data = self.grank
            actual = self.monocomm.collect(data)
            expected = None
        self.monocomm.sync()
        if self.monocomm.is_manager():
            for a in actual:
                self.assertTrue(a in expected)
        else:
            self.assertEqual(actual, expected)

    def testMultiCollectInt(self):
        if self.multicomm.is_manager():
            data = None
            actual = [self.multicomm.collect() for _ in range(1, self.multicomm.get_size())]
            expected = [
                i
                for i in enumerate(
                    [j + self.rank * len(self.groups) for j in range(1, self.multicomm.get_size())],
                    1,
                )
            ]
        else:
            data = self.grank
            actual = self.multicomm.collect(data)
            expected = None
        self.multicomm.sync()
        if self.multicomm.is_manager():
            for a in actual:
                self.assertTrue(a in expected)
        else:
            self.assertEqual(actual, expected)

    def testMonoCollectList(self):
        if self.monocomm.is_manager():
            data = None
            actual = [self.monocomm.collect() for _ in range(1, self.monocomm.get_size())]
            expected = [
                (i, list(range(x)))
                for i, x in enumerate(
                    range(len(self.groups) + self.color, self.gsize, len(self.groups)),
                    1,
                )
            ]
        else:
            data = list(range(self.grank))
            actual = self.monocomm.collect(data)
            expected = None
        self.monocomm.sync()
        if self.monocomm.is_manager():
            for a in actual:
                self.assertTrue(a in expected)
        else:
            self.assertEqual(actual, expected)

    def testMultiCollectList(self):
        if self.multicomm.is_manager():
            data = None
            actual = [self.multicomm.collect() for _ in range(1, self.multicomm.get_size())]
            expected = [
                (i, list(range(x)))
                for (i, x) in enumerate(
                    [j + self.rank * len(self.groups) for j in range(1, self.multicomm.get_size())],
                    1,
                )
            ]
        else:
            data = list(range(self.grank))
            actual = self.multicomm.collect(data)
            expected = None
        self.multicomm.sync()
        if self.multicomm.is_manager():
            for a in actual:
                self.assertTrue(a in expected)
        else:
            self.assertEqual(actual, expected)

    def testMonoRationInt(self):
        if self.monocomm.is_manager():
            data = [100 * self.color + i for i in range(1, self.monocomm.get_size())]
            actual = [self.monocomm.ration(d) for d in data]
            expected = [None] * (self.monocomm.get_size() - 1)
        else:
            data = None
            actual = self.monocomm.ration()
            expected = [100 * self.color + i for i in range(1, self.monocomm.get_size())]
        self.monocomm.sync()
        if self.monocomm.is_manager():
            self.assertEqual(actual, expected)
        else:
            self.assertTrue(actual in expected)

    def testMultiRationInt(self):
        if self.multicomm.is_manager():
            data = [100 * self.rank + i for i in range(1, self.multicomm.get_size())]
            actual = [self.multicomm.ration(d) for d in data]
            expected = [None] * (self.multicomm.get_size() - 1)
        else:
            data = None
            actual = self.multicomm.ration()
            expected = [100 * self.rank + i for i in range(1, self.multicomm.get_size())]
        self.multicomm.sync()
        if self.multicomm.is_manager():
            self.assertEqual(actual, expected)
        else:
            self.assertTrue(actual in expected)

    def testTreeScatterInt(self):
        if self.gcomm.is_manager():
            data = 10
        else:
            data = None

        if self.monocomm.is_manager():
            mydata = self.multicomm.partition(data, func=Duplicate(), involved=True)
        else:
            mydata = None

        actual = self.monocomm.partition(mydata, func=Duplicate(), involved=True)
        expected = 10
        self.assertEqual(actual, expected)

    def testTreeGatherInt(self):
        data = self.grank

        if self.monocomm.is_manager():
            mydata = [data]
            for _ in range(1, self.monocomm.get_size()):
                mydata.append(self.monocomm.collect()[1])
        else:
            mydata = self.monocomm.collect(data)

        if self.gcomm.is_manager():
            actual = [mydata]
            for _ in range(1, self.multicomm.get_size()):
                actual.append(self.multicomm.collect()[1])
        elif self.monocomm.is_manager():
            actual = self.multicomm.collect(mydata)
        else:
            actual = None

        # expected = 10
        # self.assertEqual(actual, expected)


if __name__ == '__main__':
    try:
        from cStringIO import StringIO
    except ImportError:
        from io import StringIO

    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(SimpleCommParDivTests)
    unittest.TextTestRunner(stream=mystream).run(tests)
    MPI_COMM_WORLD.Barrier()

    results = MPI_COMM_WORLD.gather(mystream.getvalue())
    if MPI_COMM_WORLD.Get_rank() == 0:
        for rank, result in enumerate(results):
            print('RESULTS FOR RANK ' + str(rank) + ':')
            print(str(result))
