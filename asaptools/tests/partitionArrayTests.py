"""
These are the unit tests for the partition module functions

Copyright 2017, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from __future__ import absolute_import, division, print_function

import unittest
from os import linesep

from numpy import arange, array, dstack, testing

from asaptools import partition


def test_info_msg(name, data, index, size, actual, expected):
    spcr = ' ' * len(name)
    msg = ''.join(
        [
            linesep,
            name,
            ' - Data: ',
            str(data),
            linesep,
            spcr,
            ' - Index/Size: ',
            str(index),
            '/',
            str(size),
            linesep,
            spcr,
            ' - Actual:   ',
            str(actual),
            linesep,
            spcr,
            ' - Expected: ',
            str(expected),
        ]
    )
    return msg


class partitionArrayTests(unittest.TestCase):

    """
    Unit tests for the partition module
    """

    def setUp(self):
        data = [arange(3), arange(5), arange(7)]
        indices_sizes = [(0, 1), (1, 3), (5, 9)]
        self.inputs = []
        for d in data:
            for (i, s) in indices_sizes:
                self.inputs.append((d, i, s))

    def testOutOfBounds(self):
        self.assertRaises(IndexError, partition.EqualLength(), [1, 2, 3], 3, 3)
        self.assertRaises(IndexError, partition.EqualStride(), [1, 2, 3], 7, 3)

    def testDuplicate(self):
        for inp in self.inputs:
            pfunc = partition.Duplicate()
            actual = pfunc(*inp)
            expected = inp[0]
            testing.assert_array_equal(actual, expected)

    def testEquallength(self):
        results = [
            arange(3),
            array([1]),
            array([]),
            arange(5),
            array([2, 3]),
            array([]),
            arange(7),
            array([3, 4]),
            array([5]),
        ]
        for (ii, inp) in enumerate(self.inputs):
            pfunc = partition.EqualLength()
            actual = pfunc(*inp)
            expected = results[ii]
            testing.assert_array_equal(actual, expected)

    def testEqualStride(self):
        for inp in self.inputs:
            pfunc = partition.EqualStride()
            actual = pfunc(*inp)
            expected = inp[0][inp[1] :: inp[2]]
            testing.assert_array_equal(actual, expected)

    def testSortedStride(self):
        for inp in self.inputs:
            weights = array([(20 - i) for i in inp[0]])
            pfunc = partition.SortedStride()
            data = dstack((inp[0], weights))[0]
            actual = pfunc(data, inp[1], inp[2])
            expected = inp[0][::-1]
            expected = expected[inp[1] :: inp[2]]
            testing.assert_array_equal(actual, expected)

    def testWeightBalanced(self):
        results = [
            set([0, 1, 2]),
            set([1]),
            set(),
            set([3, 2, 4, 1, 0]),
            set([1]),
            set(),
            set([3, 2, 4, 1, 5, 0, 6]),
            set([3, 6]),
            set([4]),
        ]
        for (ii, inp) in enumerate(self.inputs):
            weights = array([(3 - i) ** 2 for i in inp[0]])
            pfunc = partition.WeightBalanced()
            data = dstack((inp[0], weights))[0]
            actual = set(pfunc(data, inp[1], inp[2]))
            expected = results[ii]
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
