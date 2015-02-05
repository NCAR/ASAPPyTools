'''
These are the unit tests for the partfunc module functions
_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''
import unittest
import partfunc
from os import linesep


class partfuncTests(unittest.TestCase):
    '''
    Unit tests for the partfunc module
    '''

    def setUp(self):
        data = [range(3), range(5), range(7)]
        indices_sizes = [(0, 1), (1, 3), (5, 9)]
        self.inputs = []
        for d in data:
            for (i, s) in indices_sizes:
                self.inputs.append((d, i, s))

    def tearDown(self):
        pass

    def testOutOfBounds(self):
        self.assertRaises(IndexError, partfunc.unity, [1, 2, 3], 3, 3)
        self.assertRaises(IndexError, partfunc.unity, [1, 2, 3], 7, 3)

    def testUnity(self):
        for inp in self.inputs:
            actual = partfunc.unity(*inp)
            expected = inp[0]
            name = 'unity'
            spcr = ' ' * len(name)
            msg = ''.join([linesep,
                           name, ' - Data: ', str(inp[0]), linesep,
                           spcr, ' - Index/Size: ', str(inp[1]), '/', str(inp[2]), linesep,
                           spcr, ' - Actual:   ', str(actual), linesep,
                           spcr, ' - Expected: ', str(expected)])
            print msg
            self.assertEqual(actual, expected, msg)

    def testEqualStride(self):
        for inp in self.inputs:
            actual = partfunc.equal_stride(*inp)
            expected = inp[0][inp[1]::inp[2]]
            name = 'equal_stride'
            spcr = ' ' * len(name)
            msg = ''.join([linesep,
                           name, ' - Data: ', str(inp[0]), linesep,
                           spcr, ' - Index/Size: ', str(inp[1]), '/', str(inp[2]), linesep,
                           spcr, ' - Actual:   ', str(actual), linesep,
                           spcr, ' - Expected: ', str(expected)])
            print msg
            self.assertEqual(actual, expected, msg)

    def testSortedStrid(self):
        for inp in self.inputs:
            weights = [(20 - i) for i in inp[0]]
            actual = partfunc.sorted_stride(zip(inp[0], weights),
                                            inp[1], inp[2])
            expected = inp[0][:]
            expected.reverse()
            expected = expected[inp[1]::inp[2]]
            name = 'sorted_stride'
            spcr = ' ' * len(name)
            msg = ''.join([linesep,
                           name, ' - Data:    ', str(inp[0]), linesep,
                           spcr, ' - Weights: ', str(weights), linesep,
                           spcr, ' - Index/Size: ', str(inp[1]), '/', str(inp[2]), linesep,
                           spcr, ' - Actual:   ', str(actual), linesep,
                           spcr, ' - Expected: ', str(expected)])
            print msg
            self.assertEqual(actual, expected, msg)

    def testWeightBalanced(self):
        results = [[2, 1, 0],
                   [1],
                   [],
                   [3, 2, 4, 1, 0],
                   [2, 0],
                   [],
                   [3, 2, 4, 1, 5, 0, 6],
                   [2, 5],
                   [0]]
        for (ii, inp) in enumerate(self.inputs):
            weights = [(3 - i) ** 2 for i in inp[0]]
            actual = partfunc.sorted_stride(zip(inp[0], weights),
                                            inp[1], inp[2])
            expected = results[ii]
            name = 'weight_balanced'
            spcr = ' ' * len(name)
            msg = ''.join([linesep,
                           name, ' - Data:    ', str(inp[0]), linesep,
                           spcr, ' - Weights: ', str(weights), linesep,
                           spcr, ' - Index/Size: ', str(inp[1]), '/', str(inp[2]), linesep,
                           spcr, ' - Actual:   ', str(actual), linesep,
                           spcr, ' - Expected: ', str(expected)])
            print msg
            self.assertEqual(actual, expected, msg)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBasicInt']
    unittest.main()
