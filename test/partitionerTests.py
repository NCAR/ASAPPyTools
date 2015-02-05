'''
These are the unit tests for the partitioner module functions
_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''
import unittest
import partitioner


class partitionerTests(unittest.TestCase):
    '''
    Unit tests for the partitioner module
    '''

    def setUp(self):
        self.data = range(7)
        self.index = 1
        self.size = 3
        self.input = (self.data, self.index, self.size)

    def tearDown(self):
        pass

    def testUnity(self):
        actual = partitioner.unity(self.input)
        expected = self.input
        msg = ''.join(['unity - Actual: ', str(actual),
                       ' - Expected: ', str(expected)])
        print msg
        self.assertEqual(actual, expected, msg)

    def testEqualLength(self):
        actual = partitioner.equal_length(*self.input)
        expected = self.data[self.index::self.size]
        msg = ''.join(['equal_length - Actual: ', str(actual),
                       ' - Expected: ', str(expected)])
        print msg
        self.assertEqual(actual, expected, msg)

    def testWeightedEqualLength(self):
        weights = [20 - i for i in self.data]
        actual = partitioner.weighted_equal_length([self.data, weights],
                                                   self.index, self.size)
        expected = self.data[:]
        expected.reverse()
        expected = expected[self.index::self.size]
        msg = ''.join(['weighted_equal_length - Actual: ', str(actual),
                       ' - Expected: ', str(expected)])
        print msg
        self.assertEqual(actual, expected, msg)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testBasicInt']
    unittest.main()
