'''
Unit tests for the Messenger class (serial only)

-----------------------
Created on May 13, 2014

@author: kpaul
'''
import unittest
import messenger


class MessengerSerialTests(unittest.TestCase):
    '''
    Serial Messenger unit tests.
    '''

    def test_init(self):
        msngr = messenger.Messenger()
        self.assertIsInstance(msngr, messenger.Messenger,
                              'Failed to create class instance')
        self.assertEqual(msngr._mpi_rank, 0,
                         'Rank is wrong after initialization')
        self.assertEqual(msngr._mpi_size, 1,
                         'Size is wrong after initialization')
        self.assertEqual(msngr._is_master, (0 == 0),
                         'Is_master is wrong after initialization')
        self.assertEqual(msngr.verbosity, 1,
                         'Verbosity is wrong after initialization')

    def test_partition(self):
        msngr = messenger.Messenger()
        data = [1, 2, 3]
        p_data = msngr.partition(data)
        self.assertListEqual(data, p_data,
                         'Serial partition is wrong')

    def test_is_master(self):
        msngr = messenger.Messenger()
        self.assertTrue(msngr.is_master(),
                        'Serial messenger should be master')

    def test_get_rank(self):
        msngr = messenger.Messenger()
        self.assertEqual(msngr.get_rank(), 0,
                        'Serial messenger rank should be 0')

    def test_get_size(self):
        msngr = messenger.Messenger()
        self.assertEqual(msngr.get_size(), 1,
                        'Serial messenger size should be 1')

    def test_sum_list(self):
        msngr = messenger.Messenger()
        data = [1, 2, 3, 4]
        msngr_sum = msngr.reduce(data, op='sum')
        print msngr_sum
        self.assertEqual(msngr_sum, sum(data),
                        'Serial messenger list sum not working')

    def test_sum_dict(self):
        msngr = messenger.Messenger()
        data = {'a': 1, 'b': [2, 6], 'c': 3}
        rslt = {'a': 1, 'b': 8, 'c': 3}
        msngr_sum = msngr.reduce(data, op='sum')
        print msngr_sum
        self.assertDictEqual(msngr_sum, rslt,
                        'Serial messenger dict sum not working')

    def test_max_list(self):
        msngr = messenger.Messenger()
        data = [1, 2, 3]
        msngr_max = msngr.reduce(data, op='max')
        print msngr_max
        self.assertEqual(msngr_max, 3,
                        'Serial messenger list max not working')

    def test_max_dict(self):
        msngr = messenger.Messenger()
        data = {'a': 1, 'b': [2, 7], 'c': 3}
        rslt = {'a': 1, 'b': 7, 'c': 3}
        msngr_max = msngr.reduce(data, op='max')
        print msngr_max
        self.assertDictEqual(msngr_max, rslt,
                        'Serial messenger dict max not working')

    def test_print_once(self):
        msngr = messenger.Messenger()
        msg = 'TEST - ONCE - SERIAL'
        msngr.prinfo(msg, vlevel=0)

    def test_print_all(self):
        msngr = messenger.Messenger()
        msg = 'TEST - ALL - SERIAL'
        msngr.prinfo(msg, vlevel=0, all=True)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
