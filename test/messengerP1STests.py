'''
Unit tests for the Messenger class (serial vs 1-rank parallel comparison)

These tests should be run through mpirun/mpiexec in order to test them
in true parallel.  Specifically, these tests compare the output of a
truly serial run to the output of a 1-rank parallel run.  Testing this
suite should be done with the command:

    mpirun -n 1 python messengerParallelTests.py

-----------------------
Created on Jan 7, 2015

@author: kpaul
'''
import unittest
import messenger
import numpy as np
from mpi4py import MPI


class MessengerParallelTests(unittest.TestCase):
    '''
    Parallel Messenger unit tests.
    '''
    def test_get_rank(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        self.assertEqual(smsngr.get_rank(), pmsngr.get_rank(),
                        'Parallel rank not same as serial rank')

    def test_get_size(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        self.assertEqual(smsngr.get_size(), pmsngr.get_size(),
                        'Parallel size not same as serial size')

    def test_is_master(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        self.assertEqual(smsngr.is_master(), pmsngr.is_master(),
                         'Parallel master not the same as serial master')

    def test_partition_unweighted(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = range(20)
        self.assertListEqual(smsngr.partition(data), pmsngr.partition(data),
                         'Parallel unweighted partition not the same as serial')

    def test_partition_weighted(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = {i:(10 - i) for i in range(20)}
        spart = smsngr.partition(data)
        ppart = pmsngr.partition(data)
        self.assertListEqual(spart, ppart,
                         'Parallel weighted partition not the same as serial')

    def test_sum_list(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = [1, 2, 3, 4]
        self.assertEqual(smsngr.reduce(data, op='sum'), pmsngr.reduce(data, op='sum'),
                        'Parallel list sum not the same as serial')

    def test_sum_dict(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = {'a': 1, 'b': [2, 6], 'c': 3}
        self.assertDictEqual(smsngr.reduce(data, op='sum'), pmsngr.reduce(data, op='sum'),
                        'Parallel dict sum not the same as serial')

    def test_sum_ndarray(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = np.array([1, 2, 3, 4], dtype=np.int)
        self.assertEqual(smsngr.reduce(data, op='sum'), pmsngr.reduce(data, op='sum'),
                        'Parallel list NDArray sum not the same as serial')

    def test_max_list(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = range(10)
        self.assertEqual(smsngr.reduce(data, op='max'), pmsngr.reduce(data, op='max'),
                        'Parallel list max not the same as serial')

    def test_max_dict(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = {'a': 1, 'b': [2, 7], 'c': 3}
        self.assertDictEqual(smsngr.reduce(data, op='max'), pmsngr.reduce(data, op='max'),
                        'Parallel dict max not the same as serial')

    def test_max_ndarray(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = np.array([1, 2, 3, 4], dtype=np.int)
        self.assertEqual(smsngr.reduce(data, op='max'), pmsngr.reduce(data, op='max'),
                        'Parallel list NDArray max not the same as serial')

    def test_split(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        subsmsngr = smsngr.split(2)
        subpmsngr = pmsngr.split(2)
        self.assertEqual(subsmsngr.get_rank(), subpmsngr.get_rank(),
                         'Parallel split gives different rank from serial')

    def test_gather_int(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        i = 5
        self.assertListEqual(smsngr.gather(i), pmsngr.gather(i),
                         'Parallel gather int not the same as serial')

    def test_gather_ndarray(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = np.array([5, 6, 7], dtype=np.int)
        np.testing.assert_array_equal(smsngr.gather(data), pmsngr.gather(data),
                         'Parallel gather NDarray not the same as serial')

    def test_scatter_int(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        d = [5]
        self.assertEqual(smsngr.scatter(d), pmsngr.scatter(d),
                         'Parallel scatter int not the same as serial')

    def test_scatter_ndarray(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = np.array([5], dtype=np.int)
        np.testing.assert_array_equal(smsngr.scatter(data), pmsngr.scatter(data),
                         'Parallel scatter NDArray not the same as serial')

    def test_bcast_int(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = 10
        self.assertEqual(smsngr.broadcast(data), pmsngr.broadcast(data),
                         'Parallel broadcast int not the same as serial')

    def test_bcast_ndarray(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = np.array([10, 11, 12, 13], dtype=np.float)
        np.testing.assert_array_equal(smsngr.broadcast(data), pmsngr.broadcast(data),
                         'Parallel broadcast NDArray not the same as serial')

    def test_sendrecv_int(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = 5
        self.assertEqual(smsngr.sendrecv(data), pmsngr.sendrecv(data),
                         'Parallel sendrecv int not the same as serial')

    def test_sendrecv_ndarray(self):
        smsngr = messenger.Messenger()
        pmsngr = messenger.MPIMessenger()
        data = np.array([10, 11, 12, 13], dtype=np.float)
        np.testing.assert_array_equal(smsngr.sendrecv(data), pmsngr.sendrecv(data),
                         'Parallel sendrecv NDArray not the same as serial')



if __name__ == "__main__":
    if MPI.COMM_WORLD.Get_size() > 1:
        raise RuntimeError('Must run these tests in parallel with 1 rank')

    hline = '=' * 70
    print hline
    print 'STANDARD OUTPUT FROM ALL TESTS:'
    print hline

    from cStringIO import StringIO
    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(MessengerParallelTests)
    unittest.TextTestRunner(stream=mystream).run(tests)

    results = mystream.getvalue()
    print hline
    print 'TESTS RESULTS:'
    print hline
    print str(results)
