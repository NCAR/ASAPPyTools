'''
Unit tests for the Messenger class (parallel only)

These tests should be run through mpirun/mpiexec in order to test them
in true parallel.  If this is run like a normal unit test suite, then it
will only test the 1-rank MPI case.  In other words, testing this suite with
the command:

    python messengerParallelTests.py

will give the same results as:

    mpirun -n 1 python messengerParallelTests.py

To get true test coverage, one should run this test suite with the command:

    mpirun -n N python messengerParallelTests.py

With various values for N.

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
    def test_init(self):
        msngr = messenger.MPIMessenger()
        self.assertIsInstance(msngr, messenger.Messenger,
                              'Failed to create class instance')
        self.assertEqual(msngr._mpi_rank, MPI.COMM_WORLD.Get_rank(),
                         'Rank is wrong after initialization')
        self.assertEqual(msngr._mpi_size, MPI.COMM_WORLD.Get_size(),
                         'Size is wrong after initialization')
        self.assertEqual(msngr._is_master, (0 == MPI.COMM_WORLD.Get_rank()),
                         'Is_master is wrong after initialization')
        self.assertEqual(msngr._mpi_comm, MPI.COMM_WORLD,
                         'MPI Comm is wrong after initialization')
        self.assertEqual(msngr.verbosity, 1,
                         'Verbosity is wrong after initialization')

    def test_get_rank(self):
        msngr = messenger.MPIMessenger()
        self.assertEqual(msngr.get_rank(), MPI.COMM_WORLD.Get_rank(),
                        'Parallel messenger rank wrong')

    def test_get_size(self):
        msngr = messenger.MPIMessenger()
        self.assertEqual(msngr.get_size(), MPI.COMM_WORLD.Get_size(),
                        'Parallel messenger size wrong')

    def test_is_master(self):
        msngr = messenger.MPIMessenger()
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.assertTrue(msngr.is_master(),
                            'Parallel messenger should be master')
        else:
            self.assertFalse(msngr.is_master(),
                            'Parallel messenger should not be master')

    def test_partition_unweighted(self):
        msngr = messenger.MPIMessenger()
        data = [i for i in range(msngr.get_size())]
        p_data = msngr.partition(data)
        self.assertListEqual(p_data, [msngr.get_rank()],
                         'Parallel partition is wrong')

    def test_partition_weighted(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        data = {i:(size - i) for i in range(size)}
        p_data = msngr.partition(data)
        self.assertListEqual(p_data, [size - rank - 1],
                         'Parallel partition is wrong')

    def test_sum_list(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        lst = [1, 2, 3, 4]
        data = map(lambda x: x + rank, lst)
        rslt = sum(lst) * size + len(lst) * sum(range(size))
        msngr_sum = msngr.reduce(data, op='sum')
        self.assertEqual(msngr_sum, rslt,
                        'Parallel messenger list sum not working')

    def test_sum_dict(self):
        msngr = messenger.MPIMessenger()
        data = {'a': 1, 'b': [2, 6], 'c': 3}
        size = msngr.get_size()
        rslt = {'a': 1 * size, 'b': 8 * size, 'c': 3 * size}
        msngr_sum = msngr.reduce(data, op='sum')
        self.assertDictEqual(msngr_sum, rslt,
                        'Parallel messenger dict sum not working')

    def test_sum_ndarray(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        lst = [1, 2, 3, 4]
        data = np.array(map(lambda x: x + rank, lst))
        rslt = sum(lst) * size + len(lst) * sum(range(size))
        msngr_sum = msngr.reduce(data, op='sum')
        self.assertEqual(msngr_sum, rslt,
                        'Parallel messenger NDArray sum not working')

    def test_max_list(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        data = [rank + i for i in range(4)]
        rslt = (msngr.get_size() - 1) + 3
        msngr_max = msngr.reduce(data, op='max')
        self.assertEqual(msngr_max, rslt,
                        'Parallel messenger list max not working')

    def test_max_dict(self):
        msngr = messenger.MPIMessenger()
        data = {'a': 1, 'b': [2, 7], 'c': 3}
        rslt = {'a': 1, 'b': 7, 'c': 3}
        msngr_max = msngr.reduce(data, op='max')
        self.assertDictEqual(msngr_max, rslt,
                        'Parallel messenger dict max not working')

    def test_max_ndarray(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        data = np.array([rank + i for i in range(4)])
        rslt = (size - 1) + 3
        msngr_max = msngr.reduce(data, op='max')
        self.assertEqual(msngr_max, rslt,
                        'Parallel messenger NDarray max not working')

    def test_split(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        color = rank % 2
        submsngr = msngr.split(color)
        subrank = submsngr.get_rank()
        msg = 'SPLIT - Rank:' + str(rank) + '  SubRank:' + \
            str(subrank) + '  Color:' + str(color)
        msngr.prinfo(msg, vlevel=0, master=False)
        self.assertEqual(subrank, rank / 2,
                         'Unexpected subrank result from split')

    def test_gather_int(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        allranks = msngr.gather(rank)
        rslt = range(msngr.get_size())
        if msngr.is_master():
            self.assertEqual(allranks, rslt,
                             'Unexpected integer gather result on master')
        else:
            self.assertIsNone(allranks,
                              'Unexpected integer gather result on slave')

    def test_gather_ndarray(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        data = np.array([rank])
        alldata = msngr.gather(data)
        if msngr.is_master():
            rslt = np.array(range(size))
            rslt.shape = (size, 1)
            np.testing.assert_array_equal(alldata, rslt,
                              'Unexpected NDArray gather result on master')
        else:
            self.assertTupleEqual(alldata.shape, (size, 1),
                              'Unexpected NDArray gather result on subordinate')

    def test_scatter_int(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        if msngr.is_master():
            data = [i ** 2 for i in range(size)]
        else:
            data = None
        subdata = msngr.scatter(data)
        self.assertEqual(subdata, rank ** 2,
                         'Unexpected integer scatter result')

    def test_scatter_ndarray(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        if msngr.is_master():
            data = np.array([i ** 2 for i in range(size)], dtype=np.float)
            data.shape = (size, 1)
        else:
            data = np.empty((size, 1), dtype=np.float)
        subdata = msngr.scatter(data)
        rslt = np.array([rank ** 2], dtype=np.float)
        np.testing.assert_array_equal(subdata, rslt,
                         'Unexpected NDArray scatter result')

    def test_bcast_int(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        data = size + rank ** 2
        bcdata = msngr.broadcast(data)
        self.assertEqual(bcdata, size,
                         'Unexpected integer broadcast result')

    def test_bcast_ndarray(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        data = np.array([size + rank ** 2], dtype=np.int)
        bcdata = msngr.broadcast(data)
        np.testing.assert_array_equal(bcdata, np.array([size], dtype=np.int),
                         'Unexpected NDArray broadcast result')

    def test_sendrecv_int(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        data = None
        dest = 0
        source = size - 1
        if rank == source:
            data = 5
        recvd = msngr.sendrecv(data, source=source, dest=dest)
        msg = 'SENDRECV (' + str(source) + '-->' \
            + str(dest) + ') - data=' + str(data) + ' - recvd:' + str(recvd)
        msngr.prinfo(msg, vlevel=0, master=False)
        if rank == dest:
            self.assertEqual(recvd, 5,
                         'Unexpected integer sendrecv result on destination')
        else:
            self.assertEqual(recvd, None,
                         'Unexpected integer sendrecv result on other')

    def test_sendrecv_int_self(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        dest = 0
        source = 0
        if rank == source:
            data = 5
        else:
            data = None
        rcvd = msngr.sendrecv(data, source=source, dest=dest)
        if rank == dest:
            self.assertEqual(rcvd, 5,
                         'Unexpected integer self sendrecv result on source/dest')
        else:
            self.assertEqual(rcvd, None,
                         'Unexpected integer self sendrecv result on other')

    def test_sendrecv_ndarray(self):
        msngr = messenger.MPIMessenger()
        rank = msngr.get_rank()
        size = msngr.get_size()
        dest = 0
        source = size - 1
        rslt = np.array([[5, 9, 11], [13, 21, 37]], dtype=np.int)
        if rank == source:
            data = rslt
        elif rank == dest:
            data = np.empty((6,), dtype=np.int)  # size same, shape not!
        else:
            data = None
        recvd = msngr.sendrecv(data, source=source, dest=dest)
        if rank == dest:
            np.testing.assert_array_equal(recvd, rslt,
                         'Unexpected NDArray sendrecv result on destination')
        else:
            self.assertEqual(recvd, None,
                         'Unexpected NDArray sendrecv result on other')

    def test_print_once(self):
        msngr = messenger.MPIMessenger()
        msg = 'TEST - ONCE - Parallel'
        msngr.prinfo(msg, vlevel=0, master=True)

    def test_print_all(self):
        msngr = messenger.MPIMessenger()
        msg = 'TEST - ALL - Parallel'
        msngr.prinfo(msg, vlevel=0, master=False)



if __name__ == "__main__":
    hline = '=' * 70
    if MPI.COMM_WORLD.Get_rank() == 0:
        print hline
        print 'STANDARD OUTPUT FROM ALL TESTS:'
        print hline

    from cStringIO import StringIO
    mystream = StringIO()
    tests = unittest.TestLoader().loadTestsFromTestCase(MessengerParallelTests)
    unittest.TextTestRunner(stream=mystream).run(tests)

    results = MPI.COMM_WORLD.gather(mystream.getvalue())
    if MPI.COMM_WORLD.Get_rank() == 0:
        for rank, result in enumerate(results):
            print hline
            print 'TESTS RESULTS FOR RANK ' + str(rank) + ':'
            print hline
            print str(result)
