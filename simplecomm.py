'''
SimpleComm - Simple MPI Communication

The SimpleComm class is designed to provide a simplified interface to the
MPI4Py module.

_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''


class SimpleComm(object):
    '''
    Simple Communicator
    '''

    def __init__(self):
        '''
        Constructor
        '''

        # Try importing the MPI4Py MPI module
        try:
            from mpi4py import MPI
        except:
            err_msg = 'MPI could not be found.'
            raise ImportError(err_msg)

        ## Hold on to the MPI module
        self.mpi = MPI

        ## Hold on to the MPI communicator (by default, COMM_WORLD)
        self.comm = self.mpi.COMM_WORLD

        ## Local buffer for self-communication
        self.__buffer = None

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator (including
        the master rank).

        @return  The integer number of ranks in this communicator.
                 (Same on all ranks in this communicator.)
        '''
        return self.comm.Get_size()

    def is_master(self):
        '''
        Simple check to determine if this MPI task is on the 'master' rank
        (rank 0).

        @return  True if this MPI task is on the master rank, False otherwise.
        '''
        return (self.comm.Get_rank() == 0)

    def reduce(self, data, op=sum):
        '''
        Reduction: Applies a function/operator to a collection of
        data values and reduces the data to a single value.  The 'sum'
        function is an example of a reduction operator, as is the 'max'
        function.  (Note: This implementation requires that the operator
        be commutative and associative.)

        @param  data  The data to be reduced

        @param  op  A reduction operator/function

        @return  The single value constituting the reduction of the input data.
                 (Same on all ranks in this communicator.)
        '''
        if hasattr(data, '__len__'):
            return op(self.comm.allgather(op(data)))
        else:
            return op(self.comm.allgather(data))

    def send(self, data, part=None):
        '''
        Send data to 'neighbor' ranks.  If this MPI task is on the 'master'
        rank, then this sends the data to all of the 'slave' ranks.  If this
        MPI task is on a 'slave' rank, then it sends the data to the 'master'
        rank.

        @param  data  The data to be sent

        @param  part  A three-argument partitioning function
                      (Only partitions when sending from the 'master')
        '''
        if self.is_master():
            if part:
                self.__buffer = part(data, 0, self.get_size())
                reqs = [self.comm.isend(part(data, i, self.get_size()), dest=i)
                        for i in xrange(1, self.get_size())]
                self.mpi.Request.Waitall(reqs)
            else:
                self.__buffer = data
                reqs = [self.comm.isend(data, dest=i)
                        for i in xrange(1, self.get_size())]
                self.mpi.Request.Waitall(reqs)
        else:
            req = self.comm.isend(data, dest=0)
            req.Wait()

    def receive(self):
        '''
        Receive data from 'neighbor' ranks.  If this MPI task is on the
        'master' rank, then this expects to receive from all of the 'slave'
        ranks.  If this MPI task is on a 'slave' rank, then this expects to
        receive from the 'master' rank.
        '''
        if self.is_master():
            data = [None] * self.get_size()
            data[0] = self.__buffer
            for i in xrange(1, self.get_size()):
                data[i] = self.comm.recv(source=i)
            self.__buffer = None
            return data
        else:
            return self.comm.recv(source=0)
