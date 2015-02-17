'''
SimpleComm - Simple MPI Communication

The SimpleComm class is designed to provide a simplified interface to the
MPI4Py module.

_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''

#==============================================================================
# SimpleComm - Simple Communicator
#==============================================================================
class SimpleComm(object):
    '''
    Simple Communicator
    '''

    def __init__(self, mpi=True):
        '''
        Constructor

        @param  mpi  A boolean flag indicating whether the SimpleComm should
                     use MPI for communication or not.  If this is False, then
                     the SimpleComm object will assume serial operation and
                     only communicate with itself.
        '''

        # If mpi, try importing the MPI4Py MPI module
        if mpi:
            try:
                from mpi4py import MPI
                COMM = MPI.COMM_WORLD
            except:
                err_msg = 'MPI could not be found.'
                raise ImportError(err_msg)
        else:
            MPI = None
            COMM = None

        ## Hold on to the MPI module
        self.mpi = MPI

        ## Hold on to the MPI communicator (by default, COMM_WORLD)
        self.comm = COMM

        # Try importing the Numpy module
        try:
            import numpy
        except:
            numpy = None

        ## Hold on to the Numpy module
        self.numpy = numpy

        ## Local buffer for self-communication
        self.__buffer = None

    def __is_ndarray(self, data):
        '''
        Helper function to determing if a given data object is a Numpy
        NDArray object or not.

        @param  data  The data object to be tested

        @return  True if the data object is an NDarray, False otherwise.
        '''
        if self.numpy:
            return type(data) is self.numpy.ndarray
        else:
            return False

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator (including
        the master rank).

        @return  The integer number of ranks in this communicator.
                 (Same on all ranks in this communicator.)
        '''
        if self.comm:
            return self.comm.Get_size()
        else:
            return 1

    def is_master(self):
        '''
        Simple check to determine if this MPI task is on the 'master' rank
        (rank 0).

        @return  True if this MPI task is on the master rank, False otherwise.
        '''
        if self.comm:
            return (self.comm.Get_rank() == 0)
        else:
            return True

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
        if self.comm and self.get_size() > 1:
            if hasattr(data, '__len__'):
                return op(self.comm.allgather(op(data)))
            else:
                return op(self.comm.allgather(data))
        else:
            if hasattr(data, '__len__'):
                return op(data)
            else:
                return data

    def gather(self, data=None):
        '''
        Send data into the 'master' rank from the 'slave' ranks.  If this
        MPI task is on the 'master' rank, then this receives the data from
        the 'slave' ranks and assumes that all 'slave' ranks have sent data.
        If this MPI task is on a 'slave' rank, then this sends the data to the
        'master' rank and assumes that the 'master' rank will receive.

        @param  data  The data to be gathered on the 'master' rank.

        @return  A list of length equal to the communicator size containing
                 each rank's given data.
        '''
        if self.comm and self.get_size() > 1:
            return self.comm.gather(data, root=0)
        else:
            return [data]

    def scatter(self, data=None, part=None, skip=False):
        '''
        Send data from the 'master' rank to the 'slave' ranks.  If this MPI
        task is on the 'master' rank, then this sends the data to all 'slave'
        ranks and assumes that every 'slave' rank will receive the data.  If
        this MPI task is on a 'slave' rank, then this receives the data from
        the 'master' rank and assumes that the 'master' rank has sent the data.

        @param  data  The data to be scattered from the 'master' rank.

        @param  part  A three-argument partitioning function.  Allowing the
                      data to be partitioned across the 'slave' ranks.

        @param  skip  A boolean flag indicating that the 'master' rank should
                      be "skipped" during the partitioning step.  (I.e., if
                      the 'master' rank is skipped during partitioning, then
                      the data is partitioned over the 'slaves' only.)

        @return  A (possibly partitioned) subset of the data on the 'master'
                 rank
        '''
        if self.is_master():
            op = part if part else lambda *x: x[0]
            j = int(skip)
            if self.comm and self.get_size() > 1:
                reqs = [self.comm.isend(op(data, i - j, self.get_size() - j), dest=i)
                        for i in xrange(1, self.get_size())]
                self.mpi.Request.Waitall(reqs)
            if skip:
                return None
            else:
                return op(data, 0, self.get_size())
        else:
            return self.comm.recv(source=0)
