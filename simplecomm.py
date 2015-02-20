'''
SimpleComm - Simple MPI Communication

The SimpleComm class is designed to provide a simplified interface to the
MPI4Py module.

_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''

from functools import partial

# Define a supported reduction operator map
__OP_MAP = {'sum': sum,
            'prod': partial(reduce, lambda x, y: x * y),
            'max': max,
            'min': min}

#==============================================================================
# create_comm - Simple Communicator Factory Function
#==============================================================================
def create_comm(serial=False):
    '''
    This is a factory function for SimpleComm objects.  Depending on the
    argument given, it returns an instance of a serial or parallel SimpleComm
    object.

    @param  serial  A boolean flag with True indicating the desire for a
                    serial SimpleComm instance, and False incidicating the
                    desire for a parallel SimpleComm instance.

    @return  An instance of a SimpleComm object, either serial or parallel
    '''
    if type(serial) is not bool:
        raise TypeError('Serial parameter must be a bool')
    if serial:
        return SimpleComm()
    else:
        return SimpleCommMPI()


#==============================================================================
# SimpleComm - Simple Communicator
#==============================================================================
class SimpleComm(object):
    '''
    Simple Communicator for serial operation
    '''

    def __init__(self):
        '''
        Constructor
        '''

        # Try importing the Numpy module
        try:
            import numpy
        except:
            numpy = None

        ## To the Numpy module, if found
        self._numpy = numpy

    def _is_ndarray(self, data):
        '''
        Helper function to determing if a given data object is a Numpy
        NDArray object or not.

        @param  data  The data object to be tested

        @return  True if the data object is an NDarray, False otherwise.
        '''
        if self._numpy:
            return type(data) is self._numpy.ndarray
        else:
            return False

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator (including
        the master rank).

        @return  The integer number of ranks in this communicator.
                 (Same on all ranks in this communicator.)
        '''
        return 1

    def get_rank(self):
        '''
        Get the integer rank of this MPI process in this communicator.

        @return  The integer rank of this MPI process
                 (Unique to this MPI process)
        '''
        return 0

    def is_master(self):
        '''
        Simple check to determine if this MPI task is on the 'master' rank
        (rank 0).

        @return  True if this MPI task is on the master rank, False otherwise.
        '''
        return True

    def _allreduce(self, data, op):
        '''
        Reduction: Applies a function/operator to a collection of
        data values and reduces the data to a single value.  The 'sum'
        function is an example of a reduction operator, as is the 'max'
        function.  (Note: This implementation requires that the operator
        be commutative and associative.)

        @param  data  The data to be reduced

        @param  op    A supported reduction operator/function name

        @return  The single value constituting the reduction of the input data.
                 (Same on all ranks in this communicator.)
        '''
        if (isinstance(data, dict)):
            totals = {}
            for name in data:
                totals[name] = self._allreduce(data[name], op)
            return totals
        elif self._is_ndarray(data):
            return self._allreduce(getattr(self.numpy, op)(data), op)
        elif hasattr(data, '__len__'):
            return self._allreduce(__OP_MAP[op](data), op)
        else:
            return data

    def allsum(self, data):
        '''
        Sum reduction across all ranks in the SimpleComm domain.

        @param  data  The data to sum across all ranks

        @return  The sum of the data across all ranks
        '''
        return self._allreduce(data, 'sum')

    def allprod(self, data):
        '''
        Product reduction across all ranks in the SimpleComm domain.

        @param  data  The data to multiply across all ranks

        @return  The product of the data across all ranks
        '''
        self._allreduce(data, 'prod')

    def allmax(self, data):
        '''
        Maximum reduction across all ranks in the SimpleComm domain.

        @param  data  The data to search

        @return  The maximum of the data across all ranks
        '''
        self._allreduce(data, 'max')

    def allmin(self, data):
        '''
        Minimum reduction across all ranks in the SimpleComm domain.

        @param  data  The data to search

        @return  The minimum of the data across all ranks
        '''
        self._allreduce(data, 'min')

    def gather(self, data):
        '''
        Send data into the 'master' rank from the 'slave' ranks.  If this
        MPI task is on the 'master' rank, then this receives the data from
        the 'slave' ranks and assumes that all 'slave' ranks have sent data.
        If this MPI task is on a 'slave' rank, then this sends the data to the
        'master' rank and assumes that the 'master' rank will receive.

        @param  dummy  The data to be gathered on the 'master' rank.

        @return  A list of length equal to the communicator size containing
                 each rank's given data.
        '''
        return [data]

    def scatter(self, data, part=None):
        '''
        Send data from the 'master' rank to the 'slave' ranks.  If this MPI
        task is on the 'master' rank, then this sends the data to all 'slave'
        ranks and assumes that every 'slave' rank will receive the data.  If
        this MPI task is on a 'slave' rank, then this receives the data from
        the 'master' rank and assumes that the 'master' rank has sent the data.

        @param  data  The data to be scattered from the 'master' rank.

        @param  part  A three-argument partitioning function.  Allowing the
                      data to be partitioned across the 'slave' ranks.

        @return  A (possibly partitioned) subset of the data on the 'master'
                 rank
        '''
        op = part if part else lambda *x: x[0]
        return op(data, 0, 1)

    def pull(self, data):
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
        return self.gather(data)

    def push(self, data, part=None):
        '''
        Send data from the 'master' rank to the 'slave' ranks.  If this MPI
        task is on the 'master' rank, then this sends the data to all 'slave'
        ranks and assumes that every 'slave' rank will receive the data.  If
        this MPI task is on a 'slave' rank, then this receives the data from
        the 'master' rank and assumes that the 'master' rank has sent the data.

        @param  data  The data to be scattered from the 'master' rank.

        @param  part  A three-argument partitioning function.  Allowing the
                      data to be partitioned across the 'slave' ranks.

        @return  A (possibly partitioned) subset of the data on the 'master'
                 rank
        '''
        return self.scatter(data, part)


#==============================================================================
# SimpleCommMPI - Simple Communicator using MPI
#==============================================================================
class SimpleCommMPI(SimpleComm):
    '''
    Simple Communicator using MPI
    '''

    def __init__(self):
        '''
        Constructor
        '''
        super(SimpleCommMPI, self).__init__()

        # Try importing the MPI4Py MPI module
        try:
            from mpi4py import MPI
        except:
            err_msg = 'MPI could not be found.'
            raise ImportError(err_msg)

        ## Hold on to the MPI module
        self._mpi = MPI

        ## The "internal" MPI communicator (by default, COMM_WORLD)
        self._incomm = self._mpi.COMM_WORLD

        ## The "external" MPI communicator (by default, COMM_NULL)
        self._excomm = self._mpi.COMM_NULL

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator (including
        the master rank).

        @return  The integer number of ranks in this communicator.
                 (Same on all ranks in this communicator.)
        '''
        return self._incomm.Get_size()

    def get_rank(self):
        '''
        Get the integer rank of this MPI process in this communicator.

        @return  The integer rank of this MPI process
                 (Unique to this MPI process)
        '''
        return self._incomm.Get_rank()

    def is_master(self):
        '''
        Simple check to determine if this MPI task is on the 'master' rank
        (rank 0).

        @return  True if this MPI task is on the master rank, False otherwise.
        '''
        return (self._incomm.Get_rank() == 0)

    def _allreduce(self, data, op):
        '''
        Reduction: Applies a function/operator to a collection of
        data values and reduces the data to a single value.  The 'sum'
        function is an example of a reduction operator, as is the 'max'
        function.  (Note: This implementation requires that the operator
        be commutative and associative.)

        @param  data  The data to be reduced

        @param  op    A reduction operator/function

        @return  The single value constituting the reduction of the input data.
                 (None on the 'slave' ranks.)
        '''
        if (isinstance(data, dict)):
            totals = {}
            for name in data:
                totals[name] = self._allreduce(data[name], op)
            return totals
        else:
            return self.incomm.allreduce(
                     SimpleComm._allreduce(self, data,
                                           op=getattr(self._mpi, op.upper())))

    def gather(self, data):
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
        return self._incomm.gather(data)

    def scatter(self, data, part=None):
        '''
        Send data from the 'master' rank to the 'slave' ranks.  If this MPI
        task is on the 'master' rank, then this sends the data to all 'slave'
        ranks and assumes that every 'slave' rank will receive the data.  If
        this MPI task is on a 'slave' rank, then this receives the data from
        the 'master' rank and assumes that the 'master' rank has sent the data.

        @param  data  The data to be scattered from the 'master' rank.

        @param  part  A three-argument partitioning function.  Allowing the
                      data to be partitioned across the 'slave' ranks.

        @return  A (possibly partitioned) subset of the data on the 'master'
                 rank
        '''
        if self.is_master():
            op = part if part else lambda *x: x[0]
            if self.get_size() > 1:
                reqs = [self._incomm.isend(op(data, i, self.get_size()), dest=i)
                        for i in xrange(1, self.get_size())]
                self._mpi.Request.Waitall(reqs)
            return op(data, 0, self.get_size())
        else:
            return self._incomm.recv(source=0)

    def pull(self, data):
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
        if self._excomm == self._mpi.COMM_NULL:
            return SimpleComm.pull(self, data)
        return self._excomm.gather(data)

    def push(self, data, part=None):
        '''
        Send data from the 'master' rank to the 'slave' ranks.  If this MPI
        task is on the 'master' rank, then this sends the data to all 'slave'
        ranks and assumes that every 'slave' rank will receive the data.  If
        this MPI task is on a 'slave' rank, then this receives the data from
        the 'master' rank and assumes that the 'master' rank has sent the data.

        @param  data  The data to be scattered from the 'master' rank.

        @param  part  A three-argument partitioning function.  Allowing the
                      data to be partitioned across the 'slave' ranks.

        @return  A (possibly partitioned) subset of the data on the 'master'
                 rank
        '''
        if self._excomm == self._mpi.COMM_NULL:
            return SimpleComm.push(self, data, part)
        if self.is_master():
            op = part if part else lambda *x: x[0]
            if self.get_size() > 1:
                reqs = [self._incomm.isend(op(data, i, self.get_size()), dest=i)
                        for i in xrange(1, self.get_size())]
                self._mpi.Request.Waitall(reqs)
            return op(data, 0, self.get_size())
        else:
            return self._incomm.recv(source=0)

