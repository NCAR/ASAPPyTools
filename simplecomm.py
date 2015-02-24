'''
SimpleComm - Simple MPI Communication

The SimpleComm class is designed to provide a simplified MPI-based
communication strategy using the MPI4Py module.

To accomplish this task, the SimpleComm object provides a single communication
pattern with a simple, light-weight API.  The communication pattern is a
common 'master'/'slave' pattern, with the 0th rank assumed to be the 'master'
rank.  The SimpleComm API provides a way of sending data out from the 'master'
rank to the 'slave' ranks, and for collecting the data from the 'slave' ranks
back on the 'master' rank.

PARTITIONING:

Within the SimpleComm paradigm, the 'master' rank is assumed to be responsible
for partitioning (or distributing) the necessary work to the 'slave' ranks.
The *partition* mathod provides this functionality.  Using a *partition
function*, the *partition* method takes data known on the 'master' rank and
gives each 'slave' rank a part of the data according to the algorithm of the
partition function.

The *partition* method is *synchronous*, meaning that every rank (from the
'master' rank to all of the 'slave' ranks) must be in synch when the method
is called.  This means that every rank must participate in the call, and
every rank will wait until all of the data has been partitioned before
continuing.  Remember, whenever the 'master' rank speaks, all of the
'slave' ranks listen!  And they continue to listen until dismissed by the
'master' rank.

Additionally, the 'master' rank can be considered *involved* or *uninvolved*
in the partitioning process.  If the 'master' rank is *involved*, then the
master will take a part of the data for itself.  If the 'master' is
*uninvolved*, then the data will be partitioned only across the 'slave' ranks.

COLLECTING:

Once each 'slave' has received its assigned part of the data, the 'slave'
will perform some work pertaining to the data it received.  In such a case,
the 'slave' may (though not necessarily) return one or more results back to
the 'master'.  The *collect* method provides this functionality.

The *collect* method is *asynchronous*, meaning that each slave can send
its data back to the master at any time and in any order.  Since the 'master'
rank does not care where the data came from, the 'master' rank simply receives
the result from the 'slave' rank and processes it.  Hence, all that matters
is that for every *collect* call made by all of the 'slave' ranks, a *collect*
call must also be made by the 'master' rank.

The *collect* method is a *handshake* method, meaning that while the 'master'
rank doesn't care which 'slave' rank sends it data, the 'master' rank does
acknowledge the 'slave' rank and record the 'slave' rank's identity.

REDUCING:

In general, it is assumed that each 'slave' rank works independently from the
other 'slave' ranks.  However, it may be occasionally necessary for the 'slave'
ranks to know something about the work being done on (or the data given to)
each of the other ranks.  The only allowed communication of this type is
provided by the *allreduce* method.

The *allreduce* method allows for *reductions* of the data distributed across
all of the ranks to be made available to every rank.  Reductions are operations
such as 'max', 'min', 'sum', and 'prod', which compute and distribute to the
ranks the 'maximum', 'minimum', 'sum', or 'product' of the data distributed
across the ranks.  Since the *reduction* computes a reduced quantity of data
distributed across all ranks, the *allreduce* method is a *synchronous* method
(i.e., all ranks must participate in the call, including the 'master').

DIVIDING:

It can be occasionally useful to subdivide the 'slave' ranks into different
groups to perform different tasks in each group.  When this is necessary, the
'master' rank will assign itself and each 'slave' rank a *color* ID.  Then,
the 'master' will assign each rank (including itself) to 2 new groups:

    (1) Each rank with the same color ID will be assigned to the same group,
        and within this new *color* group, each rank will be given a new rank
        ID ranging from 0 (identifying the color group's 'master' rank) to
        the number of 'slave' ranks in the color group.  This is called the
        *monocolor* grouping.

    (2) Each rank with the same new rank ID across all color groups will be
        assigned to the same group.  Hence, all ranks with rank ID 0 (but
        different color IDs) will be in the same group, all ranks with rank ID
        1 (but different color IDs) will be the in another group, etc.  This
        is called the *multicolor* grouping.  NOTE: This grouping will look
        like grouping (1) except with the rank ID and the color ID swapped.

The *divide* method provides this functionality, and it returns 2 new
SimpleComm objects for each of the 2 groupings described above.  This means
that within each group, the same *partitioning*, *collecting*, and *reducing*
operations can be performed in the same way as described above for the *global*
group.

_______________________________________________________________________________
Created on Feb 4, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''

from functools import partial
from collections import defaultdict
from numpy import rank

## Define the supported reduction operators
OPERATORS = ['sum', 'prod', 'max', 'min']

## Define the reduction operators map
__OP_MAP = {'sum': {'py': sum,
                    'np': 'sum',
                    'mpi': 'SUM'},
            'prod': {'py': partial(reduce, lambda x, y: x * y),
                     'np': 'prod',
                     'mpi': 'PROD'},
            'max': {'py': max,
                    'np': 'max',
                    'mpi': 'MAX'},
            'min': {'py': min,
                    'np': 'min',
                    'mpi': 'MIN'}}


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
        Get the integer rank ID of this MPI process in this communicator.

        @return  The integer rank ID of this MPI process
                 (Unique to this MPI process and communicator)
        '''
        return 0

    def is_master(self):
        '''
        Simple check to determine if this MPI process is on the 'master' rank
        (i.e., if the rank ID is 0).

        @return  True if this MPI process is on the master rank, False
                 otherwise.
        '''
        return True

    def get_color(self):
        '''
        Get the integer color ID of this MPI process in this communicator.

        @return  The integer color ID of this MPI communicator
        '''
        return None

    def get_group(self):
        '''
        Get the group ID associated with the color ID of this MPI communicator.

        @return  The group ID of this communicator
        '''
        return None

    def allreduce(self, data, op):
        '''
        An AllReduction operation.

        The data is "reduced" across all ranks in the communicator.  The
        data reduction is returned to all ranks in the communicator.  (Reduce
        operations such as 'sum', 'prod', 'min', and 'max' are allowed.)

        @param  data  The data to be reduced

        @param  op    A string identifier for a reduce operation

        @return  The single value constituting the reduction of the input data.
                 (Same on all ranks in this communicator.)
        '''
        if (isinstance(data, dict)):
            totals = {}
            for k, v in data.items():
                totals[k] = self.allreduce(v, op)
            return totals
        elif self._is_ndarray(data):
            return self.allreduce(
                getattr(self._numpy, __OP_MAP[op]['np'])(data), op)
        elif hasattr(data, '__len__'):
            return self.allreduce(
                __OP_MAP[op]['py'](data), op)
        else:
            return data

    def partition(self, data=None, func=None, involved=False):
        '''
        Send data from the 'master' rank to 'slave' ranks.  By default, the
        data is duplicated from the 'master' rank onto every 'slave' rank.

        If a partition function is supplied via the "func" argument, then the
        data will be partitioned across the 'slave' ranks, giving each 'slave'
        rank a different part of the data according to the partition function
        supplied.

        If the "involved" argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'master' rank.  Otherwise, ("involved" argument is
        False) the data will be partitioned only across the 'slave' ranks.

        @param  data  The data to be partitioned across the ranks in the
                      communicator.

        @param  func  A PartitionFunction object (i.e., an object that has
                      three-argument __call__(data, index, size) method that
                      returns a part of the data given the index and assumed
                      size of the partition)

        @param  involved  True, if a part of the data should be given to the
                          'master' rank in addition to the 'slave' ranks.
                          False, otherwise.

        @return  A (possibly partitioned) subset (i.e., part) of the data
        '''
        op = func if func else lambda *x: x[0]
        return op(data, 0, 1)

    def collect(self, data=None):
        '''
        Send data from a 'slave' rank to the 'master' rank.  If the calling
        MPI process is the 'master' rank, then it receives and returns the
        data sent from the 'slave'.  If the calling MPI process is a 'slave'
        rank, then it sends the data to the 'master' rank.

        NOTE: This method cannot be used for communication between the 'master'
        rank and itself.  Attempting this will cause the code to hang.

        @param  data  The data to be collected asynchronously on the 'master'
                      rank.

        @return  On the 'master' rank, a dictionary containing the source
                 rank ID and the the data collected.  None on all other ranks.
        '''
        err_msg = 'Collection cannot be used in serial operation'
        raise RuntimeError(err_msg)

    def divide(self, group):
        '''
        Divide this communicator's ranks into 2 kinds of groups: (1) groups
        with ranks of the same color ID but different rank IDs, and (2) groups
        with ranks of the same rank ID but different color IDs.

        @param  group  A unique group ID to which will be assigned an integer
                       color ID ranging from 0 to the number of group ID's
                       supplied across all ranks

        @return  A tuple containing (first) the SimpleComm for ranks with the
                 same color ID and (second) the SimpleComm for ranks with the
                 same rank ID (but different colors)
        '''
        err_msg = 'Division cannot be done on a serial communicator'
        raise RuntimeError(err_msg)


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

        ## The MPI communicator (by default, COMM_WORLD)
        self._comm = self._mpi.COMM_WORLD

        ## The color ID associated with this communicator
        self._color = None

        ## The group ID associated with the color ID
        self._group = None

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator (including
        the master rank).

        @return  The integer number of ranks in this communicator.
                 (Same on all ranks in this communicator.)
        '''
        return self._comm.Get_size()

    def get_rank(self):
        '''
        Get the integer rank ID of this MPI process in this communicator.

        @return  The integer rank ID of this MPI process
                 (Unique to this MPI process and communicator)
        '''
        return self._comm.Get_rank()

    def is_master(self):
        '''
        Simple check to determine if this MPI process is on the 'master' rank
        (i.e., if the rank ID is 0).

        @return  True if this MPI process is on the master rank, False
                 otherwise.
        '''
        return (self._comm.Get_rank() == 0)

    def get_color(self):
        '''
        Get the integer color ID of this MPI process in this communicator.

        @return  The integer color ID of this MPI communicator
        '''
        return self._color

    def get_group(self):
        '''
        Get the group ID associated with the color ID of this MPI communicator.

        @return  The group ID of this communicator
        '''
        return self._group

    def allreduce(self, data, op):
        '''
        An AllReduction operation.

        The data is "reduced" across all ranks in the communicator.  The
        data reduction is returned to all ranks in the communicator.  (Reduce
        operations such as 'sum', 'prod', 'min', and 'max' are allowed.)

        @param  data  The data to be reduced

        @param  op    A string identifier for a reduce operation

        @return  The single value constituting the reduction of the input data.
                 (Same on all ranks in this communicator.)
        '''
        if (isinstance(data, dict)):
            all_list = self._comm.gather(SimpleComm.allreduce(self, data, op))
            if self.is_master():
                all_dict = defaultdict(list)
                for d in all_list:
                    for k, v in d.items():
                        all_dict[k].append(v)
                result = {}
                for k, v in all_dict.items():
                    result[k] = SimpleComm.allreduce(self, v, op)
            else:
                result = None
            return self.scatter(result)
        else:
            return self.comm.allreduce(
                SimpleComm.allreduce(self, data,
                                     op=getattr(self._mpi, op.upper())))

    def partition(self, data=None, func=None, involved=False):
        '''
        Send data from the 'master' rank to 'slave' ranks.  By default, the
        data is duplicated from the 'master' rank onto every 'slave' rank.

        If a partition function is supplied via the "func" argument, then the
        data will be partitioned across the 'slave' ranks, giving each 'slave'
        rank a different part of the data according to the partition function
        supplied.

        If the "involved" argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'master' rank.  Otherwise, ("involved" argument is
        False) the data will be partitioned only across the 'slave' ranks.

        @param  data  The data to be partitioned across the ranks in the
                      communicator.

        @param  func  A PartitionFunction object (i.e., an object that has
                      three-argument __call__(data, index, size) method that
                      returns a part of the data given the index and assumed
                      size of the partition)

        @param  involved  True, if a part of the data should be given to the
                          'master' rank in addition to the 'slave' ranks.
                          False, otherwise.

        @return  A (possibly partitioned) subset (i.e., part) of the data
        '''
        if self.is_master():
            op = func if func else lambda *x: x[0]
            if self.get_size() > 1:
                reqs = [self._comm.isend(op(data, i, self.get_size()), dest=i)
                        for i in xrange(1, self.get_size())]
                self._mpi.Request.Waitall(reqs)
            return op(data, 0, self.get_size())
        else:
            return self._comm.recv(source=0)

    def collect(self, data=None):
        '''
        Send data from a 'slave' rank to the 'master' rank.  If the calling
        MPI process is the 'master' rank, then it receives and returns the
        data sent from the 'slave'.  If the calling MPI process is a 'slave'
        rank, then it sends the data to the 'master' rank.

        @param  data  The data to be collected asynchronously on the 'master'
                      rank.

        @return  On the 'master' rank, a dictionary containing the source
                 rank ID and the the data collected.  None on all other ranks.
        '''
        if self.get_size() > 1:
            if self.is_master():
                return self._comm.recv(source=self._mpi.ANY_SOURCE, tag=111)
            else:
                self._comm.send(data, dest=0, tag=111)
        else:
            err_msg = 'Collection cannot be used in a 1-rank communicator'
            raise RuntimeError(err_msg)

    def divide(self, group):
        '''
        Divide this communicator's ranks into 2 kinds of groups: (1) groups
        with ranks of the same color ID but different rank IDs, and (2) groups
        with ranks of the same rank ID but different color IDs.

        @param  group  A unique group ID to which will be assigned an integer
                       color ID ranging from 0 to the number of group ID's
                       supplied across all ranks

        @return  A tuple containing (first) the SimpleComm for ranks with the
                 same color ID and (second) the SimpleComm for ranks with the
                 same rank ID (but different colors)
        '''
        if self.get_size() > 1:
            allgroups = list(set(self._comm.allgather(group)))
            color = allgroups.index(group)
            monocomm = SimpleCommMPI()
            monocomm._color = color
            monocomm._group = group
            monocomm._comm = self._comm.split(color)

            rank = monocomm.get_rank()
            multicomm = SimpleCommMPI()
            multicomm._color = rank
            multicomm._group = rank
            multicomm._comm = self._comm.split(rank)

            return monocomm, multicomm
        else:
            err_msg = 'Division cannot be done on a 1-rank communicator'
            raise RuntimeError(err_msg)

