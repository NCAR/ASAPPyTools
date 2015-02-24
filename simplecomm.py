'''
SimpleComm - Simple MPI Communication

The SimpleComm class is designed to provide a simplified MPI-based
communication strategy using the MPI4Py module.

To accomplish this task, the SimpleComm object provides a single communication
pattern with a simple, light-weight API.  The communication pattern is a
common 'manager'/'worker' pattern, with the 0th rank assumed to be the
'manager' rank.  The SimpleComm API provides a way of sending data out from the
'manager' rank to the 'worker' ranks, and for collecting the data from the
'worker' ranks back on the 'manager' rank.

PARTITIONING:

Within the SimpleComm paradigm, the 'manager' rank is assumed to be responsible
for partitioning (or distributing) the necessary work to the 'worker' ranks.
The *partition* mathod provides this functionality.  Using a *partition
function*, the *partition* method takes data known on the 'manager' rank and
gives each 'worker' rank a part of the data according to the algorithm of the
partition function.

The *partition* method is *synchronous*, meaning that every rank (from the
'manager' rank to all of the 'worker' ranks) must be in synch when the method
is called.  This means that every rank must participate in the call, and
every rank will wait until all of the data has been partitioned before
continuing.  Remember, whenever the 'manager' rank speaks, all of the
'worker' ranks listen!  And they continue to listen until dismissed by the
'manager' rank.

Additionally, the 'manager' rank can be considered *involved* or *uninvolved*
in the partitioning process.  If the 'manager' rank is *involved*, then the
master will take a part of the data for itself.  If the 'manager' is
*uninvolved*, then the data will be partitioned only across the 'worker' ranks.

COLLECTING:

Once each 'worker' has received its assigned part of the data, the 'worker'
will perform some work pertaining to the data it received.  In such a case,
the 'worker' may (though not necessarily) return one or more results back to
the 'manager'.  The *collect* method provides this functionality.

The *collect* method is *asynchronous*, meaning that each slave can send
its data back to the master at any time and in any order.  Since the 'manager'
rank does not care where the data came from, the 'manager' rank simply receives
the result from the 'worker' rank and processes it.  Hence, all that matters
is that for every *collect* call made by all of the 'worker' ranks, a *collect*
call must also be made by the 'manager' rank.

The *collect* method is a *handshake* method, meaning that while the 'manager'
rank doesn't care which 'worker' rank sends it data, the 'manager' rank does
acknowledge the 'worker' rank and record the 'worker' rank's identity.

REDUCING:

In general, it is assumed that each 'worker' rank works independently from the
other 'worker' ranks.  However, it may be occasionally necessary for the
'worker' ranks to know something about the work being done on (or the data
given to) each of the other ranks.  The only allowed communication of this
type is provided by the *allreduce* method.

The *allreduce* method allows for *reductions* of the data distributed across
all of the ranks to be made available to every rank.  Reductions are operations
such as 'max', 'min', 'sum', and 'prod', which compute and distribute to the
ranks the 'maximum', 'minimum', 'sum', or 'product' of the data distributed
across the ranks.  Since the *reduction* computes a reduced quantity of data
distributed across all ranks, the *allreduce* method is a *synchronous* method
(i.e., all ranks must participate in the call, including the 'manager').

DIVIDING:

It can be occasionally useful to subdivide the 'worker' ranks into different
groups to perform different tasks in each group.  When this is necessary, the
'manager' rank will assign itself and each 'worker' rank a *color* ID.  Then,
the 'manager' will assign each rank (including itself) to 2 new groups:

    (1) Each rank with the same color ID will be assigned to the same group,
        and within this new *color* group, each rank will be given a new rank
        ID ranging from 0 (identifying the color group's 'manager' rank) to
        the number of 'worker' ranks in the color group.  This is called the
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
from __builtin__ import None

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

    def _type_is_ndarray(self, dt):
        '''
        Helper function to determing if a given data object is a Numpy
        NDArray object or not.

        @param  dt  The type of the data object to be tested

        @return  True if the data object is an NDarray, False otherwise.
        '''
        if self._numpy:
            return dt is self._numpy.ndarray
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

    def is_manager(self):
        '''
        Simple check to determine if this MPI process is on the 'manager' rank
        (i.e., if the rank ID is 0).

        @return  True if this MPI process is on the master rank, False
                 otherwise.
        '''
        return self.get_rank() == 0

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
        elif self._type_is_ndarray(type(data)):
            return self.allreduce(
                getattr(self._numpy, __OP_MAP[op]['np'])(data), op)
        elif hasattr(data, '__len__'):
            return self.allreduce(
                __OP_MAP[op]['py'](data), op)
        else:
            return data

    def partition(self, data=None, func=None, involved=False):
        '''
        Send data from the 'manager' rank to 'worker' ranks.  By default, the
        data is duplicated from the 'manager' rank onto every 'worker' rank.

        If a partition function is supplied via the "func" argument, then the
        data will be partitioned across the 'worker' ranks, giving each
        'worker' rank a different part of the data according to the partition
        function supplied.

        If the "involved" argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'manager' rank.  Otherwise, ("involved" argument is
        False) the data will be partitioned only across the 'worker' ranks.

        @param  data  The data to be partitioned across the ranks in the
                      communicator.

        @param  func  A PartitionFunction object (i.e., an object that has
                      three-argument __call__(data, index, size) method that
                      returns a part of the data given the index and assumed
                      size of the partition)

        @param  involved  True, if a part of the data should be given to the
                          'manager' rank in addition to the 'worker' ranks.
                          False, otherwise.

        @return  A (possibly partitioned) subset (i.e., part) of the data
        '''
        op = func if func else lambda *x: x[0]
        if involved:
            return op(data, 0, 1)
        else:
            return None

    def collect(self, data=None):
        '''
        Send data from a 'worker' rank to the 'manager' rank.  If the calling
        MPI process is the 'manager' rank, then it receives and returns the
        data sent from the 'worker'.  If the calling MPI process is a 'worker'
        rank, then it sends the data to the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        @param  data  The data to be collected asynchronously on the 'manager'
                      rank.

        @return  On the 'manager' rank, a dictionary containing the source
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
            if self.is_manager():
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
            return self._comm.allreduce(
                SimpleComm.allreduce(self, data,
                                     op=getattr(self._mpi, op.upper())))

    def partition(self, data=None, func=None, involved=False):
        '''
        Send data from the 'manager' rank to 'worker' ranks.  By default, the
        data is duplicated from the 'manager' rank onto every 'worker' rank.

        If a partition function is supplied via the "func" argument, then the
        data will be partitioned across the 'worker' ranks, giving each
        'worker' rank a different part of the data according to the partition
        function supplied.

        If the "involved" argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'manager' rank.  Otherwise, ("involved" argument is
        False) the data will be partitioned only across the 'worker' ranks.

        @param  data  The data to be partitioned across the ranks in the
                      communicator.

        @param  func  A PartitionFunction object (i.e., an object that has
                      three-argument __call__(data, index, size) method that
                      returns a part of the data given the index and assumed
                      size of the partition)

        @param  involved  True, if a part of the data should be given to the
                          'manager' rank in addition to the 'worker' ranks.
                          False, otherwise.

        @return  A (possibly partitioned) subset (i.e., part) of the data
        '''
        if self.is_manager():
            op = func if func else lambda *x: x[0]
            j = int(not involved)
            for i in xrange(1, self.get_size()):

                # Get the part of the data to send to rank i
                part = op(data, i - j, self.get_size() - j)

                # Create the handshake message
                msg = {}
                msg['rank'] = self.get_rank()
                msg['type'] = type(part)
                msg['shape'] = part.shape if hasattr(part, 'shape') else None
                msg['dtype'] = part.dtype if hasattr(part, 'dtype') else None

                # Send the handshake message to the worker rank
                self._comm.send(msg, dest=i, tag=100)

                # Receive the acknowledgement from the worker
                ack = self._comm.recv(source=i, tag=101)

                # Check the acknowledgement, if bad skip this rank
                if not ack:
                    continue

                # If OK, send the data to the worker
                if self._type_is_ndarray(type(part)):
                    self._comm.Send(part, dest=i, tag=102)
                else:
                    self._comm.send(part, dest=i, tag=103)
            if involved:
                return op(data, 0, self.get_size())
            else:
                return None
        else:

            # Get the data message from the manager
            msg = self._comm.recv(source=0, tag=100)

            # Check the message content
            ack = type(msg) is dict and \
                all([key in msg for key in ['rank', 'type', 'shape', 'dtype']])

            # If the message is good, acknowledge
            self._comm.send(ack, dest=0, tag=101)

            # if acknowledgement is bad, skip
            if not ack:
                return None

            # Receive the data
            if self._type_is_ndarray(msg['type']):
                recvd = self._np.empty(msg['shape'], dtype=msg['dtype'])
                self._comm.Recv(recvd, source=0, tag=102)
            else:
                recvd = self._comm.recv(source=0, tag=103)
            return recvd

    def collect(self, data=None):
        '''
        Send data from a 'worker' rank to the 'manager' rank.  If the calling
        MPI process is the 'manager' rank, then it receives and returns the
        data sent from the 'worker'.  If the calling MPI process is a 'worker'
        rank, then it sends the data to the 'manager' rank.

        @param  data  The data to be collected asynchronously on the 'manager'
                      rank.

        @return  On the 'manager' rank, a dictionary containing the source
                 rank ID and the the data collected.  None on all other ranks.
        '''
        if self.get_size() > 1:
            if self.is_manager():

                # Receive the message from the worker
                msg = self._comm.recv(source=self._mpi.ANY_SOURCE, tag=200)

                # Check the message content
                ack = type(msg) is dict and \
                    all([key in msg for key in ['rank', 'type',
                                                'shape', 'dtype']])

                # Send acknowledgement back to the worker
                self._comm.send(ack, dest=msg['rank'], tag=201)

                # If acknowledgement is bad, don't receive
                if not ack:
                    return None

                # Receive the data
                if self._type_is_ndarray(msg['type']):
                    recvd = self._np.empty(msg['shape'], dtype=msg['dtype'])
                    self._comm.Recv(recvd, source=msg['rank'], tag=202)
                else:
                    recvd = self._comm.recv(source=msg['rank'], tag=203)
                return recvd

            else:

                # Create the handshake message
                msg = {}
                msg['rank'] = self.get_rank()
                msg['type'] = type(data)
                msg['shape'] = data.shape if hasattr(data, 'shape') else None
                msg['dtype'] = data.dtype if hasattr(data, 'dtype') else None

                # Send the handshake message to the manager
                self._comm.send(msg, dest=0, tag=200)

                # Receive the acknowledgement from the manager
                ack = self._comm.recv(source=0, tag=201)

                # Check the acknowledgement, if not OK skip
                if not ack:
                    return

                # If OK, send the data to the manager
                if self._type_is_ndarray(type(data)):
                    self._comm.Send(data, dest=0, tag=202)
                else:
                    self._comm.send(data, dest=0, tag=203)
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

