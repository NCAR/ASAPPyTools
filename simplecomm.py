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
for partition (or distributing) the necessary work to the 'worker' ranks.
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
in the partition process.  If the 'manager' rank is *involved*, then the
master will take a part of the data for itself.  If the 'manager' is
*uninvolved*, then the data will be partitioned only across the 'worker' ranks.

*Partitioning* is a *synchronous* communication call that implements a
*static partitioning* algorithm.

RATIONING:

An alternative approach to the *partitioning* communication method is the
*rationing* communication method.  This method involves the individual
'worker' ranks requesting data to work on.  In this approach, each 'worker'
rank, when the 'worker' rank is ready, asks the 'manager' rank for a new
piece of data on which to work.  The 'manager' rank receives the request
and gives the next piece of data for processing out to the requesting
'worker' rank.  It doesn't matter what order the ranks request data, and
they do not all have to request data at the same time.  However, it is
critical to understand that if a 'worker' requests data when the 'manager'
rank does not listen for the request, or the 'manager' expects a 'worker'
to request work but the 'worker' never makes the request, the entire
process will hang and wait forever!

*Rationing* is an *asynchronous* communication call that allows the 'manager'
to implement a *dynamic partitioning* algorithm.

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
that within each group, the same *partition*, *collecting*, and *reducing*
operations can be performed in the same way as described above for the *global*
group.

_______________________________________________________________________________
Created on Feb 4, 2015

Author: Kevin Paul <kpaul@ucar.edu>
'''

from functools import partial
from collections import defaultdict

# Define the supported reduction operators
OPERATORS = ['sum', 'prod', 'max', 'min']

# Define the reduction operators map (Maps names to function names.
# The 'py' function names are passed to 'eval(*)' and executed as python code.
# The 'np' function names are passed to 'getattr(numpy,*)' and executed as
# numpy code.  The 'mpi' function names are passed to 'getattr(mpi4py,*)'
# and return an MPI operator object which is passed as an argument to MPI
# reduce functions.
_OP_MAP = {'sum': {'py': 'sum',
                   'np': 'sum',
                   'mpi': 'SUM'},
           'prod': {'py': 'partial(reduce, lambda x, y: x * y)',
                    'np': 'prod',
                    'mpi': 'PROD'},
           'max': {'py': 'max',
                   'np': 'max',
                   'mpi': 'MAX'},
           'min': {'py': 'min',
                   'np': 'min',
                   'mpi': 'MIN'}}


#==============================================================================
# create_comm - Simple Communicator Factory Function
#==============================================================================
def create_comm(serial=False):
    '''
    This is a factory function for creating SimpleComm objects.

    Depending on the argument given, it returns an instance of a serial or
    parallel SimpleComm object.

    Args:
        serial: A boolean flag with True indicating the desire for a
            serial SimpleComm instance, and False incidicating the
            desire for a parallel SimpleComm instance.

    Returns:
        An instance of a SimpleComm object, either serial (if serial == True)
        or parallel (if serial == False)

    Raises:
        TypeError, if the serial argument is not a bool.

    Examples:

        >>> sercomm = create_comm(serial=True)
        >>> type(sercomm)
        <class 'simplecomm.SimpleComm'>

        >>> parcomm = create_comm()
        >>> type(parcomm)
        <class 'simplecomm.SimpleCommMPI'>
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
    Simple Communicator for serial operation.

    Attributes:
        _numpy: Reference to the Numpy module, if found
        _color: The color associated with the communicator, if colored
        _group: The group ID associated with the communicator's color
    '''

    def __init__(self):
        '''
        Constructor.
        '''

        # Try importing the Numpy module
        try:
            import numpy
        except:
            numpy = None

        # To the Numpy module, if found
        self._numpy = numpy

        # The color ID associated with this communicator
        self._color = None

        # The group ID associated with the color
        self._group = None

    def _type_is_ndarray(self, dt):
        '''
        Helper function to determing if an object is a Numpy NDArray.

        Args:
            dt: The type of the data object to be tested

        Returns:
            True if the object is a Numpy NDArray.  
            Folse otherwise, or if the Numpy module was not found during
            the SimpleComm constructor.

        Examples:

            >>> _type_is_ndarray(type(1))
            False

            >>> alist = [1,2,3,4]
            >>> _type_is_ndarray(type(alist))
            False

            >>> aarray = numpy.array(alist)
            >>> _type_is_ndarray(type(aarray))
            True
        '''
        if self._numpy:
            return dt is self._numpy.ndarray
        else:
            return False

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator.

        The size includes the 'manager' rank.

        Returns:
            The integer number of ranks in this communicator.
        '''
        return 1

    def get_rank(self):
        '''
        Get the integer rank ID of this MPI process in this communicator.

        This call can be made independently from other ranks.

        Returns:
            The integer rank ID of this MPI process
        '''
        return 0

    def is_manager(self):
        '''
        Check if this MPI process is on the 'manager' rank (i.e., rank 0).

        This call can be made independently from other ranks.

        Returns:
            True, if this MPI process is on the master rank (or rank 0).
            False, otherwise.
        '''
        return self.get_rank() == 0

    def get_color(self):
        '''
        Get the integer color ID of this MPI process in this communicator.

        By default, a communicator's color is None, but a communicator can
        be divided into color groups using the 'divide' method.

        This call can be made independently from other ranks.

        Returns:
            The integer color of this MPI communicator
        '''
        return self._color

    def get_group(self):
        '''
        Get the group ID of this MPI communicator.

        The group ID is the argument passed to the 'divide' method, and it
        represents a unique identifier for all ranks in the same color group.
        It can be any type of object (e.g., a string name).

        This call can be made independently from other ranks.

        Returns:
            The group ID of this communicator
        '''
        return self._group

    def sync(self):
        '''
        Synchronize all MPI processes at the point of this call.

        Immediately after this method is called, you can guarantee that all
        ranks in this communicator will be synchronized.

        This call must be made by all ranks.
        '''
        return

    def allreduce(self, data, op):
        '''
        Perform an MPI AllReduction operation.

        The data is "reduced" across all ranks in the communicator, and the
        result is returned to all ranks in the communicator.  (Reduce
        operations such as 'sum', 'prod', 'min', and 'max' are allowed.)

        This call must be made by all ranks.

        Args:
            data: The data to be reduced
            op: A string identifier for a reduce operation (any string
                found in the OPERATORS list)

        Returns:
            The single value constituting the reduction of the input data.
            (The same value is returned on all ranks in this communicator.)
        '''
        if (isinstance(data, dict)):
            totals = {}
            for k, v in data.items():
                totals[k] = SimpleComm.allreduce(self, v, op)
            return totals
        elif self._type_is_ndarray(type(data)):
            return SimpleComm.allreduce(self,
                                        getattr(self._numpy,
                                                _OP_MAP[op]['np'])(data),
                                        op)
        elif hasattr(data, '__len__'):
            return SimpleComm.allreduce(self,
                                        eval(_OP_MAP[op]['py'])(data),
                                        op)
        else:
            return data

    def partition(self, data=None, func=None, involved=False):
        '''
        Partition and send data from the 'manager' rank to 'worker' ranks.

        By default, the data is partitioned using an "equal stride" across the
        data, with the stride equal to the number of ranks involved in the
        partitioning.  If a partition function is supplied via the `func`
        argument, then the data will be partitioned across the 'worker' ranks,
        giving each 'worker' rank a different part of the data according to
        the algorithm used by partition function supplied.

        If the `involved` argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'manager' rank.  Otherwise, ('involved' argument is
        False) the data will be partitioned only across the 'worker' ranks.

        This call must be made by all ranks.

        Args:
            data: The data to be partitioned across the ranks in the
                communicator.
            func: A PartitionFunction object/function that returns a part
                of the data given the index and assumed size of the
                partition
            involved: True, if a part of the data should be given to the
                'manager' rank in addition to the 'worker' ranks.
                False, otherwise.

        Returns:
            A (possibly partitioned) subset (i.e., part) of the data.
            Depending on the PartitionFunction used (or if it is used at all),
            this method may return a different part on each rank.
        '''
        op = func if func else lambda *x: x[0][x[1]::x[2]]
        if involved:
            return op(data, 0, 1)
        else:
            return None

    def ration(self, data=None):
        '''
        Send a single piece of data from the 'manager' rank to a 'worker' rank.

        If this method is called on a 'worker' rank, the worker will send a
        "request" for data to the 'manager' rank.  When the 'manager' receives
        this request, the 'manager' rank sends a single piece of data back to
        the requesting 'worker' rank.

        For each call to this function on a given 'worker' rank, there must
        be a matching call to this function made on the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        Args:
            data: The data to be asynchronously sent to the 'worker' rank/

        Returns:
            On the 'worker' rank, the data sent by the manager.  On the
            'manager' rank, None.

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        '''
        err_msg = 'Rationing cannot be used in serial operation'
        raise RuntimeError(err_msg)

    def collect(self, data=None):
        '''
        Send data from a 'worker' rank to the 'manager' rank.

        If the calling MPI process is the 'manager' rank, then it will
        receive and return the data sent from the 'worker'.  If the calling
        MPI process is a 'worker' rank, then it will send the data to the
        'manager' rank.

        For each call to this function on a given 'worker' rank, there must
        be a matching call to this function made on the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        Args:
            data: The data to be collected asynchronously on the 'manager'
                rank.

        Returns:
            On the 'manager' rank, a tuple containing the source rank ID
            and the data collected.  None on all other ranks.

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        '''
        err_msg = 'Collection cannot be used in serial operation'
        raise RuntimeError(err_msg)

    def divide(self, group):
        '''
        Divide this communicator's ranks into groups.

        Creates and returns two (2) kinds of groups:

            (1) groups with ranks of the same color ID but different rank IDs
                (called a "monocolor" group), and

            (2) groups with ranks of the same rank ID but different color IDs
                (called a "multicolor" group).

        Args:
            group: A unique group ID to which will be assigned an integer
                color ID ranging from 0 to the number of group ID's
                supplied across all ranks

        Returns:
            A tuple containing (first) the "monocolor" SimpleComm for ranks
            with the same color ID (but different rank IDs) and (second) the
            "multicolor" SimpleComm for ranks with the same rank ID (but
            different color IDs)

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        '''
        err_msg = 'Division cannot be done on a serial communicator'
        raise RuntimeError(err_msg)


#==============================================================================
# SimpleCommMPI - Simple Communicator using MPI
#==============================================================================
class SimpleCommMPI(SimpleComm):

    '''
    Simple Communicator using MPI.

    Attributes:
        PART_MSG_TAG: Partition Message Tag
        PART_ACK_TAG: Partition Acknowledgement Tag
        PART_NPY_TAG: Partition Numpy Send/Recv Tag
        PART_PYT_TAG: Partition Python Send/Recv Tag
        RATN_REQ_TAG: Ration Request Tag
        RATN_MSG_TAG: Ration Message Tag
        RATN_ACK_TAG: Ration Acknowledgement Tag
        RATN_NPY_TAG: Ration Numpy Send/Recv Tag
        RATN_PYT_TAG: Ration Python Send/Recv Tag
        CLCT_MSG_TAG: Collect Message Tag
        CLCT_ACK_TAG: Collect Acknowledgement Tag
        CLCT_NPY_TAG: Collect Numpy Send/Recv Tag
        CLCT_PYT_TAG: Collect Python Send/Recv Tag
        _mpi: A reference to the mpi4py.MPI module
        _comm: A reference to the mpi4py.MPI communicator
    '''

    PART_MSG_TAG = 100  # Partition Message Tag
    PART_ACK_TAG = 101  # Partition Acknowledgement Tag
    PART_NPY_TAG = 102  # Partition Numpy Send/Recv Tag
    PART_PYT_TAG = 103  # Partition Python Send/Recv Tag

    RATN_REQ_TAG = 200  # Ration Request Tag
    RATN_MSG_TAG = 201  # Ration Message Tag
    RATN_ACK_TAG = 202  # Ration Acknowledgement Tag
    RATN_NPY_TAG = 203  # Ration Numpy Send/Recv Tag
    RATN_PYT_TAG = 204  # Ration Python Send/Recv Tag

    CLCT_MSG_TAG = 301  # Collect Message Tag
    CLCT_ACK_TAG = 302  # Collect Acknowledgement Tag
    CLCT_NPY_TAG = 303  # Collect Numpy Send/Recv Tag
    CLCT_PYT_TAG = 304  # Collect Python Send/Recv Tag

    def __init__(self):
        '''
        Constructor.
        '''

        # Call the base class constructor
        super(SimpleCommMPI, self).__init__()

        # Try importing the MPI4Py MPI module
        try:
            from mpi4py import MPI
        except:
            err_msg = 'MPI could not be found.'
            raise ImportError(err_msg)

        # Hold on to the MPI module
        self._mpi = MPI

        # The MPI communicator (by default, COMM_WORLD)
        self._comm = self._mpi.COMM_WORLD

    def __del__(self):
        '''
        Destructor.

        Free the communicator if this SimpleComm goes out of scope
        '''
        if self._comm != self._mpi.COMM_WORLD:
            self._comm.Free()

    def get_size(self):
        '''
        Get the integer number of ranks in this communicator.

        The size includes the 'manager' rank.

        Returns:
            The integer number of ranks in this communicator.
        '''
        return self._comm.Get_size()

    def get_rank(self):
        '''
        Get the integer rank ID of this MPI process in this communicator.

        This call can be made independently from other ranks.

        Returns:
            The integer rank ID of this MPI process
        '''
        return self._comm.Get_rank()

    def sync(self):
        '''
        Synchronize all MPI processes at the point of this call.

        Immediately after this method is called, you can guarantee that all
        ranks in this communicator will be synchronized.

        This call must be made by all ranks.
        '''
        self._comm.Barrier()

    def allreduce(self, data, op):
        '''
        Perform an MPI AllReduction operation.

        The data is "reduced" across all ranks in the communicator, and the
        result is returned to all ranks in the communicator.  (Reduce
        operations such as 'sum', 'prod', 'min', and 'max' are allowed.)

        This call must be made by all ranks.

        Args:
            data: The data to be reduced
            op: A string identifier for a reduce operation (any string
                found in the OPERATORS list)

        Returns:
            The single value constituting the reduction of the input data.
            (The same value is returned on all ranks in this communicator.)
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
                return self._comm.bcast(result)
            else:
                return self._comm.bcast(None)
        else:
            return self._comm.allreduce(SimpleComm.allreduce(self, data, op),
                                        op=getattr(self._mpi,
                                                   _OP_MAP[op]['mpi']))

    def partition(self, data=None, func=None, involved=False):
        '''
        Partition and send data from the 'manager' rank to 'worker' ranks.

        By default, the data is partitioned using an "equal stride" across the
        data, with the stride equal to the number of ranks involved in the
        partitioning.  If a partition function is supplied via the 'func'
        argument, then the data will be partitioned across the 'worker' ranks,
        giving each 'worker' rank a different part of the data according to
        the algorithm used by partition function supplied.

        If the 'involved' argument is True, then a part of the data (as
        determined by the given partition function, if supplied) will be
        returned on the 'manager' rank.  Otherwise, ('involved' argument is
        False) the data will be partitioned only across the 'worker' ranks.

        This call must be made by all ranks.

        Args:
            data: The data to be partitioned across the ranks in the
                communicator.
            func: A PartitionFunction object/function that returns a part
                of the data given the index and assumed size of the
                partition.
            involved: True, if a part of the data should be given to the
                'manager' rank in addition to the 'worker' ranks.
                False, otherwise.

        Returns:
            A (possibly partitioned) subset (i.e., part) of the data.
            Depending on the PartitionFunction used (or if it is used at all),
            this method may return a different part on each rank.
        '''
        if self.is_manager():
            op = func if func else lambda *x: x[0][x[1]::x[2]]
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
                self._comm.send(msg, dest=i, tag=SimpleCommMPI.PART_MSG_TAG)

                # Receive the acknowledgement from the worker
                ack = self._comm.recv(source=i, tag=SimpleCommMPI.PART_ACK_TAG)

                # Check the acknowledgement, if bad skip this rank
                if not ack:
                    continue

                # If OK, send the data to the worker
                if self._type_is_ndarray(type(part)):
                    self._comm.Send(self._numpy.array(part), dest=i,
                                    tag=SimpleCommMPI.PART_NPY_TAG)
                else:
                    self._comm.send(part, dest=i,
                                    tag=SimpleCommMPI.PART_PYT_TAG)

            if involved:
                return op(data, 0, self.get_size())
            else:
                return None
        else:

            # Get the data message from the manager
            msg = self._comm.recv(source=0, tag=SimpleCommMPI.PART_MSG_TAG)

            # Check the message content
            ack = type(msg) is dict and \
                all([key in msg for key in ['rank', 'type', 'shape', 'dtype']])

            # If the message is good, acknowledge
            self._comm.send(ack, dest=0, tag=SimpleCommMPI.PART_ACK_TAG)

            # if acknowledgement is bad, skip
            if not ack:
                return None

            # Receive the data
            if self._type_is_ndarray(msg['type']):
                recvd = self._numpy.empty(msg['shape'], dtype=msg['dtype'])
                self._comm.Recv(recvd, source=0,
                                tag=SimpleCommMPI.PART_NPY_TAG)
            else:
                recvd = self._comm.recv(source=0,
                                        tag=SimpleCommMPI.PART_PYT_TAG)
            return recvd

    def ration(self, data=None):
        '''
        Send a single piece of data from the 'manager' rank to a 'worker' rank.

        If this method is called on a 'worker' rank, the worker will send a
        "request" for data to the 'manager' rank.  When the 'manager' receives
        this request, the 'manager' rank sends a single piece of data back to
        the requesting 'worker' rank.

        For each call to this function on a given 'worker' rank, there must
        be a matching call to this function made on the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        Args:
            data: The data to be asynchronously sent to the 'worker' rank/

        Returns:
            On the 'worker' rank, the data sent by the manager.  On the
            'manager' rank, None.

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        '''
        if self.get_size() > 1:
            if self.is_manager():

                # Listen for a requesting worker rank
                rank = self._comm.recv(source=self._mpi.ANY_SOURCE,
                                       tag=SimpleCommMPI.RATN_REQ_TAG)

                # Create the handshake message
                msg = {}
                msg['type'] = type(data)
                msg['shape'] = data.shape if hasattr(data, 'shape') else None
                msg['dtype'] = data.dtype if hasattr(data, 'dtype') else None

                # Send the handshake message to the requesting worker
                self._comm.send(msg, dest=rank,
                                tag=SimpleCommMPI.RATN_MSG_TAG)

                # Receive the acknowledgement from the requesting worker
                ack = self._comm.recv(source=rank,
                                      tag=SimpleCommMPI.RATN_ACK_TAG)

                # Check the acknowledgement, if not OK skip
                if not ack:
                    return

                # If OK, send the data to the requesting worker
                if self._type_is_ndarray(type(data)):
                    self._comm.Send(data, dest=rank,
                                    tag=SimpleCommMPI.RATN_NPY_TAG)
                else:
                    self._comm.send(data, dest=rank,
                                    tag=SimpleCommMPI.RATN_PYT_TAG)
            else:

                # Send a request for data to the manager
                self._comm.send(self.get_rank(), dest=0,
                                tag=SimpleCommMPI.RATN_REQ_TAG)

                # Receive the handshake message from the manager
                msg = self._comm.recv(source=0,
                                      tag=SimpleCommMPI.RATN_MSG_TAG)

                # Check the message content
                ack = type(msg) is dict and \
                    all([key in msg for key in ['type', 'shape', 'dtype']])

                # Send acknowledgement back to the manager
                self._comm.send(ack, dest=0, tag=SimpleCommMPI.RATN_ACK_TAG)

                # If acknowledgement is bad, don't receive
                if not ack:
                    return None

                # Receive the data from the manager
                if self._type_is_ndarray(msg['type']):
                    recvd = self._numpy.empty(msg['shape'], dtype=msg['dtype'])
                    self._comm.Recv(recvd, source=0,
                                    tag=SimpleCommMPI.RATN_NPY_TAG)
                else:
                    recvd = self._comm.recv(source=0,
                                            tag=SimpleCommMPI.RATN_PYT_TAG)
                return recvd
        else:
            err_msg = 'Rationing cannot be used in 1-rank parallel operation'
            raise RuntimeError(err_msg)

    def collect(self, data=None):
        '''
        Send data from a 'worker' rank to the 'manager' rank.

        If the calling MPI process is the 'manager' rank, then it will
        receive and return the data sent from the 'worker'.  If the calling
        MPI process is a 'worker' rank, then it will send the data to the
        'manager' rank.

        For each call to this function on a given 'worker' rank, there must
        be a matching call to this function made on the 'manager' rank.

        NOTE: This method cannot be used for communication between the
        'manager' rank and itself.  Attempting this will cause the code to
        hang.

        Args:
            data: The data to be collected asynchronously on the 'manager'
                rank.

        Returns:
            On the 'manager' rank, a tuple containing the source rank ID
            and the the data collected.  None on all other ranks.

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        '''
        if self.get_size() > 1:
            if self.is_manager():

                # Receive the message from the worker
                msg = self._comm.recv(source=self._mpi.ANY_SOURCE,
                                      tag=SimpleCommMPI.CLCT_MSG_TAG)

                # Check the message content
                ack = type(msg) is dict and \
                    all([key in msg for key in ['rank', 'type',
                                                'shape', 'dtype']])

                # Send acknowledgement back to the worker
                self._comm.send(ack, dest=msg['rank'],
                                tag=SimpleCommMPI.CLCT_ACK_TAG)

                # If acknowledgement is bad, don't receive
                if not ack:
                    return None

                # Receive the data
                if self._type_is_ndarray(msg['type']):
                    recvd = self._numpy.empty(msg['shape'], dtype=msg['dtype'])
                    self._comm.Recv(recvd, source=msg['rank'],
                                    tag=SimpleCommMPI.CLCT_NPY_TAG)
                else:
                    recvd = self._comm.recv(source=msg['rank'],
                                            tag=SimpleCommMPI.CLCT_PYT_TAG)
                return msg['rank'], recvd

            else:

                # Create the handshake message
                msg = {}
                msg['rank'] = self.get_rank()
                msg['type'] = type(data)
                msg['shape'] = data.shape if hasattr(data, 'shape') else None
                msg['dtype'] = data.dtype if hasattr(data, 'dtype') else None

                # Send the handshake message to the manager
                self._comm.send(msg, dest=0, tag=SimpleCommMPI.CLCT_MSG_TAG)

                # Receive the acknowledgement from the manager
                ack = self._comm.recv(source=0, tag=SimpleCommMPI.CLCT_ACK_TAG)

                # Check the acknowledgement, if not OK skip
                if not ack:
                    return

                # If OK, send the data to the manager
                if self._type_is_ndarray(type(data)):
                    self._comm.Send(data, dest=0,
                                    tag=SimpleCommMPI.CLCT_NPY_TAG)
                else:
                    self._comm.send(data, dest=0,
                                    tag=SimpleCommMPI.CLCT_PYT_TAG)
        else:
            err_msg = 'Collection cannot be used in a 1-rank communicator'
            raise RuntimeError(err_msg)

    def divide(self, group):
        '''
        Divide this communicator's ranks into groups.

        Creates and returns two (2) kinds of groups:

            (1) groups with ranks of the same color ID but different rank IDs
                (called a "monocolor" group), and

            (2) groups with ranks of the same rank ID but different color IDs
                (called a "multicolor" group).

        Args:
            group: A unique group ID to which will be assigned an integer
                color ID ranging from 0 to the number of group ID's
                supplied across all ranks

        Returns:
            A tuple containing (first) the "monocolor" SimpleComm for ranks
            with the same color ID (but different rank IDs) and (second) the
            "multicolor" SimpleComm for ranks with the same rank ID (but
            different color IDs)

        Raises:
            RuntimeError: If executed during a serial or 1-rank parallel run
        '''
        if self.get_size() > 1:
            allgroups = list(set(self._comm.allgather(group)))
            color = allgroups.index(group)
            monocomm = SimpleCommMPI()
            monocomm._color = color
            monocomm._group = group
            monocomm._comm = self._comm.Split(color)

            rank = monocomm.get_rank()
            multicomm = SimpleCommMPI()
            multicomm._color = rank
            multicomm._group = rank
            multicomm._comm = self._comm.Split(rank)

            return monocomm, multicomm
        else:
            err_msg = 'Division cannot be done on a 1-rank communicator'
            raise RuntimeError(err_msg)
