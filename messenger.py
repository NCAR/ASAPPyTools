'''
This is the parallel/messenging tool to simplify and streamline some of the
common MPI messaging tasks needed by many of our in-house tools in the
NCAR Application Scalability and Performance group (ASAP).  The Messenger
class is designed to provide simplified functionality to/from the MPI
communicators, without completely wrapping the entire MPI API.  If more
functionality is needed than is provided currently, one should seriously
consider directly using the mpi4py interface itself.

If, however, a simplified interface to parallel messaging is all that is
needed, and you want to hide the MPI interface completely, then the
Messenger may be the class for you.  It is designed to work the same way
in serial as in parallel, and there is provided a 'create_messenger'
factory function for creating the kind of Messenger desired (i.e., serial
or parallel).

One of the main advantages of the Messenger class is that it works regardless
of whether mpi4py is installed on your system.  In other words, the interface
is the same for the serial operation or the parallel operation.

_______________________
Created on Apr 30, 2014
Last modified on Jan 6, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''

import sys
from functools import partial

#==============================================================================
# Define mapped "reduce" operations, mapping going from MPI name (lowercase) to
# Python function name.  If a reduce operation is given that is not in this
# mapping, then assume the names are the same (all lowercase for Python and
# all uppercase for MPI Op name)
#==============================================================================
_MAPPED_OPS = {'prod':partial(reduce, lambda x, y: x * y),
               'sum' :partial(reduce, lambda x, y: x + y)}

#==============================================================================
# create_messenger factory function
#==============================================================================
def create_messenger(serial=False):
    '''
    This is the factory function the creates the necessary decomposition
    object for parallel (or serial) operation.  The type must be specified
    by the user with the 'serial' argument, as there is not portable way
    of determining if the run should be assumed to be serial or parallel
    from the environment.

    @param serial  True or False, indicating whether the serial or parallel
                   decomposition utility object should be constructed and
                   returned.  DEFAULT: False (parallel operation)

    @return  A decomposition utility object.
    '''
    # Check type
    if (type(serial) is not bool):
        err_msg = "The serial argument must be a bool."
        raise TypeError(err_msg)

    # Construct and return the desired decomp utility
    if (serial):
        return Messenger()
    else:
        return MPIMessenger()


#==============================================================================
# Messenger Base Class
#==============================================================================
class Messenger(object):
    '''
    This is the base class for decomposition/parallel utilities.  This defines
    serial operation, and has no dependencies.  These methods are reimplemented
    in the derived class for parallel decomposition.
    '''

    def __init__(self):
        '''
        Constructor
        '''

        ## The type of decomp utility constructed
        self.messenger_type = 'serial'

        ## Whether this is the master process/rank
        self._is_master = True

        ## MPI namespace for common classes and methods
        self._mpi = None

        ## The rank of the processor
        self._mpi_rank = 0

        ## Size of the MPI communicator
        self._mpi_size = 1

        ## Pointer to the MPI communicator itself
        self._mpi_comm = None

        ## Reference to the Messenger parent
        self._parent = None

        ## Group Identifier (color)
        self._color = None

        ## Indicates verbosity level
        self.verbosity = 1

    def split(self, unused):
        '''
        Returns a new Messenger instance that carries messages only between
        the ranks with the same color.  In serial, this returns None.

        @param unused  An identifier (e.g., int) for this rank's group

        @return A new Messenger instance that communicates among ranks within
                the same group.
        '''
        return None

    def partition(self, global_items):
        '''
        Given a container of items, this algorithm selects a subset of
        items to be used by the local process.  If the 'global_items' object
        is a dictionary, then the values of the dictionary are treated as
        weights for the partitioning algorithm, attempting to give each
        process an equal weight.  If the global_items is a list, then all
        items are considered equally weighted.  For serial execution, this
        just returns the global list.

        @param global_items  If this is a list, then the weights of the items
                             are considered unity.  If this is a dictionary,
                             then the values in the dictionary are considered
                             weights.

        @return A list containing a subset of the global items to be used
                on the current (local) processor
        '''
        if isinstance(global_items, list):
            return global_items
        elif isinstance(global_items, dict):
            return global_items.keys()
        else:
            err_msg = 'Global items object has unrecognized type (' \
                    + str(type(global_items)) + ')'
            raise TypeError(err_msg)

    def sync(self):
        '''
        A wrapper on the MPI Barrier method.  Forces execution to wait for
        all other processors to get to this point in the code.  Does nothing
        in serial.
        '''
        return

    def is_master(self):
        '''
        Returns True or False depending on whether this rank is the master
        rank (i.e., rank 0).  In serial, always returns True.

        @return True or False depending on whether this rank is the master
                rank (i.e., rank 0).
        '''
        return self._is_master

    def get_rank(self):
        '''
        Returns the integer associated with the local MPI processor/rank.
        In serial, it always returns 0.

        @return The integer ID associated with the local MPI processor/rank
        '''
        return self._mpi_rank

    def get_size(self):
        '''
        Returns the number of ranks in the MPI communicator.
        In serial, it always returns 1.

        @return The integer number of ranks in the MPI communicator
        '''
        return self._mpi_size

    def get_color(self):
        '''
        Returns the color of the rank in the (presumably split) MPI
        communicator.

        @return The color of this rank
        '''
        return self._color

    def reduce(self, data, op='sum'):
        '''
        This reduces data across all processors, returning the result of the
        given reduction operation.

        If the data is a dictionary (with assumed numberic values), then this
        reduces the values associated with each key across all processors.  It
        returns a dictionary with the reduction for each key.  Every processor
        must have the same keys in their associated dictionaries.

        If the data is not a dictionary, but is list-like (i.e., iterable),
        then this returns a single value with the reduction of all values
        across all processors.

        If the data is not iterable (i.e., a single value), then it reduces
        across all processors and returns one value.

        In serial, this returns just the reduction on the local processor.

        @note: This is an "allreduce" operation, so the final reduction is
               valid on all ranks.

        @param data The data with values to be summed

        @return The sum of the data values
        '''
        if type(op) is not str:
            err_msg = 'Reduce operator name must be a string'
            raise TypeError(err_msg)

        if (isinstance(data, dict)):
            totals = {}
            for name in data:
                totals[name] = self.reduce(data[name], op=op)
            return totals
        elif (hasattr(data, '__len__')):
            if op in _MAPPED_OPS:
                return _MAPPED_OPS[op](data)
            else:
                return eval(op.lower())(data)
        else:
            return data

    def gather(self, data):
        '''
        Implements a gather operation, where data is send from all ranks
        in the Messenger's domain to the master rank.

        @param  data  The data to be gathered on the master rank

        @return  The accumulated data (on the master rank), or None (on
                 all other ranks).
        '''
        # In serial, just return the data unchanged
        return [data]

    def scatter(self, data):
        '''
        Implements a scatter operation, where data is send from the master rank
        in the Messenger's domain to all of the subordinate ranks.

        @param  data  The data to be scattered from the master rank

        @return  The scattered data
        '''
        return data[0]

    def broadcast(self, data):
        '''
        Implements a broadcast operation, where data is send from the master
        rank in the Messenger's domain to all of the subordinate ranks.

        @param  data  The data to be scattered from the master rank

        @return  The scattered data
        '''
        return data

    def sendrecv(self, data, source=None, dest=None):
        '''
        Implements a point-to-point communication, sending data from 'orig'
        to the 'dest' rank.

        @param  data  The data to be sent (only from 'orig' rank)

        @param  orig  The origin rank to send the data (Defaults to

        @param  dest  The destination rank to received the data

        @return  The received data (only on 'dest' rank)
        '''
        return data

    def prinfo(self, output, vlevel=0, master=True):
        '''
        Short for "print info", this method prints output to stdout, but only
        if the "verbosity level" (vlevel) is less than the Messenger's
        defined verbosity.  If the "master" parameter is True, then the
        message is printed from only the master rank.  Otherwise, the
        message is printed from every rank in the Messenger's domain.

        @param output The thing that should be printed to stdout.  (It is
                      converted to a string before printing.)

        @param vlevel The verbosity level associated with the message.  If
                      this level is less than the messenger's verbosity, no
                      output is generated.  The default is 0.

        @param master True or False, indicating if the message should be
                      printed from all ranks (True) or only from the master
                      rank (False).  The default is False.
        '''
        if (vlevel < self.verbosity):
            if (master and self.is_master()):
                print str(output)
                sys.stdout.flush()
            elif (not master):
                print '[' + str(self._mpi_rank) + '/' + str(self._mpi_size) \
                    + ']: ' + str(output)
                sys.stdout.flush()


#==============================================================================
# MPIMessenger Class
#==============================================================================
class MPIMessenger(Messenger):
    '''
    This is the parallel-operation class for decomposition/parallel utilities.
    This is derived from the Messenger class, which defines the serial
    operation.  This defines basic operations using MPI.
    '''

    def __init__(self):
        '''
        Constructor

        @param comm  This is an optional parameter to associate the Messenger
                     with an already known MPI communicator (intracomm).
                     If None (not specified), the messenger will be associated
                     with the COMM_WORLD, by default.
        '''

        # Call the parent class initialization first
        super(MPIMessenger, self).__init__()

        ## Type of decomp utility constructed
        self.messenger_type = 'parallel'

        # Try to import the MPI module
        try:
            from mpi4py import MPI
        except:
            raise ImportError('Failed to import MPI.')

        ## MPI namespace for common classes and methods
        self._mpi = MPI

        ## Pointer to the MPI module (defaults to COMM_WORLD)
        self._mpi_comm = MPI.COMM_WORLD

        ## The rank of the processor
        self._mpi_rank = self._mpi_comm.Get_rank()

        ## MPI Communicator size
        self._mpi_size = self._mpi_comm.Get_size()

        ## Whether this is the master process/rank
        self._is_master = (self._mpi_rank == 0)

    def split(self, color):
        '''
        Returns a new Messenger instance that carries messages only between
        the ranks with the same color.  In serial, this returns None.

        @param color  An identifier (e.g., int) for this rank

        @return A new Messenger instance that communicates among ranks with
                the same color.
        '''
        # Note, this is essentially a constructor...
        newcomm = self._mpi_comm.Split(color, self._mpi_rank)
        newmsgr = MPIMessenger()
        newmsgr._mpi_comm = newcomm
        newmsgr._mpi_rank = newmsgr._mpi_comm.Get_rank()
        newmsgr._mpi_size = newmsgr._mpi_comm.Get_size()
        newmsgr._is_master = (newmsgr._mpi_rank == 0)
        newmsgr._parent = self
        newmsgr._color = color
        return newmsgr

    def partition(self, global_items):
        '''
        Given a container of items, this algorithm selects a subset of
        items to be used by the local process.  If the 'global_items' object
        is a dictionary, then the values of the dictionary are treated as
        weights for the partitioning algorithm, attempting to give each
        process an equal weight.  If the global_items is a list, then all
        items are considered equally weighted.  For serial execution, this
        just returns the global list.

        @param global_items  If this is a list, then the weights of the items
                             are considered unity.  If this is a dictionary,
                             then the values in the dictionary are considered
                             weights.

        @return A list containing a subset of the global items to be used
                on the current (local) processor
        '''
        if isinstance(global_items, list):
            global_dict = dict((name, 1) for name in global_items)
        elif isinstance(global_items, dict):
            global_dict = global_items
        else:
            err_msg = 'Global items object has unrecognized type (' \
                    + str(type(global_items)) + ')'
            raise TypeError(err_msg)

        # Sort the names of the variables by their weight
        global_list = list(zip(*sorted(global_dict.items(), key=lambda p: p[1]))[0])

        # KMP: A better partitioning algorithm should be implemented.  The
        #      above line with the striding below does not necessarily load
        #      balance as well as could be done.  It is easy, though...

        # Return a subset of the list by striding though the list
        return global_list[self._mpi_rank::self._mpi_size]

    def sync(self):
        '''
        A wrapper on the MPI Barrier method.  Forces execution to wait for
        all other processors to get to this point in the code.
        '''
        self._mpi_comm.Barrier()

    def reduce(self, data, op='sum'):
        '''
        This reduces data across all processors, returning the result of the
        given reduction operation.

        If the data is a dictionary (with assumed numberic values), then this
        reduces the values associated with each key across all processors.  It
        returns a dictionary with the reduction for each key.  Every processor
        must have the same keys in their associated dictionaries.

        If the data is not a dictionary, but is list-like (i.e., iterable),
        then this returns a single value with the reduction of all values
        across all processors.

        If the data is not iterable (i.e., a single value), then it reduces
        across all processors and returns one value.

        In serial, this returns just the reduction on the local processor.

        @note: This is an "allreduce" operation, so the final reduction is
               valid on all ranks.

        @param data The data with values to be summed

        @return The sum of the data values
        '''
        if type(op) is not str:
            err_msg = 'Reduce operator name must be a string'
            raise TypeError(err_msg)

        if (isinstance(data, dict)):
            totals = {}
            for name in data:
                totals[name] = self.reduce(data[name], op=op)
            return totals
        elif (hasattr(data, '__len__')):
            total = Messenger.reduce(self, data, op=op)
            return self.reduce(total, op=op)
        else:
            return self._mpi_comm.allreduce(data, op=getattr(self._mpi, op.upper()))

    def gather(self, data):
        '''
        Implements a gather operation, where data is send from all ranks
        in the Messenger's domain to the master rank.

        @param  data  The data to be gathered on the master rank

        @return  The accumulated data (on the master rank), or None (on
                 all other ranks).
        '''
        alldata = self._mpi_comm.gather(data)
        return alldata

    def scatter(self, data):
        '''
        Implements a scatter operation, where data is send from the master rank
        in the Messenger's domain to all of the subordinate ranks.

        @param  data  The data to be scattered from the master rank.  The nth
                      item in data will be sent to rank n, and data must have
                      length equal to the size of the Messenger's domain.

        @return  The appropriate subset of the scattered data
                 (i.e., the nth element of data on rank n)
        '''
        subdata = self._mpi_comm.scatter(data)
        return subdata

    def broadcast(self, data):
        '''
        Implements a broadcast operation, where data is send from the master
        rank in the Messenger's domain to all of the subordinate ranks.

        @param  data  The data to be scattered from the master rank

        @return  The scattered data
        '''
        newdata = self._mpi_comm.bcast(data)
        return newdata

    def sendrecv(self, data, source=0, dest=0):
        '''
        Implements a point-to-point communication, sending data from 'source'
        to the 'dest' rank.  This communication uses a hand-shake.

        @param  data  The data to be sent (only from 'source' rank)

        @param  source  The origin rank to send the data (Defaults to 0, or
                      the master rank).

        @param  dest  The destination rank to receive the data (Defaults to
                      0, or the master rank).

        @return  The received data (only on 'dest' rank, None on others)
        '''
        recvd = None
        tag = 3 * source + 1
        if self._mpi_rank == source:
            self._mpi_comm.send(data, dest=dest, tag=tag)
        if self._mpi_rank == dest:
            recvd = self._mpi_comm.recv(source=source, tag=tag)
        return recvd
