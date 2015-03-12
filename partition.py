'''
Data Partitioning Functions

This provides a collection of 'partitioning' functions.  A partitioning
function is a three-argument function that takes, as the first argument, a
given data object and, as the second argument, an index into that object and,
as the third argument, a maximum index.  The operation of the partitioning
function is to return a subset of the data corresponding to the given index.

By design, partitioning functions should keep the data "unchanged" except for
subselecting parts of the data.
_______________________________________________________________________________
Created on Feb 4, 2015

Author: Kevin Paul <kpaul@ucar.edu>
'''

from abc import ABCMeta, abstractmethod
from operator import itemgetter


#==============================================================================
# PartitionFunction -
# Base class for all partitioning functions
#==============================================================================
class PartitionFunction(object):

    '''
    The abstract base-class for all Partitioning Function objects.

    A PartitionFunction object is one with a '__call__' method that takes
    three arguments: the data to be partitioned, the index of the partition
    (or part) requested, and the number of partitions to assume when dividing
    the data.
    '''
    __metaclass__ = ABCMeta

    @staticmethod
    def _interface(*args, **kwargs):
        '''
        Define the common interface for PartitionFunction objects
        '''

        # Parse the positional arguments
        if len(args) == 0:
            err_msg = 'PartitionFunction objects take at least 1 positional ' \
                      'argument'
            raise TypeError(err_msg)
        elif len(args) > 3:
            err_msg = 'PartitionFunction objects take at most 3 positional ' \
                      'arguments'
            raise TypeError(err_msg)

        data = args[0]
        index = args[1] if len(args) > 1 else kwargs.pop('index', 0)
        size = args[2] if len(args) > 2 else kwargs.pop('size', 1)

        # If there are any keyword args left, list them as unrecognized
        if len(kwargs) > 0:
            err_msg = 'Unrecognized keyword arguments to PartitionFunction: '
            unrecognized_keys = ', '.join([str(k) for k in kwargs.keys()])
            err_msg += unrecognized_keys
            raise ValueError(err_msg)

        # Check the type of the index
        if type(index) is not int:
            raise TypeError('Partition index must be an integer')

        # Check the value of index
        if index > size - 1 or index < 0:
            raise IndexError('Partition index out of bounds')

        # Check the type of the size
        if type(size) is not int:
            raise TypeError('Partition size must be an integer')

        # Check the value of size
        if size < 1:
            raise TypeError('Partition size less than 1 is invalid')

        # Return the data, index, and size
        return data, index, size

    @staticmethod
    def _is_indexable(data):
        '''
        Check if the data object is indexable
        '''
        if hasattr(data, '__len__') and hasattr(data, '__getitem__'):
            return True
        else:
            return False

    @staticmethod
    def _are_pairs(data):
        '''
        Check if the data object is an indexable list of pairs
        '''
        if PartitionFunction._is_indexable(data):
            return all(map(lambda i: PartitionFunction._is_indexable(i)
                           and len(i) == 2, data))
        else:
            return False

    @abstractmethod
    def __call__(self, *args, **kwargs):
        '''
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Args:

            data:   The data to be partitioned

        Kwargs:

            index:  A partition index into a part of the data

            size:   The largest number of partitions allowed
        '''
        return self._interface(*args, **kwargs)


#==============================================================================
# Duplicate Partitioning Function -
# Grab parts of a list-like object with equal lengths
#==============================================================================
class Duplicate(PartitionFunction):

    '''
    Return a copy of the original input data in each partition
    '''

    def __call__(self, *args, **kwargs):
        '''
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Args:

            data:   The data to be partitioned

        Kwargs:

            index:  A partition index into a part of the data

            size:   The largest number of partitions allowed
        '''
        data, dummy1, dummy2 = super(Duplicate, self).__call__(*args, **kwargs)
        return data


#==============================================================================
# EqualLength Partitioning Function -
# Grab parts of a list-like object with equal lengths
#==============================================================================
class EqualLength(PartitionFunction):

    '''
    Partition an indexable object into sublists of equal (or roughly equal)
    length by returning sections of the list.

    If the partition size is greater than the length of the input data, then
    it will return an empty list for 'empty' partitions.  If the data is not
    indexable, then it will return the data for index=0 only, and an empty
    list otherwise.
    '''

    def __call__(self, *args, **kwargs):
        '''
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Args:

            data:   The data to be partitioned

        Kwargs:

            index:  A partition index into a part of the data

            size:   The largest number of partitions allowed
        '''
        data, index, size = super(EqualLength, self).__call__(*args, **kwargs)
        if self._is_indexable(data):
            (lenpart, remdata) = divmod(len(data), size)
            psizes = [lenpart] * size
            for i in xrange(remdata):
                psizes[i] += 1
            ibeg = 0
            for i in xrange(index):
                ibeg += psizes[i]
            iend = ibeg + psizes[index]
            return data[ibeg:iend]
        else:
            if index == 0:
                return [data]
            else:
                return []


#==============================================================================
# EqualStride Partitioning Function -
# Grab parts of a list-like object with equal lengths
#==============================================================================
class EqualStride(PartitionFunction):

    '''
    Partition an indexable object into sublists of equal (or roughly equal)
    length by striding through the data with a fixed stride.

    If the partition size is greater than the length of the input data, then
    it will return an empty list for 'empty' partitions.  If the data is not
    indexable, then it will return the data for index=0 only, and an empty
    list otherwise.
    '''

    def __call__(self, *args, **kwargs):
        '''
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Args:

            data:   The data to be partitioned

        Kwargs:

            index:  A partition index into a part of the data

            size:   The largest number of partitions allowed
        '''
        data, index, size = super(EqualStride, self).__call__(*args, **kwargs)
        if self._is_indexable(data):
            if index < len(data):
                return data[index::size]
            else:
                return []
        else:
            if index == 0:
                return [data]
            else:
                return []


#==============================================================================
# SortedStride PartitionFunction -
# Grab parts of an indexable object with equal length  after sorting by weights
#==============================================================================
class SortedStride(PartitionFunction):

    '''
    Partition an indexable list of pairs.  The first index of each pair is
    assumed to be an item of data (which will be partitioned), and the second
    index in each pair is assumed to be a numeric weight.  The pairs are
    first sorted by weight, and then partitions are returned by striding
    through the sorted data.

    The results are partitions of roughly equal length and roughly equal
    total weight.  Equal length is prioritized over total weight.
    '''

    def __call__(self, *args, **kwargs):
        '''
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Args:

            data:   The data to be partitioned

        Kwargs:

            index:  A partition index into a part of the data

            size:   The largest number of partitions allowed
        '''
        data, index, size = super(SortedStride, self).__call__(*args, **kwargs)
        if self._are_pairs(data):
            subdata = [q[0] for q in sorted(data, key=itemgetter(1))]
            return EqualStride()(subdata, index=index, size=size)
        else:
            return EqualStride()(data, index=index, size=size)


#==============================================================================
# WeightBalanced PartitionFunction -
# Grab parts of an indexable object that have equal (or roughly equal)
# total weight, though not necessarily equal length
#==============================================================================
class WeightBalanced(PartitionFunction):

    '''
    Partition an indexable list of pairs.  The first index of each pair is
    assumed to be an item of data (which will be partitioned), and the second
    index in each pair is assumed to be a numeric weight.

    The results are partitions of roughly equal length and roughly equal
    total weight.  Equal total weight is prioritized over length.

    '''

    def __call__(self, *args, **kwargs):
        '''
        Define the common interface for all partitioning functions.

        The abstract base class implements the check on the input for correct
        format and typing.

        Args:

            data:   The data to be partitioned

        Kwargs:

            index:  A partition index into a part of the data

            size:   The largest number of partitions allowed
        '''
        data, index, size = super(WeightBalanced,
                                  self).__call__(*args, **kwargs)
        if self._are_pairs(data):
            sorted_pairs = sorted(data, key=itemgetter(1), reverse=True)
            partition = []
            weights = [0] * size
            for (item, weight) in sorted_pairs:
                k = min(enumerate(weights), key=itemgetter(1))[0]
                if k == index:
                    partition.append(item)
                weights[k] += weight
            return partition
        else:
            return EqualStride()(data, index=index, size=size)
