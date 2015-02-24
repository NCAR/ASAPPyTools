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

@author: Kevin Paul <kpaul@ucar.edu>
'''

from operator import itemgetter


#==============================================================================
# PartitioningFunction -
# Base class for all partitioning functions
#==============================================================================
class PartitioningFunction(object):
    '''
    The base-class for all Partitioning Function objects.

    A PartitioningFunction object is one with a '__call__' method that takes
    three arguments: the data to be partitioned, the index of the partition
    (or part) requested, and the number of partitions to assume when dividing
    the data.
    '''

    def __init__(self):
        '''
        Constructor
        '''

    @staticmethod
    def _ifc(data, index=0, size=1):
        '''
        Define the common interface for all partitioning functions.  Checks the
        input for correct format and returns the input if everything is
        correct.

        @param  data  The data to be partitioned

        @param  index  A partition index into a part of the data

        @param  size  The largest number of partitions allowed

        @return  If correct, it returns the input
        '''
        if type(index) is not int:
            raise TypeError('Partition index must be an integer')
        if type(size) is not int:
            raise TypeError('Partition size must be an integer')
        if size < 1:
            raise TypeError('Partition size less than 1 is invalid')
        if index > size - 1 or index < 0:
            raise IndexError('Partition index out of bounds')
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
        if PartitioningFunction._is_indexable(data):
            return all(map(lambda i: PartitioningFunction._is_indexable(i)
                           and len(i) == 2, data))
        else:
            return False

    def __call__(self, *args, **kwargs):
        '''
        Partition the data given
        '''
        return PartitioningFunction._ifc(*args, **kwargs)[0]


#==============================================================================
# EqualStride Partitioning Function -
# Grab parts of a list-like object with equal lengths
#==============================================================================
class EqualStride(PartitioningFunction):
    '''
    Partition an indexable object into sublists of equal (or roughly equal)
    length by striding through the data with a fixed stride.

    If the partition size is greater than the length of the input data, then
    it will return an empty list for 'empty' partitions.  If the data is not
    indexable, then it will return the data for index=0 only, and an empty
    list otherwise.
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def __call__(self, *args, **kwargs):
        data, index, size = PartitioningFunction._ifc(*args, **kwargs)
        if PartitioningFunction._is_indexable(data):
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
# SortedStride PartitioningFunction -
# Grab parts of an indexable object with equal length  after sorting by weights
#==============================================================================
class SortedStride(PartitioningFunction):
    '''
    Partition an indexable list of pairs.  The first index of each pair is
    assumed to be an item of data (which will be partitioned), and the second
    index in each pair is assumed to be a numeric weight.  The pairs are
    first sorted by weight, and then partitions are returned by striding
    through the sorted data.

    The results are partitions of roughly equal length and roughly equal
    total weight.  Equal length is prioritized over total weight.
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def __call__(self, *args, **kwargs):
        '''
        Partition the data given
        '''
        data, index, size = PartitioningFunction._ifc(*args, **kwargs)
        if PartitioningFunction._are_pairs(data):
            subdata = [q[0] for q in sorted(data, key=itemgetter(1))]
            return EqualStride(subdata, index=index, size=size)
        else:
            return EqualStride(data, index=index, size=size)


#==============================================================================
# WeightBalanced PartitioningFunction -
# Grab parts of an indexable object that have equal (or roughly equal)
# total weight, though not necessarily equal length
#==============================================================================
class WeightBalanced(PartitioningFunction):
    '''
    Partition an indexable list of pairs.  The first index of each pair is
    assumed to be an item of data (which will be partitioned), and the second
    index in each pair is assumed to be a numeric weight.

    The results are partitions of roughly equal length and roughly equal
    total weight.  Equal total weight is prioritized over length.

    '''

    def __init__(self):
        '''
        Constructor
        '''

    def __call__(self, *args, **kwargs):
        '''
        Partition the data given
        '''
        data, index, size = PartitioningFunction._ifc(*args, **kwargs)
        if PartitioningFunction._are_pairs(data):
            sorted_pairs = sorted(data, key=itemgetter(1), reverse=True)
            partition = []
            weights = [0] * size
            for (item, weight) in sorted_pairs:
                k = min(enumerate(weights), key=itemgetter(1))[0]
                if k == index:
                    partition.append(item)
                weights[k] += weight
            print weights
            return partition
        else:
            return EqualStride(*args, **kwargs)
