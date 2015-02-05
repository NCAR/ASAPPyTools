'''
Partitioner - Data Partitioning Functions

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


#==============================================================================
# unity - Leave data unchanged
#==============================================================================
def unity(data, index=0, size=1):
    '''
    The unity partitioning function returns the data unchanged.

    @param  data  The data to be partitioned

    @param  index  An index into a part of the data

    @param  size  The largest number of partitions allowed

    @return  The partition of the data corresponding to the given index
    '''
    return data


#==============================================================================
# equal_length - Grab parts of a list-like object with equal lengths
#==============================================================================
def equal_length(data, index=0, size=1):
    '''
    Partition a list-like object into sublists of equal
    (or roughly equal) length.

    @param  data  The data to be partitioned

    @param  index  An index into a part of the data

    @param  size  The largest number of partitions allowed

    @return  The partition of the data corresponding to the given index
    '''
    if hasattr(data, '__getitem__') and hasattr(data, '__len__'):
        return data[index % len(data)::size]
    else:
        return data


#==============================================================================
# weighted_equal_length - Grab parts of a list-like object with equal length
#                         and distributed total weights
#==============================================================================
def weighted_equal_length(data, index=0, size=1):
    '''
    Partition a pair of equal-length list-like objects --- with the first
    of the list-like objects containing the data and the second of the
    list-like objects containing corresponding weights --- into sublists of
    the data with roughly equal length and evenly distributed total weight.

    @param  data  The data to be partitioned

    @param  index  An index into a part of the data

    @param  size  The largest number of partitions allowed

    @return  The partition of the data corresponding to the given index
    '''
    if not hasattr(data, '__len__'):
        return data
    if len(data) != 2:
        raise ValueError('Input data must be a pair of objects')
    if len(data[0]) != len(data[1]):
        raise ValueError('Input data pair must have equal lengths')
    subdata = [q[0] for q in sorted(zip(*data), key=lambda p: p[1])]
    return equal_length(subdata, index=index, size=size)


#==============================================================================
# weighted_balanced - Grab parts of a list-like object with equal total weight
#                     (but not, necessarily, equal length)
#==============================================================================
def weighted_balanced(data, index=0, size=1):
    '''
    Partition a pair of equal-length list-like objects --- with the first
    of the list-like objects containing the data and the second of the
    list-like objects containing corresponding weights --- into sublists of
    the data with roughly equal total weight (but not necessarily equal
    lengths).

    @param  data  The data to be partitioned

    @param  index  An index into a part of the data

    @param  size  The largest number of partitions allowed

    @return  The partition of the data corresponding to the given index
    '''
    subdata = [q[0] for q in sorted(zip(*data), key=lambda p: p[1])]
    return equal_length(subdata, index=index, size=size)
