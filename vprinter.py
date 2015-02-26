'''
VPrinter - A module for erbosity-enabled printing

This module implements a VPrinter class that enables clean printing to
standard out (or a string) in a verbosity-level print management.

_______________________________________________________________________________
Created on Feb 26, 2015

@author: Kevin Paul <kpaul@ucar.edu>
'''


class VPrinter(object):

    '''
    A Verbosity-enabled Printing Class

    The VPrinter is designed to print messages to standard out, or optionally
    a string, as determined by a pre-set verbosity-level and/or on which
    parallel rank the VPrinter is instantiated.
    '''

    def __init__(self, header='', verbosity=1):
        '''
        Constructor - Creates an instance of a VPrinter object

        Kwargs:
            rank:       The parallel rank on which this VPrinter is 
                        instantiated.
            size:       The size of the parallel job (total number of ranks)
            verbosity:  The verbosity level to use when determining if a
                        message should be printed to standard out
            is_root:    A bool indicating whether this parallel rank is the
                        root rank, on which special messages will be printed
        '''
        ## The message header to prepend to messages if desired
        self.header = header

        ## The verbosity level for determining if a message is printed
        self.verbosity = verbosity

    def to_str(self, *args, **kwargs):
        '''
        Concatenates string representations of the input arguments

        This takes a list of arguments of any length, converts each argument
        to a string representation, and concatenates them into a single string.

        Args:
            *args:  A list of arguments supplied to the function.  All of these
                    arguments will be concatenated together.

        Kwargs:
            header: A bool that will print a message header before the
                    string with the parallel rank and size indicated
                    [Default: False]

        Returns:
            A single string with the arguments given converted to strings and
            concatenated together (in order).  If 'header==True', then prepend
            to the string a '[rank/size] ' header.

        Raises:
            TypeError, if the header kwarg is not a bool
        '''
        out_args = []
        if 'header' in kwargs:
            if type(kwargs['header']) is bool:
                if kwargs['header']:
                    out_args.append(self.header)
            else:
                raise TypeError('Header keyword argument not bool')
        out_args.extend(args)

        return ''.join([str(arg) for arg in out_args])

    def __call__(self, *args, **kwargs):
        '''
        Print the supplied arguments to standard out

        Prints all supplied positional arguments to standard output, if the
        message verbosity is less than the VPrinter's verbosity level.  Can
        also print a useful header based on the parallel rank and size.

        Args:
            *args:      A list of arguments supplied to the function.  All of
                        these arguments will be concatenated together.

        Kwargs:
            header:     A bool that will print a message header before the
                        string with the parallel rank and size indicated
                        [Default: False]
            verbosity:  The integer verbosity level assigned to this message.
                        [Default: 0]
        '''
        verbosity = 0
        if 'verbosity' in kwargs and type(kwargs['verbosity']) is int:
            verbosity = kwargs['verbosity']

        if verbosity < self.verbosity:
            print self.to_str(*args, **kwargs)
