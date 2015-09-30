"""
A module containing the TimeKeeper class.

This module contains is a simple class to act as a time keeper for internal
performance monitoring (namely, timing given processes).

Copyright 2015, University Corporation for Atmospheric Research
See the LICENSE.txt file for details
"""

from time import time
from collections import OrderedDict


class TimeKeeper(object):

    """
    Class to keep timing recordings, start/stop/reset timers.

    Attributes:
        _time: The method to use for getting the time (e.g., time.time)
        _start_times (OrderedDict): A dictionary of start times for each
            named timer
        _accumulated_times (OrderedDict): A dictionary of the total
            accumulated times for each named timer
    """

    def __init__(self, time=time, enabled=True):
        """
        Constructor.

        Parameters:
            time: The function to use for measuring the time.  By default,
                it is the Python 'time.time()' method.
            enabled: Whether to create the TimeKeeper with timing enabled
                (True) or disabled (False) upon construction
        """

        # The method to use for time measurements
        self._time = time

        # Dictionary of start times associated with a string name
        self._start_times = OrderedDict()

        # Dictionary of accumulated times associated with a string name
        self._accumulated_times = OrderedDict()

        # Switch to enable/disable the timers
        self._enabled = enabled

    def enable(self):
        """
        Method to enable the TimeKeeper
        """
        self._enabled = True

    def disable(self):
        """
        Method to disable the TimeKeeper

        All calls to a disabled TimeKeeper will do nothing
        """
        self._enabled = False

    def reset(self, name):
        """
        Method to reset a timer associated with a given name.

        If the name has never been used before, the timer is created and the
        accumulated time is set to 0.  If the timer has been used before, the
        accumulated time is set to 0.

        Parameters:
            name: The name or ID of the timer to reset
        """
        if self._enabled:

            # Reset the named timer (creates it if it doesn't exist yet)
            self._accumulated_times[name] = 0.0
            self._start_times[name] = self._time()

    def start(self, name):
        """
        Method to start a timer associated with a given name.

        If the name has never been used before, the timer is created and
        the accumulated time is set to 0.

        Parameters:
            name: The name or ID of the timer to start
        """
        if self._enabled:

            # Start the named timer (creates it if it doesn't exist yet)
            if name not in self._accumulated_times:
                self.reset(name)
            else:
                self._start_times[name] = self._time()

    def stop(self, name):
        """
        Stop the timing and add the accumulated time to the timer.

        Method to stop a timer associated with a given name, and adds
        the accumulated time to the timer when stopped.  If the given timer
        name has never been used before (either by calling reset() or start()),
        the timer is created and the accumulated time is set to 0.

        Parameters:
            name: The name or ID of the timer to stop
        """
        if self._enabled:

            # Stop the named timer, add to accumulated time
            if name not in self._accumulated_times:
                self.reset(name)
            else:
                self._accumulated_times[name] += \
                    self._time() - self._start_times[name]

    def get_names(self):
        """
        Method to return the clock names in the order in which they were added.

        Returns:
            list: The list of timer names in the order they were added
        """
        if self._enabled:
            return self._accumulated_times.keys()
        else:
            return []

    def get_time(self, name):
        """
        Returns the accumulated time of the given timer.

        If the given timer name has never been created, it is created and the
        accumulated time is set to zero before returning.

        Parameters:
            name: The name or ID of the timer to stop

        Returns:
            float: The accumulated time of the named timer (or 0.0 if the
                named timer has never been created before).
        """
        if self._enabled:
            if name not in self._accumulated_times:
                self.reset(name)
            return self._accumulated_times[name]
        else:
            return 0.0

    def get_all_times(self):
        """
        Returns the dictionary of accumulated times on the local processor.

        Returns:
            OrderedDict: The dictionary of accumulated times
        """
        if self._enabled:
            return self._accumulated_times
        else:
            return OrderedDict()
