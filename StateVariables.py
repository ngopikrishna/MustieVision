#!/usr/bin/env python3

"""
@author: N. Gopi Krishna
Class for maintaining state variables
"""


class VideoProcessingState:
    """
        Module for maintaining state management variable
    """
    def __init__(self):
        """ ***** """
        self.__stop_processing = True

    def set_stop(self, value):
        """ Set the value"""
        self.__stop_processing = value

    def get_stop(self):
        """ get the value """
        return self.__stop_processing

    # Specify them as properties
    stop = property(get_stop, set_stop)
