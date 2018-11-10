#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:05:21 2018

@author: N. Gopi Krishna
"""

from os import listdir
from os.path import isfile, join
import os
import threading
import cv2

class ImageStreamReader(threading.Thread):
    """
        Module is responsible for reading image streams
    """

    def __init__(self, image_buffer, state_variable_stop_processing, images_source):
        """
        Constructor. if images_source is integer (0,1 etc), reads from Webcam assigned to that integer.
        Reading from a video file is yet to be supported
        """
        threading.Thread.__init__(self)
        self.__state_varibale_stop_processing = state_variable_stop_processing
        self.__images_buffer = image_buffer
        self.__images_source = images_source

    def __safe_clear_buffer(self):
        """
        Clear the buffer. Thread safe.
        """
        with self.__images_buffer.mutex:
            self.__images_buffer.queue.clear()
        return


    def __read_images_from_webcam(self):
        """
            We read from webcam here
        """

        try:
            vcap = cv2.VideoCapture(self.__images_source)
            assert vcap.isOpened()

            self.__safe_clear_buffer()

            while not self.__state_varibale_stop_processing.stop:
                _, im = vcap.read()
                assert im is not None
                self.__images_buffer.put(im)
        except:
            print("ImageStreamReader::exception in __read_images_from_webcam")
            self.__state_varibale_stop_processing.stop = True
            raise #Lets rethrow the exception so that caller knows it

    def run(self):
        """
        Main entry point. Can read video from the source specified in constructor
        """
        if isinstance(self.__images_source, int):
            self.__read_images_from_webcam()
        else:
            raise NotImplementedError()
