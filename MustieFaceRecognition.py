#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:24:54 2018

@author: N. Gopi Krishna
"""

import os

import MustieFaceRecogController as controller
import MustieFaceRecogView as view

class MustieFaceRecogCreator:
    """
    MVC Pattern's master is here
    Master simply creates the view and controller.
    He informs the view about the controller and then gets out of the way.
    """
    def __init__(self, startup_folder):
        """ Private constructor"""
        # Create the controller and view.
        # Inform view about controller and get out of it's way. View and Controller shall work
        # together.
        self.__dyn_gall_controller = controller.MustieFaceRecogController(startup_folder)
        self.__dyn_gall_view = view.MustieFaceRecogView(self.__dyn_gall_controller)

    def Go(self):
        """ Main Entry point for master"""
        # Launch the view
        self.__dyn_gall_view.Launch()



# This is the main implementation.
if __name__ == '__main__':
    # Set the working directory, system folders, model locations etc.
    # This will be the place to process command line arguments, configuration files etc.

    # Create the DynamicGalleryCreator object and run it
    try:
        cwd = os.getcwd()
        dgc = MustieFaceRecogCreator(cwd)
        dgc.Go()
        print("This is the End")
    except IOError as e:
        print(e)
    except Exception as e:
        print(e)
