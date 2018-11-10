#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:24:54 2018

@author: N. Gopi Krishna
"""

from tkinter import Tk, Frame, Button, LEFT, TOP, DISABLED, NORMAL, BOTTOM, RIGHT

class MustieFaceRecogView:
    """ # MVC Pattern's View is here """


    def __disable_all_buttons(self):
        self.__stop_button.configure(state=DISABLED)
        self.__recognize_button.configure(state=DISABLED)
        self.__train_button.configure(state=DISABLED)


    def __init__(self, MustieFaceRecogController):
        """ constructor """
        self.__controller = MustieFaceRecogController

        # Create the UI elements
        self.__root_window = Tk()
        frame = Frame(self.__root_window)
        frame.pack()

        topframe = Frame(frame)
        topframe.pack(side=TOP)

        bottomframe = Frame(frame)
        bottomframe.pack(side=BOTTOM)

        self.__recognize_button = Button(topframe,
                                         text="Recognize Faces",
                                         command=self.__recognize_button_callback)
        self.__recognize_button.pack(side=LEFT)

        self.__stop_button = Button(topframe,
                                    text="Stop Processing",
                                    command=self.__stop_button_callback,
                                    state=DISABLED)
        self.__stop_button.pack(side=RIGHT)

        self.__train_button = Button(bottomframe,
                                     text="Train new faces",
                                     command=self.__train_button_callback)
        self.__train_button.pack(side=LEFT)

    def __recognize_button_callback(self):
        """ Call back for Recognize button """
        self.__disable_all_buttons()
        self.__stop_button.configure(state=NORMAL)
        self.__controller.recognize_faces()

    def __train_button_callback(self):
        """ Call back for Train button """
        self.__disable_all_buttons()

        self.__controller.train_model()

        self.__recognize_button.configure(state=NORMAL)
        self.__train_button.configure(state=NORMAL)


    def __stop_button_callback(self):
        """ Call back for Stop button """
        self.__disable_all_buttons()
        self.__controller.stop_processing()
        self.__recognize_button.configure(state=NORMAL)
        self.__train_button.configure(state=NORMAL)



    def Launch(self):
        """ The window is created here. Entry point for callers"""
        self.__root_window.mainloop()
