#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:04:33 2018

@author: N. Gopi Krishna
"""

import threading
import pandas as pd
import cv2
import dlib
import math
import uuid

import FaceDetector as fd
import FaceRecognizer as frec


class ProcessVideo(threading.Thread):
    """
        Main module for processing the video and recognising the faces.
        Creates it's own thread and runs in it
    """
    # Also takes responsibility of showing the images. This can probably be changed but that is for later

    __KNOWN_FACE_BOUNDING_BOX_COLOR = (255, 255, 255)
    __DOUBTFUL_FACE_BOUNDING_BOX_COLOR = (0, 0, 255)
    __KNOWN_FACE_FONT = cv2.FONT_HERSHEY_PLAIN
    __DOUBTFUL_FACE_FONT = __KNOWN_FACE_FONT

    def __init__(self,
                 image_buffer,
                 state_variable_stop_processing,
                 shape_predictor_68_datfile_location,
                 torch_neuralnetwork_model_location,
                 classifiermodel_picklefile_location):
        """ Constructor """
        threading.Thread.__init__(self)
        self.__image_buffer = image_buffer
        self.__state_varibale_stop_processing = state_variable_stop_processing
        self.__face_detector = fd.FaceDetector(shape_predictor_68_datfile_location, torch_neuralnetwork_model_location)
        self.__face_recognizer = frec.FaceRecognizer(classifiermodel_picklefile_location)


        self.__Windowname = "Mustie vision"

    def run(self):
        """ Entry point for the thread"""
        self.__process_video_stream_and_show_on_screen()


    def __draw_bounding_box(self, one_frame, face_id, left, top, right, bottom, confidence=1):
        left = int(math.floor(left))
        top = int(math.floor(top))
        right = int(math.floor(right))
        bottom = int(math.floor(bottom))


        top_left = (left, top)
        bottom_right = (right, bottom)
        bottom_left = (left, bottom)

        if confidence > 0.5:
            face_bounding_box_color = ProcessVideo.__KNOWN_FACE_BOUNDING_BOX_COLOR
        else:
            face_bounding_box_color = ProcessVideo.__DOUBTFUL_FACE_BOUNDING_BOX_COLOR

        face_id_font = ProcessVideo.__KNOWN_FACE_FONT
        cv2.rectangle(one_frame, top_left, bottom_right, face_bounding_box_color)
        cv2.putText(one_frame, face_id, bottom_left, face_id_font, 1, face_bounding_box_color, 1)

    def __process_one_frame_for_face_recognition(self, one_frame):
        """
            Process the input cv::Mat
                1) Check for the presence of faces i.e. detect faces
                2) Recognize the faces.
                3) Set bounding box for recognized faces.
                4) Create trackers for each recognized face and return them
        """

        rgbImage = cv2.cvtColor(one_frame, cv2.COLOR_BGR2RGB)
        face_reps = self.__face_detector.detect_faces(rgbImage)

        # Create the face_position_trackers as dictionary.
        # Face ID shall be the key and tracker object shall be the value
        face_position_trackers_in_one_frame = {}

        if face_reps is None:
            # There are no faces in the frame. Do nothing
            face_position_trackers_in_one_frame.clear()
        else:
            for one_face in face_reps:
                (face_id, confidence) = self.__face_recognizer.recognize_face(one_face['Landmarks'])

                if face_id is None:
                    continue

                bounding_box = one_face['BoundingBox']

                left = bounding_box.left()
                top = bounding_box.top()
                right = bounding_box.right()
                bottom = bounding_box.bottom()
                self.__draw_bounding_box(one_frame, face_id, left, top, right, bottom, confidence)

                # Create a tracker
                one_face_tracker = dlib.correlation_tracker()
                one_face_tracker.start_track(one_frame, dlib.rectangle(left, top, right, bottom))

                face_position_trackers_in_one_frame[face_id] = one_face_tracker
            #end for
        #end if
        return face_position_trackers_in_one_frame


    def __process_video_stream_and_show_on_screen(self):
        """
            The main work horse which processes the video
        """

        try:

            # Create windows for each camera
            cv2.namedWindow(self.__Windowname, cv2.WINDOW_AUTOSIZE)

            # Process cameras
            while not self.__state_varibale_stop_processing.stop:
                if self.__image_buffer.qsize() == 0:
                    continue

                im = None
                im = self.__image_buffer.get()

                # Process each coordinated set of cameras and get the face tracker for
                # each of face in the camera
                one_set_of_face_position_trackers = self.__process_one_frame_for_face_recognition(im)
                cv2.imshow(self.__Windowname, im)

                # All windows are updated. Pause for a microsecond to allow the user to view the window
                cv2.waitKey(1)

            print("Closing windows")
            cv2.destroyWindow(self.__Windowname)
        except Exception as e:
            print("Exception {} occurred in ProcessVideo::process().".format(e))
            raise e
        return
