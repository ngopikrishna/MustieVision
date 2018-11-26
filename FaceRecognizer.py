#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:07:03 2018

@author: N. Gopi Krishna
"""
import os
import pickle
import warnings     # this is purely to suppress the distracting API deprecation warning messages during run time. distracting
import numpy as np


class FaceRecognizer:
    """
        This module loads the pickle file. Pickle file contains the classification model which was
        created from the list of faces in the training set.
        The model remains in RAM during the duration of computation. Cant avoid it.
    """

    def __init__(self, classifiermodel_picklefile_location):
        """ """
        self.__classifiermodel_picklefile_location = classifiermodel_picklefile_location
        self.__classifiermodel_le = None
        self.__classifiermodel_clf = None
        self.__CONFIDENCE_LEVEL = 0.3

        self.__load_pickle_file()

    def __load_pickle_file(self):
        """ """
        if os.path.exists(self.__classifiermodel_picklefile_location):
            with open(self.__classifiermodel_picklefile_location, 'rb') as f:
                (self.__classifiermodel_le, self.__classifiermodel_clf) = pickle.load(f, encoding='latin1')
                print(self.__classifiermodel_le, self.__classifiermodel_clf)
        else:
            print("*************** Cant locate classifier model files *****************")
            self.__classifiermodel_le = None
            self.__classifiermodel_clf = None




    def recognize_face(self, one_face_landmarks):
        """ main entry point"""
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

        if  ((self.__classifiermodel_le is None) or (self.__classifiermodel_clf is None)):
            self.__load_pickle_file()

        one_face_rep = one_face_landmarks.reshape(1, -1)
        predictions = self.__classifiermodel_clf.predict_proba(one_face_rep).ravel()
        maxI = np.argmax(predictions)
        person = self.__classifiermodel_le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if confidence > self.__CONFIDENCE_LEVEL:
            returnvalue = (person, confidence)
        else:
            returnvalue = (None, None)
        return returnvalue
