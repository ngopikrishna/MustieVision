#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:06:55 2018

@author: N. Gopi Krishna
"""
import openface

class FaceDetector:
    """ Module FaceDetector. Identifies the faces in the input image and returns
        1) their location in the image
        2) The 68 landmarks which define the face
    """
    __IMAGE_DIMENSIONS = 96
    __CUDA = False

    def __init__(self, shape_predictor_68_datfile_location, torch_neuralnetwork_model_location):
        """ """
        self.__align = openface.AlignDlib(shape_predictor_68_datfile_location)
        self.__net = openface.TorchNeuralNet(torch_neuralnetwork_model_location,
                                             imgDim=FaceDetector.__IMAGE_DIMENSIONS,
                                             cuda=FaceDetector.__CUDA)
        return

    def __getRep(self, image):
        """ """
        bbs = self.__align.getAllFaceBoundingBoxes(image)
        if not bbs:
            # No faces found. Return None
            return None

        face_reps = []
        for bb in bbs:
            alignedFace = self.__align.align(FaceDetector.__IMAGE_DIMENSIONS,
                                             image,
                                             bb,
                                             landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            if alignedFace is None:
                raise Exception("Unable to align image")

            rep = self.__net.forward(alignedFace)
            one_face_rep = {}
            one_face_rep['BoundingBox'] = bb
            one_face_rep['Landmarks'] = rep
            face_reps.append(one_face_rep)

        return face_reps


    def detect_faces(self, image):
        """ main workhorse """
        retvalue = self.__getRep(image)
        return retvalue
