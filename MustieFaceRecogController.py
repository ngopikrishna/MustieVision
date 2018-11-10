#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 13:07:11 2018

@author: N. Gopi Krishna
"""
import os
import queue
import shutil

import ProcessVideo as pv
import ImageStreamReader as isr
import TrainFaces as train
import AlignXformFaces as axf
import GenerateEmbeddings as ge
import StateVariables as sv


class MustieFaceRecogController:
    """ # MVC Pattern's Controller is here """

    def __init__(self, startup_folder):
        """ Constructor """
        self.__test_faces_webcam                   = 0
        self.__shape_predictor_68_datfile_location = startup_folder + "/3rdparty/models/dlib/shape_predictor_68_face_landmarks.dat"
        self.__torch_neuralnetwork_model_location  = startup_folder + "/3rdparty/models/openface/nn4.small2.v1.t7"
        self.__classifiermodel_picklefile_location = startup_folder + "/faces_database/classifier.pkl"
        self.__generated_embeddings_path           = startup_folder + "/faces_database"
        self.__lua_script_path                     = startup_folder + "/3rdparty/lua_scripts/main.lua"
        self.__training_images_dir                 = startup_folder + "/training_images"
        self.__aligned_images_dir                  = startup_folder + "/aligned_images"

        self.__image_buffer = queue.Queue() # A thread-safe queue in multi-threading environment
        self.__stop_processing = sv.VideoProcessingState()
        self.__image_stream_reader = isr.ImageStreamReader(self.__image_buffer,
                                                           self.__stop_processing,
                                                           self.__test_faces_webcam)
        self.__process_video = pv.ProcessVideo(self.__image_buffer,
                                               self.__stop_processing,
                                               self.__shape_predictor_68_datfile_location,
                                               self.__torch_neuralnetwork_model_location,
                                               self.__classifiermodel_picklefile_location)
        self.__train_classifier = train.TrainFaces(self.__generated_embeddings_path)
        self.__image_align = axf.AlignAndXformFaces(self.__shape_predictor_68_datfile_location,
                                            self.__training_images_dir,
                                            self.__aligned_images_dir)
        self.__generate_embeddings = ge.GenerateEmbeddings(self.__lua_script_path,
                                                           self.__generated_embeddings_path)


    def __cleanup_folders_for_training(self):
        """ """
        # We must cleanup the self.__aligned_images_folder folder, and self.__generated_embeddings_path folders
        shutil.rmtree(self.__aligned_images_dir, True)
        shutil.rmtree(self.__generated_embeddings_path, True)

        # Now create the empty folders and keep them ready
        os.mkdir(self.__aligned_images_dir)
        os.mkdir(self.__generated_embeddings_path)


    def train_model(self):
        """ ***** """
        self.stop_processing()
        self.__cleanup_folders_for_training()
        self.__image_align.alignMain()
        self.__generate_embeddings.generate_embeddings(self.__aligned_images_dir)
        self.__train_classifier.Go()

        # Remove the input and intermediate data
        shutil.rmtree(self.__aligned_images_dir, True)
        print("Training complete")

    def recognize_faces(self):
        """ ***** """
        self.__stop_processing.stop = False
        self.__image_stream_reader.start()
        self.__process_video.start()

    def stop_processing(self):
        """ ***** """
        self.__stop_processing.stop = True
        while(self.__image_stream_reader.is_alive() or self.__process_video.is_alive()):
            pass
