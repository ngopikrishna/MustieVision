# MustieVision
Computer Vision for MUSTie robot


This repository is for computer vision for the MUSTie Robot.

As of now, the following functionalities have been built
1) Face Recognition using Openface.


Tech used so far
    *) Python 3
    *) Open CV 3.4.1
    *) Openface

Coming soon...
2) Solve "Spot the Differences" puzzle





1) Face Recognition using Openface

High level overview

    1) A video stream is captured from Webcam.
    2) The frames are placed ina  queue buffer.
    3) The frames are processed for
        (a) detecting faces
        (b) recognizing the detected faces with a confidence threshold
    4) Support Vector Classificiation (SVC) is used for classifying faces
    5) GPU Ready. Toggle the __CUDA variable to run on a GPU

    6)  a) TrainFaces will train new faces in the "training_images" folder.
        b) Each face must be in a sub folder of its own
        c) One faceimage should have only a single face.
    7) In Training and recognition, Openface's "nn4.small2.v1.t7" model is used with OUTER_EYES_AND_NOSE landmarks

Steps 1&2 run in a separate thread.
Step 3 runs in a separate thread.
Step 6 is a completely different activity. It runs as a batch process and cannot happen in parallel with recognition (i.e 1,2 & 3)



Running the code
-----------------------

Prerequisite packages
-----------------------
    *) OpenCV
    *) Openface
    *)

Execution Instructions
-----------------------
*) at command prompt, run the below
      python3 MustieFaceRecognition.py
*) For recognizing faces, click on "Recognize Faces"
*) For adding new faces to the database, follow below steps.
    *) In the "training_images" folder, create subfolders.
    *) The subfolder shall be one per person. The subfolder name is used as the string seen in the rectanglular box during recognition.
    *) There can be any number of images of a person in the subfolder. They should be (a) JPGs (b) should NOT contain any other persons faces and (c) both eyes should be visible.
    *) Now run the program and click "Train new faces".



Notes
-----
*) If you are getting too many false positives then play with the value of __CONFIDENCE_LEVEL parameter in FaceRecognizer.py.
*) If you want to run the program on GPU then turn the __CUDA value to True in FaceDetector.py. This will significantly speedup the runtime performance
