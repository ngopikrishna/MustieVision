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
