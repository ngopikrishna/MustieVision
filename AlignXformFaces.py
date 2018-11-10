#!/usr/bin/env python3

"""
@author: N. Gopi Krishna

    Performs Affine Transformations to the face. Each image is opened, affine transformations are performed
    to ensure that eyes and mouth are roughly in the center. Each image is read, transformations are applied
    and stored into a separate directory. If eyes are not visible in the image then transformation generally
    fails and image is ignored.

    No fancy transformations are performed on the images. Strictly Affine.
"""

import os
import random
import warnings
import numpy as np
import openface
import openface.helper
from openface.data import iterImgs
import cv2

np.set_printoptions(precision=2)


class AlignAndXformFaces:
    """
    Performs Affine Transformations to the face.
    """
    __Image_dim = 96

    def __init__(self, shape_predictor_68_datfile_location, training_dir, align_dir):
        """ ***** """
        self.__align = openface.AlignDlib(shape_predictor_68_datfile_location)
        self.__traininig_images_dir = training_dir
        self.__aligned_images_dir = align_dir

    def alignMain(self):
        """ ***** """
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)


        openface.helper.mkdirP(self.__aligned_images_dir)
        imgs = list(iterImgs(self.__traininig_images_dir))
        random.shuffle(imgs)

        for imgObject in imgs:
            out_dir = os.path.join(self.__aligned_images_dir, imgObject.cls)
            openface.helper.mkdirP(out_dir)
            outputPrefix = os.path.join(out_dir, imgObject.name)
            imgName = outputPrefix + ".png"

            if os.path.isfile(imgName):
                print("  {} Already found, skipping.".format(imgName))
            else:
                rgb = imgObject.getRGB()
                if rgb is None:
                    print(" {} Unable to load.".format(imgName))
                    outRgb = None
                else:
                    outRgb = self.__align.align(AlignAndXformFaces.__Image_dim,
                                                rgb,
                                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE,
                                                skipMulti=True)
                    if outRgb is None:
                        print("  {} Unable to align.".format(imgName))
                if outRgb is not None:
                    print("  {} Writing aligned file to disk.".format(imgName))
                    outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(imgName, outBgr)
        return
