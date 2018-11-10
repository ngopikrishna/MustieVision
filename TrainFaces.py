"""
@author: N. Gopi Krishna
Class to generate classifier.pkl file from the generated embeddings
"""
import warnings
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

np.set_printoptions(precision=2)

class TrainFaces:
    """
        Main module for training the model/pickle file with new faces
    """
    def __init__(self, generated_embeddings_path):
        """ Constructor """
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        self.__generated_embeddings_path = generated_embeddings_path

    def Go(self):
        """ Main Entry point"""

        print("Loading embeddings.")
        fname = self.__generated_embeddings_path + "/labels.csv"
        print("fname : {}", fname)
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]


        labels = [os.path.split(os.path.dirname(label))[1] for label in labels]
        print("result :", labels)
        fname = self.__generated_embeddings_path + "/reps.csv"
        embeddings = pd.read_csv(fname, header=None).as_matrix()
        # embeddings.values
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))
        clf = SVC(C=1, kernel='linear', probability=True)
        clf.fit(embeddings, labelsNum)

        fName = "{}/classifier.pkl".format(self.__generated_embeddings_path)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'wb') as f:
            pickle.dump((le, clf), f)
        return
