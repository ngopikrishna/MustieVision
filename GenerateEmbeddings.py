"""
@author: N. Gopi Krishna
"""


import os


class GenerateEmbeddings:
    """
    Class to call lua script from python function to generate embeddings and save them to a csv file
    """

    def __init__(self, lua_script_path, generated_embeddings_path):
        """ """
        self.__lua_script_path = lua_script_path
        self.__generated_embeddings_path = generated_embeddings_path


    def generate_embeddings(self, aligned_images_dir):
        """ Main entry point """

        command = self.__lua_script_path + " -outDir " + self.__generated_embeddings_path +"/"
        command += " -data " + aligned_images_dir
        os.system(command)
        return
