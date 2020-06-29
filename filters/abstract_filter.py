"""
This file implements the inteface for filters

@Jaeho Bang
"""
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
import joblib
import os
from logger import Logger
from pp_constants import PRE, POST
import random


class FilterTemplate(metaclass = ABCMeta):

    def __init__(self):
        self.pre_models = {}
        self.post_models = {}
        self.all_models = {}
        self.logger = Logger()


    def __repr__(self):
        return 'Filters'


    @abstractmethod
    def train(self, X:np.ndarray, y:np.ndarray):
        """
        Train all preprocessing models (if needed)
        :param X: data
        :param y: labels
        :return: None
        """
        pass


    @abstractmethod
    def predict(self, X:np.ndarray, premodel_name:str, postmodel_name:str)->np.ndarray:
        """
        This function is using during inference step.
        The scenario would be
        1. We have finished training
        2. The query optimizer knows which filters / models within will be best for a given query
        3. The query optimizer orders the inference for a specific filter / preprocessing model / postprocessing model

        :param X: data
        :param premodel_name: name of preprocessing model to use
        :param postmodel_name: name of postprocessing model to use
        :return: resulting labels
        """
        pass


    @abstractmethod
    def getAllStats(self)->pd.DataFrame:
        """
        This function returns all the statistics acquired after training the preprocessing models and postprocessing models

        :return:
        """
        pass



    def load(self, full_path):
        """
        We load all the models from the txt file.

        :param txt_file:
        :return:
        """
        if not os.path.isfile(full_path):
            self.logger.error(f"directory / file {full_path} does not exist!")
            raise ValueError

        files = {}
        files[PRE] = []
        files[POST] = []
        save_directory = os.path.basename(full_path)

        with open(full_path, 'r') as file:
            for line in file:
                line_parsed = line.split(',')
                category, model_name, filename = line_parsed
                files[category].append( (model_name, os.path.join(save_directory, filename)) )

        ## now that we have files loaded, we need to actually fill in self.pre_models and self.post_models
        for key in files.keys():
            for model_name, full_model_path in files[key]:
                loaded_model = joblib.load(filename)
                if key == PRE:
                    self.pre_models[model_name] = loaded_model
                elif key == POST:
                    self.post_models[model_name] = loaded_model


        return None




    def save(self, full_path):
        """
        :param full_path: (string) save_directory + txt_file_name
        :return: None
        """
        ## need to make sure the save_directory is a folder
        ## also we need to create a txt file that saves all the models to load and where
        ### steps: 1. we save the txt file, then we save the pre/post models
        ### the format for saving the pre, post models are as follows:
        ### pre, model_name, model_save_file.sav
        ### post, model_name, model_save_file.sav
        save_directory = os.path.basename(full_path)
        random_num = random.randint(0, 10000)

        with open(full_path) as txt_save_file:
            for model_name in self.pre_models:
                model_save_full_path = os.path.join(save_directory, model_name+random_num+'.sav')
                txt_save_file.write(f"{PRE},{model_name},{model_save_full_path}")
                joblib.dump(self.pre_models[model_name], model_save_full_path)

            for model_name in self.post_models:
                model_save_full_path = os.path.join(save_directory, model_name + random_num + '.sav')
                txt_save_file.write(f"{POST},{model_name},{model_save_full_path}")
                joblib.dump(self.pre_models[model_name], model_save_full_path)

        return

