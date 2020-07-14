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
        self.all_models = {}

        self.pre_models = {}
        self.post_models = {}
        self.logger = Logger()
        self.filter_dir = '/nethome/jbang36/eva_jaeho/data/filters'
        self.catalog_filename = 'catalog.txt'


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



    def load(self, dataset_name, predicate_name):
        """

        :param dataset_name: name of dataset
        :param predicate_name: name of predicate
        :return:
        """

        full_path = os.path.join(self.filter_dir, dataset_name, predicate_name)
        ## instead of raising an error, let's return an error code
        if not os.path.isdir(full_path):
            self.logger.error(f"directory {full_path} does not exist!")
            return 1

        catalog_path = os.path.join(full_path, self.catalog_filename)
        if not os.path.isfile(catalog_path):
            self.logger.error(f"catalog does not exist, {catalog_path} we cannot load from this directory")
            return 1

        files = {}
        files[PRE] = []
        files[POST] = []
        self.logger.info("Inside load function...")
        with open(catalog_path, 'r') as file:

            for line in file:

                line_parsed = line.strip().split(',')
                model_name, pre_or_post_str = line_parsed
                model_filename = os.path.join(full_path, model_name + '.sav')
                if pre_or_post_str == PRE:
                    self.pre_models[model_name] = joblib.load(model_filename)
                elif pre_or_post_str == POST:
                    self.post_models[model_name] = joblib.load(model_filename)

        return 0




    def save(self, dataset_name, predicate_name):
        """

        :param dataset_name: name of dataset the models were trained with
        :param predicate_name: name of predicates the models are trained on
        :return: None
        """
        ## need to make sure the save_directory is a folder
        ## format will be
        """
        filters
         |_ dataset
            |_ predicate
                |_ catalog.txt
                |_ model_name.sav
                
        in catalog.txt we save which models are pre/post models
        """

        full_dir = os.path.join(self.filter_dir, dataset_name, predicate_name)
        os.makedirs(full_dir, exist_ok=True) ##create folders if it doesn't exist, otherwise we skip this step

        catalog_file = os.path.join(full_dir, self.catalog_filename)
        with open(catalog_file, 'w') as catalog:
            for model_name in self.pre_models:
                string = f"{model_name},PRE\n"
                catalog.write(string)
                model_save_full_path = os.path.join(full_dir, model_name+'.sav')
                joblib.dump(self.pre_models[model_name], model_save_full_path)
            for model_name in self.post_models:
                string = f"{model_name},POST\n"
                catalog.write(string)
                model_save_full_path = os.path.join(full_dir, model_name+'.sav')
                joblib.dump(self.post_models[model_name], model_save_full_path)

        return 0

